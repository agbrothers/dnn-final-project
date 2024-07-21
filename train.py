import os
import sys
import math
import pandas as pd
from tqdm import tqdm
import albumentations as A
from collections import OrderedDict
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
import torch.nn as nn
import torch
from transformers import get_cosine_schedule_with_warmup

from data.dataset import RSNA24TrainDataset, IMG_SIZE, IN_CHANS
from models.rsna import RSNA24Model


# N_WORKERS = 0 #1 #os.cpu_count() // 2
N_WORKERS = (3*os.cpu_count()) // 4
USE_AMP = True
SEED = 42
N_LABELS = 25
N_CLASSES = 3 * N_LABELS
N_FOLDS = 5
MODEL_NAME = "densenet201"
MODEL_PATH = None #f"models/checkpoints/{MODEL_NAME}"
CHECKPOINT_PATH = f"models/checkpoints/{MODEL_NAME}"
EPOCHS = 100
EARLY_STOPPING_EPOCH = EPOCHS
OUTPUT_DIR = "results"
MAX_GRAD_NORM = None
GRAD_ACC = 2
TGT_BATCH_SIZE = 4
BATCH_SIZE = TGT_BATCH_SIZE // GRAD_ACC
LR = 2e-4 * TGT_BATCH_SIZE / 32
WD = 1e-2

rd = 'data'
# rd = '/kaggle/input/rsna-2024-lumbar-spine-degenerative-classification'



def train():
    ## LOAD METADATA
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    # df = pd.read_csv(f'{rd}/train.csv')
    df = pd.read_csv(f'{rd}/train_series_descriptions.csv')
    labels = pd.read_csv(f'{rd}/train.csv')
    # study_ids = list(df['study_id'].unique())
    # sample_sub = pd.read_csv(f'{rd}/sample_submission.csv')
    # LABELS = list(sample_sub.columns[1:])

    ## LOAD DATASET
    transforms_train = A.Compose([
        A.Resize(IMG_SIZE[0], IMG_SIZE[1]),
        A.Normalize(mean=0.5, std=0.5)
    ])

    ## DATASETS
    y_train, y_val = train_test_split(labels, test_size=0.33, random_state=42)

    train_ids = list(y_train["study_id"])
    train_ds = RSNA24TrainDataset(df, labels=y_train, study_ids=train_ids, transform=transforms_train)

    val_ids = list(y_val["study_id"])
    val_ds = RSNA24TrainDataset(df, labels=y_val, study_ids=val_ids, transform=transforms_train)

    ## DATA LOADERS
    train_dl = DataLoader(
        train_ds, 
        batch_size=BATCH_SIZE, 
        shuffle=False,
        num_workers=N_WORKERS,
        pin_memory=True,
        drop_last=False
    )
    val_dl = DataLoader(
        val_ds, 
        batch_size=BATCH_SIZE, 
        shuffle=False,
        num_workers=N_WORKERS,
        pin_memory=True,
        drop_last=False
    )

    ## EXAMINE SAMPLE
    # x = train_dl.dataset[0]
    # v = val_dl.dataset[0]

    ## INFERENCE
    autocast = torch.cuda.amp.autocast(enabled=USE_AMP, dtype=torch.half)
    scaler = torch.cuda.amp.GradScaler(enabled=USE_AMP, init_scale=4096)

    val_losses = []
    train_losses = []


    ## INITIALIZE MODEL
    model = RSNA24Model(MODEL_NAME, IN_CHANS, N_CLASSES, pretrained_path=MODEL_PATH)
    fname = f'{CHECKPOINT_PATH}_best.pt'
    if os.path.exists(fname):
        state_dict = torch.load(fname)
        # Ignore the classifier weights when loading
        state_dict.pop('model.classifier.weight', None)
        state_dict.pop('model.classifier.bias', None)
        model.load_state_dict(state_dict, strict=False)
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    warmup_steps = EPOCHS / 10 * len(train_dl) // GRAD_ACC
    num_total_steps = EPOCHS * len(train_dl) // GRAD_ACC
    num_cycles = 0.475
    scheduler = get_cosine_schedule_with_warmup(optimizer,
                                                num_warmup_steps=warmup_steps,
                                                num_training_steps=num_total_steps,
                                                num_cycles=num_cycles)

    weights = torch.tensor([1.0, 2.0, 4.0])
    criterion = nn.CrossEntropyLoss(weight=weights.to(device))

    best_loss = 2.0 #1.2
    es_step = 0

    for epoch in range(EPOCHS): 
        print(f'start epoch {epoch}')
        model.train()  # Set the model to training mode
        total_loss = 0.0
        with tqdm(train_dl, leave=True) as pbar:
            optimizer.zero_grad()
            for idx, (x, t) in enumerate(pbar):
                x = x.to(device)
                t = t.to(device)

                with autocast:
                    y = model(x)
                    loss = 0.0  # Initialize loss as a scalar
                    for col in range(N_LABELS):
                        pred = y[:, col * 3:col * 3 + 3]
                        gt = t[:, col]
                        loss += criterion(pred, gt) / N_LABELS
                        # valid_mask = (gt >= 0) & (gt < 3)
                        # if valid_mask.any():
                        #     loss += criterion(pred[valid_mask], gt[valid_mask]) / N_LABELS
                            # test = torch.zeros((2,3)).to(device)
                            # criterion(pred[valid_mask], test) / N_LABELS

                if isinstance(loss, torch.Tensor):
                    if not math.isfinite(loss.item()):
                        print(f"Loss is {loss.item()}, stopping training")
                        sys.exit(1)

                    total_loss += loss.item()  # Accumulate total loss

                    if GRAD_ACC > 1:
                        loss = loss / GRAD_ACC

                    pbar.set_postfix(
                        OrderedDict(
                            loss=f'{loss.item() * GRAD_ACC:.6f}',
                            lr=f'{optimizer.param_groups[0]["lr"]:.3e}'
                        )
                    )
                    scaler.scale(loss).backward()  # Backward pass

                    torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM or 1e9)

                    if (idx + 1) % GRAD_ACC == 0:
                        scaler.step(optimizer)
                        scaler.update()
                        optimizer.zero_grad()
                        if scheduler is not None:
                            scheduler.step()

            train_loss = total_loss / len(train_dl)
            print(f'train_loss:{train_loss:.6f}')
            train_losses.append(train_loss)
            total_loss = 0.0

            model.eval()  # Set the model to evaluation mode
            with tqdm(val_dl, leave=True) as pbar:
                with torch.no_grad():
                    for idx, (x, t) in enumerate(pbar):
                        x = x.to(device)
                        t = t.to(device)
                        with autocast:
                            y = model(x)
                            loss = 0.0  # Initialize loss as a scalar
                            for col in range(N_LABELS):
                                pred = y[:, col * 3:col * 3 + 3]
                                gt = t[:, col]
                                loss += criterion(pred, gt) / N_LABELS
                                # valid_mask = (gt >= 0) & (gt < 3)
                                # if valid_mask.any():
                                #     loss += criterion(pred[valid_mask], gt[valid_mask]) / N_LABELS
                            if isinstance(loss, torch.Tensor):
                                total_loss += loss.item()  # Accumulate total loss

            val_loss = total_loss / len(val_dl)
            print(f'val_loss:{val_loss:.6f}')
            val_losses.append(val_loss)
            if val_loss < best_loss:
                if device != 'cuda:0':
                    model.to('cuda:0')
                print(f'epoch:{epoch}, best weighted_logloss updated from {best_loss:.6f} to {val_loss:.6f}')
                best_loss = val_loss
                fname = f'{CHECKPOINT_PATH}_loss_{best_loss:.4f}.pt'
                torch.save(model.state_dict(), fname)
                print(f'{fname} is saved')
                es_step = 0
                if device != 'cuda:0':
                    model.to(device)
            else:
                es_step += 1
                if es_step >= EARLY_STOPPING_EPOCH:
                    print('early stopping')
                    break
    
    return


if __name__ == "__main__":
    train()
