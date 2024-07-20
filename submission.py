import re
import os
import cv2
import glob
import pydicom
import numpy as np
import pandas as pd
from tqdm import tqdm
from PIL import Image
import albumentations as A
from torch.utils.data import DataLoader
import torch

from data.dataset import RSNA24TestDataset, CONDITIONS, LEVELS, IMG_SIZE


N_WORKERS = os.cpu_count() // 2
USE_AMP = True
SEED = 8620
N_LABELS = 25
N_CLASSES = 3 * N_LABELS
N_FOLDS = 5
MODEL_NAME = "tf_efficientnet_b3.ns_jft_in1k"
BATCH_SIZE = 1
# rd = '/kaggle/input/rsna-2024-lumbar-spine-degenerative-classification'
rd = 'data'


# device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
df = pd.read_csv(f'{rd}/test_series_descriptions.csv')
study_ids = list(df['study_id'].unique())
sample_sub = pd.read_csv(f'{rd}/sample_submission.csv')
LABELS = list(sample_sub.columns[1:])

## LOAD DATASET
transforms_test = A.Compose([
    A.Resize(IMG_SIZE[0], IMG_SIZE[1]),
    A.Normalize(mean=0.5, std=0.5)
])

test_ds = RSNA24TestDataset(df, study_ids, transform=transforms_test)
test_dl = DataLoader(
    test_ds, 
    batch_size=1, 
    shuffle=False,
    num_workers=N_WORKERS,
    pin_memory=True,
    drop_last=False
)

## INFERENCE
autocast = torch.cuda.amp.autocast(enabled=USE_AMP, dtype=torch.half)
y_preds = []
row_names = []

with tqdm(test_dl, leave=True) as pbar:
    with torch.no_grad():
        for idx, (x, si) in enumerate(pbar):
            # x = x.to(device)
            pred_per_study = np.zeros((25, 3))

            for cond in CONDITIONS:
                for level in LEVELS:
                    row_names.append(si[0] + '_' + cond + '_' + level)
            y_preds.append(pred_per_study)

            # with autocast:
            #     for m in models:
            #         y = m(x)[0]
            #         for col in range(N_LABELS):
            #             pred = y[col*3:col*3+3]
            #             y_pred = pred.float().softmax(0).cpu().numpy()
            #             pred_per_study[col] += y_pred / len(models)
            #     y_preds.append(pred_per_study)

y_preds = np.concatenate(y_preds, axis=0)

## JUST SET PROBABILITIES INVERSELY PROPORTIONAL TO WEIGHTING
y_preds[:,0] = 4
y_preds[:,1] = 2
y_preds[:,2] = 1
y_preds = np.exp(y_preds)/np.exp(y_preds).sum(axis=1, keepdims=True)

sub = pd.DataFrame()
sub['row_id'] = row_names
sub[LABELS] = y_preds
sub.head(25)

sub.to_csv('submission.csv', index=False)

pd.read_csv('submission.csv').head()
