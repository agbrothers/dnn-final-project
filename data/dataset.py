import re
import cv2
import glob
import pydicom
import numpy as np
from PIL import Image
from torch.utils.data import Dataset


CONDITIONS = [
    'spinal_canal_stenosis', 
    'left_neural_foraminal_narrowing', 
    'right_neural_foraminal_narrowing',
    'left_subarticular_stenosis',
    'right_subarticular_stenosis'
]
LEVELS = [
    'l1_l2',
    'l2_l3',
    'l3_l4',
    'l4_l5',
    'l5_s1',
]
LABELS = {
    "Normal/Mild": 0,
    "Moderate": 1,
    "Severe": 2,
}
IMG_SIZE = [512, 512]
IN_CHANS = 30


def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    return [ atoi(c) for c in re.split(r'(\d+)', text) ]

def save_img(img: np.ndarray, name="sample"):
    Image.fromarray(img.astype(np.uint8)).save(f"{name}.png")
    return


class RSNA24TrainDataset(Dataset):
    def __init__(self, df, labels, study_ids, dtype=np.float32, phase='train', transform=None):
        self.df = df #.iloc[indices].reset_index(drop=True)
        self.labels = labels
        self.study_ids = study_ids
        self.transform = transform
        self.dtype = dtype
        self.phase = phase
    
    def __len__(self):
        return len(self.study_ids)

    
    def get_img_paths(self, study_id, series_desc):
        pdf = self.df[self.df['study_id']==study_id]
        pdf_ = pdf[pdf['series_description']==series_desc]
        allimgs = []
        for i, row in pdf_.iterrows():
            pimgs = glob.glob(f'data/train_images/{study_id}/{row["series_id"]}/*.dcm')
            pimgs = sorted(pimgs, key=natural_keys)
            allimgs.extend(pimgs)
            
        return allimgs
    
    def read_dcm_ret_arr(self, src_path):
        dicom_data = pydicom.dcmread(src_path)
        image = dicom_data.pixel_array
        image = (image - image.min()) / (image.max() - image.min() + 1e-6) * 255
        img = cv2.resize(image, (IMG_SIZE[0], IMG_SIZE[1]),interpolation=cv2.INTER_CUBIC)
        assert img.shape==(IMG_SIZE[0], IMG_SIZE[1])
        return img

    def __getitem__(self, idx):
        x = np.zeros((IMG_SIZE[0], IMG_SIZE[1], IN_CHANS), dtype=np.uint8)
        t = self.labels.iloc[idx]
        st_id = int(t['study_id'])
        label = t.iloc[1:].apply(lambda x: LABELS[x]).values.astype(np.int8)
        
        # Sagittal T1
        allimgs_st1 = self.get_img_paths(st_id, 'Sagittal T1')
        if len(allimgs_st1)==0:
            print(st_id, ': Sagittal T1, has no images')
        
        else:
            step = len(allimgs_st1) / 10.0
            st = len(allimgs_st1)/2.0 - 4.0*step
            end = len(allimgs_st1)+0.0001
            for j, i in enumerate(np.arange(st, end, step)):
                try:
                    ind2 = max(0, int((i-0.5001).round()))
                    img = self.read_dcm_ret_arr(allimgs_st1[ind2])
                    x[..., j] = img.astype(np.uint8)
                except:
                    print(f'failed to load on {st_id}, Sagittal T1')
                    pass
            
        # Sagittal T2/STIR
        allimgs_st2 = self.get_img_paths(st_id, 'Sagittal T2/STIR')
        if len(allimgs_st2)==0:
            print(st_id, ': Sagittal T2/STIR, has no images')
            
        else:
            step = len(allimgs_st2) / 10.0
            st = len(allimgs_st2)/2.0 - 4.0*step
            end = len(allimgs_st2)+0.0001
            for j, i in enumerate(np.arange(st, end, step)):
                try:
                    ind2 = max(0, int((i-0.5001).round()))
                    img = self.read_dcm_ret_arr(allimgs_st2[ind2])
                    x[..., j+10] = img.astype(np.uint8)
                except:
                    print(f'failed to load on {st_id}, Sagittal T2/STIR')
                    pass

        # Axial T2
        allimgs_at2 = self.get_img_paths(st_id, 'Axial T2')
        if len(allimgs_at2)==0:
            print(st_id, ': Axial T2, has no images')
            
        else:
            step = len(allimgs_at2) / 10.0
            st = len(allimgs_at2)/2.0 - 4.0*step
            end = len(allimgs_at2)+0.0001

            for j, i in enumerate(np.arange(st, end, step)):
                try:
                    ind2 = max(0, int((i-0.5001).round()))
                    img = self.read_dcm_ret_arr(allimgs_at2[ind2])
                    x[..., j+20] = img.astype(np.uint8)
                except:
                    print(f'failed to load on {st_id}, Axial T2')
                    pass  

        if self.transform is not None:
            x = self.transform(image=x)['image']

        x = x.astype(self.dtype)
        x = x.transpose(2, 0, 1)
                
        return x, label
    

class RSNA24TestDataset(Dataset):
    def __init__(self, df, study_ids, phase='test', transform=None):
        self.df = df
        self.study_ids = study_ids
        self.transform = transform
        self.phase = phase
    
    def __len__(self):
        return len(self.study_ids)
    
    def get_img_paths(self, study_id, series_desc):
        pdf = self.df[self.df['study_id']==study_id]
        pdf_ = pdf[pdf['series_description']==series_desc]
        allimgs = []
        for i, row in pdf_.iterrows():
            pimgs = glob.glob(f'data/test_images/{study_id}/{row["series_id"]}/*.dcm')
            pimgs = sorted(pimgs, key=natural_keys)
            allimgs.extend(pimgs)
            
        return allimgs
    
    def read_dcm_ret_arr(self, src_path):
        dicom_data = pydicom.dcmread(src_path)
        image = dicom_data.pixel_array
        image = (image - image.min()) / (image.max() - image.min() + 1e-6) * 255
        img = cv2.resize(image, (IMG_SIZE[0], IMG_SIZE[1]),interpolation=cv2.INTER_CUBIC)
        assert img.shape==(IMG_SIZE[0], IMG_SIZE[1])
        return img

    def __getitem__(self, idx):
        x = np.zeros((IMG_SIZE[0], IMG_SIZE[1], IN_CHANS), dtype=np.uint8)
        st_id = self.study_ids[idx]        
        
        # Sagittal T1
        allimgs_st1 = self.get_img_paths(st_id, 'Sagittal T1')
        if len(allimgs_st1)==0:
            print(st_id, ': Sagittal T1, has no images')
        
        else:
            step = len(allimgs_st1) / 10.0
            st = len(allimgs_st1)/2.0 - 4.0*step
            end = len(allimgs_st1)+0.0001
            for j, i in enumerate(np.arange(st, end, step)):
                try:
                    ind2 = max(0, int((i-0.5001).round()))
                    img = self.read_dcm_ret_arr(allimgs_st1[ind2])
                    x[..., j] = img.astype(np.uint8)
                except:
                    print(f'failed to load on {st_id}, Sagittal T1')
                    pass
            
        # Sagittal T2/STIR
        allimgs_st2 = self.get_img_paths(st_id, 'Sagittal T2/STIR')
        if len(allimgs_st2)==0:
            print(st_id, ': Sagittal T2/STIR, has no images')
            
        else:
            step = len(allimgs_st2) / 10.0
            st = len(allimgs_st2)/2.0 - 4.0*step
            end = len(allimgs_st2)+0.0001
            for j, i in enumerate(np.arange(st, end, step)):
                try:
                    ind2 = max(0, int((i-0.5001).round()))
                    img = self.read_dcm_ret_arr(allimgs_st2[ind2])
                    x[..., j+10] = img.astype(np.uint8)
                except:
                    print(f'failed to load on {st_id}, Sagittal T2/STIR')
                    pass
            
        # Axial T2
        allimgs_at2 = self.get_img_paths(st_id, 'Axial T2')
        if len(allimgs_at2)==0:
            print(st_id, ': Axial T2, has no images')
            
        else:
            step = len(allimgs_at2) / 10.0
            st = len(allimgs_at2)/2.0 - 4.0*step
            end = len(allimgs_at2)+0.0001

            for j, i in enumerate(np.arange(st, end, step)):
                try:
                    ind2 = max(0, int((i-0.5001).round()))
                    img = self.read_dcm_ret_arr(allimgs_at2[ind2])
                    x[..., j+20] = img.astype(np.uint8)
                except:
                    print(f'failed to load on {st_id}, Axial T2')
                    pass  
            
            
        if self.transform is not None:
            x = self.transform(image=x)['image']

        x = x.transpose(2, 0, 1)
                
        return x, str(st_id)


