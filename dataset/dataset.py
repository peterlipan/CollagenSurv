import os
import torch
import glob
import numpy as np
import pandas as pd
from PIL import Image
import albumentations as A
from torch.utils.data import Dataset
from albumentations.pytorch import ToTensorV2


class Slide:
    def __init__(self, root: str, row: pd.Series, d_x=512):
        # root: root path to the WSI samples
        # row: a row in the WSI information dataframe
        x_path = os.path.join(root, row['Slides.ID'], 'features.pt')
        adj_path = os.path.join(root, row['Slides.ID'], 'adj_s.pt')
        self.label = row['Cls.Label']
        self.surv_label = row['survival_interval']
        self.event_time = row['Overall.Survival.Months'] * 30
        self.c = 0 if row['Death (Yes or No)']=='Yes' else 1
        self.dead = 1 if row['Death (Yes or No)']=='Yes' else 0
        self.survival = row['Overall.Survival.Months']
        self.grade = row['Tumor.Grade']
        self.id = row['Slides.ID']
        
        if os.path.exists(x_path):
            self.x = torch.load(x_path, map_location=lambda storage, loc: storage)
        else:
            print(x_path + ' not exists')
            self.x = torch.zeros(1, d_x)
        if os.path.exists(adj_path):
            self.adj = torch.load(adj_path, map_location=lambda storage, loc: storage)
        else:
            print(adj_path + ' not exists')
            self.adj = torch.zeros(1, 1)
    
    def _to_dict(self):
        return {
            'x': self.x,
            'adj': self.adj,
            'label': self.label,
            'surv_label': self.surv_label,
            'event_time': self.event_time,
            'c': self.c,
            'dead': self.dead,
            'survival': self.survival,
            'grade': self.grade,
            'id': self.id
        }
        
        
    

class featureDataset(Dataset):
    def __init__(self, args, wsi_df, patient_idx, new_label=True):

        self.root = self._retrieve_data_path(args)
        # special cases
        if args.calibrate:
            wsi_df = wsi_df[wsi_df['Grading annotation'] == 'Yes']
        
        if self.root.endswith('QMH_UNI_20x_512*512_noOverlap_Normed'):
            wsi_df = wsi_df[wsi_df['Slides.ID'] != 'CHS032-WSI02']

        self.wsi_info = wsi_df[wsi_df['Case.ID'].isin(patient_idx)]

        # Update the grading
        labels = self.wsi_info['Grade.Revised'].values
        grade2num = {'0': 0, '1': 1, '2': 2, '3': 3, 'D': 4}
        grade_nums = [grade2num[str(l)] for l in labels]
        if new_label:
            class2new = {'0': 0, '1': 1, '2': 2, '3': 2, 'D': 3}
            labels = [class2new[str(l)] for l in labels]
        
        self.wsi_info['Tumor.Grade'] = grade_nums
        self.wsi_info['Cls.Label'] = labels
        # classification task or survival task
        self.num_labels = len(set(labels)) if not args.task == 'survival' else args.surv_classes

        self.num_wsi= self.wsi_info.shape[0]

        self.wsi_list = [Slide(self.root, self.wsi_info.iloc[i]) for i in range(self.num_wsi)]
            
    def _retrieve_data_path(self, args):
        wsi_root = args.wsi_root
        tags = [
            'noOverlap' if not args.patch_overlap else f"{args.patch_overlap}overlap",
            'calibrate' if args.calibrate else '',
            'Filtered' if args.filter else '',
            'Normed' if args.stain_norm else '',
            'Aug' if args.augmentation else ''
        ]
        
        tags = '_'.join([t for t in tags if t])

        folder = f"QMH_{args.extractor}_{args.magnification}x_{args.patch_size}*{args.patch_size}"
        if tags:
            folder += f"_{tags}"

        path = os.path.join(wsi_root, folder)

        return path
    
    def __len__(self):
        return self.num_wsi
    
    def __getitem__(self, index):
        slide = self.wsi_list[index]
        return slide._to_dict()
    