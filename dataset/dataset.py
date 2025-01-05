import os
import torch
import glob
import numpy as np
import pandas as pd
from PIL import Image
import albumentations as A
from torch.utils.data import Dataset
from albumentations.pytorch import ToTensorV2


class Slide(object):
    # for each wsi
    def __init__(self, root_path, wsi_id, label, dim=512):
        self.label = label
        feature_path = os.path.join(root_path, wsi_id, 'features.pt')
        if os.path.exists(feature_path):
            self.feature = torch.load(feature_path, map_location=lambda storage, loc: storage)
        else:
            print(feature_path + ' not exists')
            self.feature = torch.zeros(1, dim)
    

class featureDataset(Dataset):
    def __init__(self, args, wsi_df, patient_idx, new_label=True):

        self.root = self._retrieve_data_path(args)

        # special cases
        if args.calibrate:
            wsi_df = wsi_df[wsi_df['Grading annotation'] == 'Yes']
        
        if self.root.endswith('QMH_UNI_20x_512*512_noOverlap_Normed'):
            wsi_df = wsi_df[wsi_df['Slides.ID'] != 'CHS032-WSI02']

        self.wsi_info = wsi_df[wsi_df['Case.ID'].isin(patient_idx)]
        # wsi label
        old_labels = self.wsi_info['Grade.Revised'].values
        class2new = {'0': 0, '1': 1, '2': 2, '3': 2, 'D': 3}
        self.patient_id = self.wsi_info['Case.ID'].values
        self.wsi_labels = [class2new[k] for k in old_labels] if new_label else old_labels
        self.num_labels = len(set(self.wsi_labels))
        # wsi index
        self.wsi_ids = self.wsi_info['Slides.ID'].values
        self.num_wsi= len(self.wsi_ids)

        # all wsi(patches) + label
        self.wsi_list = [Slide(self.root, self.wsi_ids[i], self.wsi_labels[i]) for i in range(self.num_wsi)]
            
    def _retrieve_data_path(self, args):
        wsi_root = args.wsi_root
        tags = [
            'noOverlap' if not args.patch_overlap else f"{args.patch_overlap}overlap",
            'calibrate' if args.calibrate else '',
            'Filtered' if args.filter else '',
            'Normed' if args.norm else '',
            'Aug' if args.aug else ''
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
        return slide.feature, slide.labe
