import os
import torch
import glob
import numpy as np
import pandas as pd
from PIL import Image
import albumentations as A
from torch.utils.data import Dataset
from albumentations.pytorch import ToTensorV2
       

class CollagenDataset(Dataset):
    def __init__(self, args, image_df, transform=None, include='stroma'):

        self.task = args.task
        self.root = args.image_root

        if self.task == 'survival':
            image_df = image_df.dropna(subset=['Overall.Survival.Interval', 'Overall.Survival.Months', 'Overall.Survival.Status'])
        
        if include == 'stroma':
            image_df = image_df[image_df['Tumour_Stroma'] == 'Stroma']
        elif include == 'tumour':
            image_df = image_df[image_df['Tumour_Stroma'] == 'Tumour']
        elif include == 'both':
            pass
        else:
            raise ValueError("include must be one of 'stroma', 'tumour', or 'both'.")

        task2key = {
            'survival': 'Overall.Survival.Interval',
            'grade': 'T.Grade',
            'size': 'T.Size',
            'vascular_invasion': 'T.VascularInvasion',
            'lymph_invasion': 'T.LymphInvasion',
            'node_status': 'Node.Status',
            'er': 'T.ER_Status',
            'pr': 'T.PR_Status',
            'her2': 'T.HER2',
        }
        self.df = image_df.copy()
        self.df['label'] = image_df[task2key[self.task]]
        
        # Map labels to start from 0
        unique_labels = sorted(self.df['label'].unique())  # Get sorted unique labels
        self.label_mapping = {label: idx for idx, label in enumerate(unique_labels)}  # Create mapping
        self.df['label'] = self.df['label'].map(self.label_mapping)  # Apply mapping

        self.n_classes = len(self.df['label'].unique())
        if args.surv_loss.lower() == 'cox':
            self.n_classes = 1 # Cox regression predicts a single risk factor
        self.n_images = len(self.df)
        self.transform = transform
            
    
    def __len__(self):
        return self.n_images
    
    def __getitem__(self, index):
        row = self.df.iloc[index]
        filename = row['Filename']

        path_r = os.path.join(self.root, f"{row['Folder']}_R", filename)
        path_g = os.path.join(self.root, f"{row['Folder']}_G", filename)
        path_b = os.path.join(self.root, f"{row['Folder']}_B", filename)

        # Load each grayscale image
        image_r = Image.open(path_r).convert('L')  # Convert to grayscale
        image_g = Image.open(path_g).convert('L')  # Convert to grayscale
        image_b = Image.open(path_b).convert('L')  # Convert to grayscale

        # Convert images to numpy arrays
        image_r = np.array(image_r)
        image_g = np.array(image_g)
        image_b = np.array(image_b)

        # Stack grayscale images to form a 3-channel image
        image = np.stack([image_r, image_g, image_b], axis=-1)  # Shape: (H, W, 3)

        # Apply transformations (if any)
        if self.transform:
            image = self.transform(image=image)['image']
        else:
            image = torch.from_numpy(image).permute(2, 0, 1)  # Convert HWC to CHW format
        
        # Convert labels and other data to tensors
        label = torch.tensor(row['label'], dtype=torch.long)

        base_dict = {
            'image': image,
            'filename': filename,
            'patient_id': row['BBNumber'],
            'label': label,
        }

        if self.task == 'survival':
            base_dict['c'] = torch.tensor(1 - row['Overall.Survival.Status'], dtype=torch.float)
            base_dict['survival'] = torch.tensor(row['Overall.Survival.Months'], dtype=torch.float)
            base_dict['duration'] = torch.tensor(30 * row['Overall.Survival.Months'], dtype=torch.float)
            base_dict['event'] = torch.tensor(row['Overall.Survival.Status'], dtype=torch.float)

        return base_dict
