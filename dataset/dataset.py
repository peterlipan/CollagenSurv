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
    def __init__(self, args, image_df, transform=None):

        self.task = args.task
        self.root = args.image_root
        if args.task == 'survival':
            self.image_df = image_df.dropna(subset=['Overall.Survival.Months', 'Overall.Survival.Status'])
            self.labels = self.image_df['Overall.Survival.Interval'].values
            self.n_classes = len(np.unique(self.labels))
            self.n_images = len(self.image_df)
        else:
            raise ValueError("Unsupported task: {}".format(args.task))

        self.transform = transform
            
    
    def __len__(self):
        return self.n_images
    
    def __getitem__(self, index):
        row = self.image_df.iloc[index]
        filename = row['Filename']
        path1 = os.path.join(self.root, row['Folder'], filename)
        filename_png = os.path.splitext(row['Filename'])[0] + '.png'
        path2 = os.path.join(self.root, f"{row['Folder']}_HDM", filename_png)
        path3 = os.path.join(self.root, f"{row['Folder']}_Masks", filename_png)

        label = row['Overall.Survival.Interval']
        event_time = row['Overall.Survival.Months'] * 30
        c = 1 - row['Overall.Survival.Status']
        dead = row['Overall.Survival.Status']
        survival = row['Overall.Survival.Months']
        patient_id = row['BBNumber']

        # Load each grayscale image
        image1 = Image.open(path1).convert('L')  # Convert to grayscale
        image2 = Image.open(path2).convert('L')  # Convert to grayscale
        image3 = Image.open(path3).convert('L')  # Convert to grayscale

        # Convert images to numpy arrays
        image1 = np.array(image1)
        image2 = np.array(image2)
        image3 = np.array(image3)

        # Stack grayscale images to form a 3-channel image
        image = np.stack([image1, image2, image3], axis=-1)  # Shape: (H, W, 3)

        # Apply transformations (if any)
        if self.transform:
            image = self.transform(image=image)['image']
        else:
            image = torch.from_numpy(image).permute(2, 0, 1)  # Convert HWC to CHW format

        # Convert labels and other data to tensors
        label = torch.tensor(label, dtype=torch.long)
        event_time = torch.tensor(event_time, dtype=torch.float)
        c = torch.tensor(c, dtype=torch.float)
        dead = torch.tensor(dead, dtype=torch.float)
        survival = torch.tensor(survival, dtype=torch.float)

        return {
            'image': image,
            'label': label,
            'event_time': event_time,
            'c': c,
            'dead': dead,
            'survival': survival,
            'patient_id': patient_id,
            'filename': filename,
        }
