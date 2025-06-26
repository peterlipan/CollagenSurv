import albumentations as A
import numpy as np
from albumentations.pytorch import ToTensorV2


class Transforms:

    def __init__(self, size):
        # Compute dataset-specific mean and std if possible, or adjust these values
        mean = (0.21068942, 0.15048029, 0.104741)
        std = (0.26927587, 0.17788138, 0.08125762)

        self.train_transform = A.Compose(
            [
                A.Resize(height=size, width=size),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.OneOf([
                    A.MotionBlur(p=0.3),
                    A.MedianBlur(blur_limit=3, p=0.3),
                    A.GaussianBlur(blur_limit=(3, 5), p=0.3),
                ], p=0.7),
                A.ShiftScaleRotate(
                    shift_limit=0.05, scale_limit=0.1, rotate_limit=30, border_mode=0, p=0.7
                ),
                A.OneOf([
                    A.ElasticTransform(alpha=1, sigma=50, alpha_affine=30, p=0.3),
                    A.GridDistortion(p=0.3),
                    A.OpticalDistortion(distort_limit=0.05, shift_limit=0.05, p=0.3),
                ], p=0.7),
                A.RandomBrightnessContrast(
                    brightness_limit=(0.2, 0.5),  
                    contrast_limit=(0.2, 0.5),
                    p=0.8
                ),
                A.RandomGamma(gamma_limit=(80, 120), p=0.8),
                A.CLAHE(clip_limit=(2, 16), tile_grid_size=(8, 8), p=0.5),
                A.OneOf([
                    A.GaussNoise(var_limit=(10.0, 50.0), p=0.5),
                    A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5), p=0.5),
                ], p=0.5),
                A.GridDropout(ratio=0.2, p=0.3),
                
                A.Normalize(mean=mean, std=std),
                ToTensorV2(),
            ]
        )

        self.test_transform = A.Compose(
            [
                A.Resize(height=size, width=size),
                A.CLAHE(clip_limit=(8, 8), tile_grid_size=(8, 8), p=1.0),
                A.Normalize(mean=mean, std=std),
                ToTensorV2(),
            ]
        )
