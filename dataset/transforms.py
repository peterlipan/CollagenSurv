import albumentations as A
import numpy as np
from albumentations.pytorch import ToTensorV2


class Transforms:

    def __init__(self, size):
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)

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
                    brightness_limit=0.2, contrast_limit=0.2, p=0.7
                ),
                A.GridDropout(ratio=0.2, p=0.3),
                A.Normalize(mean=mean, std=std),
                ToTensorV2(),
            ]
        )

        self.test_transform = A.Compose(
            [
                A.Resize(height=size, width=size),
                A.Normalize(mean=mean, std=std),
                ToTensorV2(),
            ]
        )
