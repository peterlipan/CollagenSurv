import albumentations as A
import numpy as np
import cv2
import torch
from albumentations.core.transforms_interface import ImageOnlyTransform
from albumentations.pytorch import ToTensorV2

# Custom frequency-domain augmentation
class FDAAugment(ImageOnlyTransform):
    def __init__(self, beta=0.01, p=0.5):
        super().__init__(p=p)
        self.beta = beta

    def get_params_dependent_on_data(self, params, data):
        # Generate a random target image for frequency domain mixing
        src_shape = data["image"].shape
        target = np.random.uniform(0, 255, src_shape).astype(np.float32)
        return {"target": target}

    def apply(self, img, target, **params):
        # Ensure 3 channels
        if img.ndim == 2:
            img = np.expand_dims(img, axis=-1)
        if img.shape[2] == 1:
            img = np.repeat(img, 3, axis=-1)

        tgt_img = target

        # FFT2
        fft_src = np.fft.fft2(img, axes=(0, 1))
        fft_tgt = np.fft.fft2(tgt_img, axes=(0, 1))

        amp_src, pha_src = np.abs(fft_src), np.angle(fft_src)
        amp_tgt = np.abs(fft_tgt)

        h, w = img.shape[:2]
        b = int(min(h, w) * self.beta)
        c_h, c_w = h // 2, w // 2

        amp_src[c_h - b:c_h + b, c_w - b:c_w + b] = amp_tgt[c_h - b:c_h + b, c_w - b:c_w + b]
        fft_src = amp_src * np.exp(1j * pha_src)

        augmented = np.fft.ifft2(fft_src, axes=(0, 1)).real
        augmented = np.clip(augmented, 0, 255).astype(np.uint8)

        return augmented


class GradientMapTransform(ImageOnlyTransform):
    def __init__(self, p=1.0):
        super().__init__(p=p)

    def get_params_dependent_on_data(self, params, data):
        return {}

    def apply(self, img, **params):
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        # Compute gradients
        grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        magnitude = np.sqrt(grad_x**2 + grad_y**2)

        # Normalize each to [0, 255] for pretrained ViT compatibility
        grad_x_norm = cv2.normalize(grad_x, None, 0, 255, cv2.NORM_MINMAX)
        grad_y_norm = cv2.normalize(grad_y, None, 0, 255, cv2.NORM_MINMAX)
        magnitude_norm = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)

        gradient_image = np.stack([grad_x_norm, grad_y_norm, magnitude_norm], axis=-1).astype(np.uint8)

        return gradient_image


class Transforms:
    def __init__(self, size):
        # Mean and std should be recomputed over gradient-based training set ideally
        mean = (0.5, 0.5, 0.5)
        std = (0.5, 0.5, 0.5)

        self.train_transform = A.Compose([
            # Convert to gradient image first
            # GradientMapTransform(p=1.0),

            # Spatial augmentations on gradient image
            A.Resize(height=size, width=size),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.ShiftScaleRotate(
                shift_limit=0.05, scale_limit=0.1, rotate_limit=30,
                border_mode=cv2.BORDER_REFLECT_101, p=0.7
            ),
            A.OneOf([
                A.ElasticTransform(alpha=1, sigma=50, alpha_affine=30, p=0.3),
                A.GridDistortion(p=0.3),
                A.OpticalDistortion(distort_limit=0.05, shift_limit=0.05, p=0.3),
            ], p=0.7),
            A.OneOf([
                A.MotionBlur(p=0.3),
                A.MedianBlur(blur_limit=3, p=0.3),
                A.GaussianBlur(blur_limit=(3, 5), p=0.3),
            ], p=0.5),
            # Frequency-domain augmentation directly on gradients (less common, but allowed)
            # FDAAugment(beta=0.02, p=0.3),

            # Use dropout sparingly; gradient patterns can be sensitive
            A.GridDropout(ratio=0.15, p=0.2),

            A.Normalize(mean=mean, std=std),
            ToTensorV2(),
        ])

        self.test_transform = A.Compose([
            # GradientMapTransform(p=1.0),
            A.Resize(height=size, width=size),
            A.Normalize(mean=mean, std=std),
            ToTensorV2(),
        ])
