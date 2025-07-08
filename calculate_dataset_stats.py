import os
import cv2
import pandas as pd
import numpy as np
from PIL import Image
from tqdm import tqdm  # For progress bar
from albumentations.core.transforms_interface import ImageOnlyTransform\

class GradientMapTransform(ImageOnlyTransform):
    def __init__(self, p=1.0):
        super().__init__(p=p)

    def get_params_dependent_on_data(self, params, data):
        return {}
    
    @staticmethod
    def safe_minmax_norm(arr):
        min_val = arr.min()
        max_val = arr.max()
        if max_val - min_val < 1e-5:
            return np.zeros_like(arr)  # or a constant if preferred
        return 255.0 * (arr - min_val) / (max_val - min_val)


    def apply(self, img, **params):
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        # Compute gradients
        grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        magnitude = np.sqrt(grad_x**2 + grad_y**2)

        # Normalize each to [0, 255] for pretrained ViT compatibility
        grad_x_norm = self.safe_minmax_norm(grad_x)
        grad_y_norm = self.safe_minmax_norm(grad_y)
        magnitude_norm = self.safe_minmax_norm(magnitude)

        gradient_image = np.stack([grad_x_norm, grad_y_norm, magnitude_norm], axis=-1).astype(np.uint8)

        return gradient_image
    

def calculate_mean_std(csv_path, root_path):
    # Load the CSV file
    df = pd.read_excel(csv_path)
    transform = GradientMapTransform(p=1.0)  # Initialize the transform

    # Initialize variables to calculate mean and std
    channel_sum = np.zeros(3)  # For R, G, B channels
    channel_squared_sum = np.zeros(3)
    num_pixels = 0

    # Iterate through the dataset
    for index, row in tqdm(df.iterrows(), total=len(df)):
        filename = row['Filename']

        # Construct paths for R, G, B images
        path_r = os.path.join(root_path, f"{row['Folder']}_R", filename)
        path_g = os.path.join(root_path, f"{row['Folder']}_G", filename)
        path_b = os.path.join(root_path, f"{row['Folder']}_B", filename)

        # Load each grayscale image
        image_r = np.array(Image.open(path_r).convert('L'), dtype=np.float32)  # Convert to grayscale
        image_g = np.array(Image.open(path_g).convert('L'), dtype=np.float32)  # Convert to grayscale
        image_b = np.array(Image.open(path_b).convert('L'), dtype=np.float32)  # Convert to grayscale

        # Stack grayscale images to form a 3-channel image
        image = np.stack([image_r, image_g, image_b], axis=-1)  # Shape: (H, W, 3)
        image = transform(image=image)['image']  # Apply the gradient map transform

        if np.isnan(image).any():
            print(f"NaNs found in image at index {index}: {filename}")
            continue

        # Update statistics
        channel_sum += image.sum(axis=(0, 1))  # Sum over height and width for each channel
        channel_squared_sum += (image ** 2).sum(axis=(0, 1))  # Sum of squares
        num_pixels += image.shape[0] * image.shape[1]  # Total number of pixels per channel

    # Calculate mean and std
    channel_mean = channel_sum / num_pixels
    channel_std = np.sqrt(channel_squared_sum / num_pixels - channel_mean ** 2)

    # Normalize mean and std for [0, 1] scale
    channel_mean /= 255.0
    channel_std /= 255.0

    return channel_mean, channel_std


# Example usage
csv_path = './Collagen_Images_Jun25.xlsx'  # Path to the CSV file
root_path = '/datastorage/li/CollagenRawImages'  # Root directory containing the images

mean, std = calculate_mean_std(csv_path, root_path)
print("Mean:", mean)
print("Std:", std)
