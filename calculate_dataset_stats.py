import os
import pandas as pd
import numpy as np
from PIL import Image
from tqdm import tqdm  # For progress bar

def calculate_mean_std(csv_path, root_path):
    # Load the CSV file
    df = pd.read_excel(csv_path)

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
