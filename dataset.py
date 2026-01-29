"""
Dataset module for Satellite to Map image translation
Handles loading and preprocessing of paired images
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image


class SatelliteMapDataset(Dataset):
    """Dataset for paired satellite and map images"""
    
    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir (str): Directory with all the images
            transform: Optional transform to be applied on a sample
        """
        self.root_dir = root_dir
        self.transform = transform
        
        # Get a sorted list of all image file names
        try:
            self.image_files = sorted([
                f for f in os.listdir(root_dir) 
                if os.path.isfile(os.path.join(root_dir, f)) and
                f.lower().endswith(('.png', '.jpg', '.jpeg'))
            ])
            if not self.image_files:
                print(f"Warning: No image files found in {root_dir}")
        except FileNotFoundError:
            print(f"Error: Directory not found: {root_dir}")
            print("Please make sure the dataset path is correct.")
            self.image_files = []

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        if idx >= len(self.image_files):
            raise IndexError("Index out of range")

        # Load the combined image
        img_path = os.path.join(self.root_dir, self.image_files[idx])
        try:
            combined_image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            raise e

        combined_array = np.array(combined_image)  # Shape: (600, 1200, 3)

        # Handle different image sizes gracefully
        h, w = combined_array.shape[:2]
        mid = w // 2
        
        if combined_array.shape[0] != 600 or combined_array.shape[1] != 1200:
            print(f"Warning: Image {img_path} has unexpected shape {combined_array.shape}. "
                  f"Expected (600, 1200, 3). Using dynamic split.")
        
        # Split into satellite and map images (left and right halves)
        satellite_image = combined_array[:, :mid, :]   # Left half
        map_image = combined_array[:, mid:, :]         # Right half

        # Convert back to PIL Images
        satellite_image = Image.fromarray(satellite_image)
        map_image = Image.fromarray(map_image)

        # Apply transforms if provided
        if self.transform:
            satellite_image = self.transform(satellite_image)
            map_image = self.transform(map_image)

        return satellite_image, map_image


def get_data_loaders(config):
    """
    Create train and validation data loaders
    
    Args:
        config: Configuration object with dataset settings
        
    Returns:
        train_loader, val_loader: PyTorch DataLoader objects
    """
    # Define transforms
    data_transform = transforms.Compose([
        transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])

    # Define paths
    train_path = os.path.join(config.DATASET_PATH, 'train')
    val_path = os.path.join(config.DATASET_PATH, 'val')

    print(f"Loading training data from: {train_path}")
    print(f"Loading validation data from: {val_path}")

    # Create datasets
    train_dataset = SatelliteMapDataset(root_dir=train_path, transform=data_transform)
    val_dataset = SatelliteMapDataset(root_dir=val_path, transform=data_transform)

    print(f"Total training images: {len(train_dataset)}")
    print(f"Total validation images: {len(val_dataset)}")

    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config.BATCH_SIZE, 
        shuffle=True, 
        num_workers=config.NUM_WORKERS,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=config.BATCH_SIZE, 
        shuffle=False, 
        num_workers=config.NUM_WORKERS,
        pin_memory=True if torch.cuda.is_available() else False
    )

    return train_loader, val_loader


def visualize_sample(train_loader):
    """Visualize a sample from the training data"""
    import matplotlib.pyplot as plt
    
    if len(train_loader.dataset) == 0:
        print("No training data available for visualization")
        return
    
    # Get one batch
    sat_batch, map_batch = next(iter(train_loader))
    print(f"Satellite image batch shape: {sat_batch.shape}")
    print(f"Map image batch shape: {map_batch.shape}")

    def show_image(tensor_img):
        """Denormalize and display tensor image"""
        tensor_img = tensor_img * 0.5 + 0.5  # Denormalize
        tensor_img = torch.clamp(tensor_img, 0, 1)
        plt_img = tensor_img.permute(1, 2, 0).numpy()
        plt.imshow(plt_img)
        plt.axis('off')

    # Display first sample
    sample_sat = sat_batch[0].cpu()
    sample_map = map_batch[0].cpu()

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.title("Input Satellite Image")
    show_image(sample_sat)

    plt.subplot(1, 2, 2)
    plt.title("Target Map Image")
    show_image(sample_map)
    
    plt.tight_layout()
    plt.savefig('sample_visualization.png', dpi=150, bbox_inches='tight')
    print("Sample visualization saved to: sample_visualization.png")
    plt.close()


if __name__ == "__main__":
    # Test the dataset loading
    from config import Config
    
    print("Testing dataset loading...")
    train_loader, val_loader = get_data_loaders(Config)
    
    if len(train_loader.dataset) > 0:
        visualize_sample(train_loader)
    else:
        print("No training data found. Please check your dataset path.")