#!/usr/bin/env python3
"""
Generate Sample Predictions
Visualize model predictions on validation set

Usage:
    python generate.py                          # Generate 10 samples
    python generate.py --num-samples 20         # Generate 20 samples
    python generate.py --checkpoint path.pth    # Use specific checkpoint
"""

import argparse
import os
import sys
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import random

sys.path.insert(0, os.path.dirname(__file__))

from config import Config
from dataset import get_data_loaders
from model import SwinUNetpp
from train_utils import find_latest_checkpoint


def generate_samples(model, val_loader, config, num_samples=10, 
                    checkpoint_path=None, seed=42, save_dir=None):
    """Generate and save sample predictions"""
    
    print("=" * 60)
    print("SAMPLE GENERATION")
    print("=" * 60)

    # Set random seeds for reproducibility
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Load checkpoint
    if checkpoint_path is None:
        checkpoint_path = find_latest_checkpoint(config)
        if checkpoint_path:
            checkpoint_path = os.path.join(config.CHECKPOINT_DIR, checkpoint_path)

    if checkpoint_path and os.path.exists(checkpoint_path):
        print(f"Loading checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=config.DEVICE)
        model.load_state_dict(checkpoint['model_state_dict'])
        epoch = checkpoint['epoch']
        print(f"✓ Loaded model from epoch {epoch}")
    else:
        print("⚠ No checkpoint found. Using current model weights.")

    print("=" * 60)

    # Set model to evaluation mode
    model.eval()
    model = model.to(config.DEVICE)

    # Collect all validation samples
    print(f"\nCollecting validation samples...")
    all_samples = []

    with torch.no_grad():
        for satellite, target_map in val_loader:
            for i in range(satellite.shape[0]):
                all_samples.append((satellite[i], target_map[i]))

    print(f"✓ Total validation samples: {len(all_samples)}")

    if num_samples > len(all_samples):
        num_samples = len(all_samples)
        print(f"⚠ Requested {num_samples} samples but only {len(all_samples)} available")

    # Randomly select samples
    selected_indices = random.sample(range(len(all_samples)), num_samples)
    print(f"✓ Randomly selected {num_samples} samples")

    # Create save directory
    if save_dir is None:
        save_dir = os.path.join(config.OUTPUT_DIR, 'predictions')
    os.makedirs(save_dir, exist_ok=True)

    print("\n" + "=" * 60)
    print("GENERATING PREDICTIONS")
    print("=" * 60)

    for idx, sample_idx in enumerate(selected_indices):
        satellite, target_map = all_samples[sample_idx]

        # Add batch dimension
        satellite_batch = satellite.unsqueeze(0).to(config.DEVICE)
        target_map_batch = target_map.unsqueeze(0).to(config.DEVICE)

        # Generate prediction
        with torch.no_grad():
            outputs = model(satellite_batch)
            predicted_map = outputs[0]

        # Move to CPU and denormalize
        satellite = satellite.cpu()
        target_map = target_map.cpu()
        predicted_map = predicted_map.squeeze(0).cpu()

        # Denormalize from [-1, 1] to [0, 1]
        satellite = (satellite + 1) / 2
        target_map = (target_map + 1) / 2
        predicted_map = (predicted_map + 1) / 2

        # Clamp values
        satellite = torch.clamp(satellite, 0, 1)
        target_map = torch.clamp(target_map, 0, 1)
        predicted_map = torch.clamp(predicted_map, 0, 1)

        # Convert to numpy
        satellite_np = satellite.permute(1, 2, 0).numpy()
        target_map_np = target_map.permute(1, 2, 0).numpy()
        predicted_map_np = predicted_map.permute(1, 2, 0).numpy()

        # Calculate error map
        error_map = np.abs(target_map_np - predicted_map_np)
        error_map = np.mean(error_map, axis=2)  # Average across channels

        # Calculate MAE for this sample
        mae = np.mean(error_map)

        print(f"\nSample {idx + 1}/{num_samples} (Index: {sample_idx})")
        print(f"  MAE: {mae:.6f}")

        # Create figure
        fig, axes = plt.subplots(1, 4, figsize=(20, 5))

        # Input Satellite Image
        axes[0].imshow(satellite_np)
        axes[0].set_title('Input Satellite', fontsize=14, fontweight='bold')
        axes[0].axis('off')

        # Ground Truth Map
        axes[1].imshow(target_map_np)
        axes[1].set_title('Ground Truth Map', fontsize=14, fontweight='bold')
        axes[1].axis('off')

        # Predicted Map
        axes[2].imshow(predicted_map_np)
        axes[2].set_title('Predicted Map', fontsize=14, fontweight='bold')
        axes[2].axis('off')

        # Error Map
        im = axes[3].imshow(error_map, cmap='hot', vmin=0, vmax=1)
        axes[3].set_title(f'Error Map (MAE: {mae:.4f})', fontsize=14, fontweight='bold')
        axes[3].axis('off')
        plt.colorbar(im, ax=axes[3], fraction=0.046, pad=0.04)

        plt.tight_layout()
        
        # Save figure
        save_path = os.path.join(save_dir, f'sample_{idx+1:03d}.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"  ✓ Saved to: {save_path}")

    print("\n" + "=" * 60)
    print("SAMPLE GENERATION COMPLETED")
    print("=" * 60)
    print(f"✓ All samples saved to: {save_dir}")


def main():
    parser = argparse.ArgumentParser(description='Generate sample predictions')
    parser.add_argument('--num-samples', type=int, default=10,
                       help='Number of samples to generate (default: 10)')
    parser.add_argument('--checkpoint', type=str, default=None,
                       help='Path to checkpoint file')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for sample selection')
    parser.add_argument('--output-dir', type=str, default=None,
                       help='Output directory for generated samples')
    parser.add_argument('--num-workers', type=int, default=None,
                       help='Number of data loading workers')
    args = parser.parse_args()

    # Override config if provided
    if args.num_workers is not None:
        Config.NUM_WORKERS = args.num_workers

    # Use best model by default
    if args.checkpoint is None:
        args.checkpoint = os.path.join(Config.CHECKPOINT_DIR, 'best_model.pth')

    print("\n" + "=" * 60)
    print("SWIN-UNET SAMPLE GENERATION")
    print("=" * 60)

    # Load data
    print("\nLoading validation data...")
    try:
        _, val_loader = get_data_loaders(Config)
    except Exception as e:
        print(f"✗ Error loading data: {e}")
        return

    if len(val_loader.dataset) == 0:
        print("✗ No validation data found!")
        return

    print(f"✓ Loaded {len(val_loader.dataset)} validation images")

    # Create model
    print("\nCreating model...")
    model = SwinUNetpp(Config)
    print(f"✓ Model created")

    # Generate samples
    try:
        generate_samples(
            model, 
            val_loader, 
            Config,
            num_samples=args.num_samples,
            checkpoint_path=args.checkpoint,
            seed=args.seed,
            save_dir=args.output_dir
        )
    except Exception as e:
        print(f"✗ Error during generation: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()