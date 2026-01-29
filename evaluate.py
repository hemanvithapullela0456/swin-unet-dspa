#!/usr/bin/env python3
"""
Evaluation Script for Swin-UNET
Calculate metrics on validation set

Usage:
    python evaluate.py                              # Use best model
    python evaluate.py --checkpoint path/to/ckpt.pth  # Use specific checkpoint
"""

import argparse
import os
import sys
import torch
import numpy as np
from tqdm import tqdm
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

sys.path.insert(0, os.path.dirname(__file__))

from config import Config
from dataset import get_data_loaders
from model import SwinUNetpp
from train_utils import find_latest_checkpoint


# ============================================================================
# METRIC CALCULATIONS
# ============================================================================

def calculate_mae(pred, target):
    """Calculate Mean Absolute Error"""
    return torch.sum(torch.abs(pred - target)).item()


def calculate_psnr(pred, target):
    """Calculate Peak Signal-to-Noise Ratio"""
    # Denormalize from [-1, 1] to [0, 1]
    pred = (pred + 1) / 2
    target = (target + 1) / 2

    batch_size = pred.shape[0]
    psnr_sum = 0.0

    for i in range(batch_size):
        mse = torch.mean((pred[i] - target[i]) ** 2).item()

        if mse == 0:
            psnr_value = float('inf')
        else:
            psnr_value = 20 * np.log10(1.0) - 10 * np.log10(mse)
        psnr_sum += psnr_value

    return psnr_sum


def calculate_ssim_batch(pred, target):
    """Calculate Structural Similarity Index"""
    # Denormalize from [-1, 1] to [0, 1]
    pred = (pred + 1) / 2
    target = (target + 1) / 2

    pred_np = pred.cpu().numpy()
    target_np = target.cpu().numpy()

    batch_size = pred_np.shape[0]
    ssim_sum = 0.0

    for i in range(batch_size):
        pred_img = np.transpose(pred_np[i], (1, 2, 0))
        target_img = np.transpose(target_np[i], (1, 2, 0))

        ssim_value = ssim(pred_img, target_img,
                         data_range=1.0,
                         channel_axis=2)
        ssim_sum += ssim_value

    return ssim_sum


def pixelLevelAcc(pred, target):
    """Calculate Pixel-level Accuracy"""
    batch_size, C, H, W = pred.shape
    batch_accuracy_sum = 0.0

    for b in range(batch_size):
        abs_diff = torch.abs(pred[b] - target[b])
        max_diff_per_pixel = torch.max(abs_diff, dim=0).values  # Shape: (H, W)
        correct_pixels_in_image = torch.sum(max_diff_per_pixel <= 5).item()
        image_accuracy = (correct_pixels_in_image / (H * W)) * 100.0
        batch_accuracy_sum += image_accuracy

    return batch_accuracy_sum


# ============================================================================
# EVALUATION
# ============================================================================

def evaluate_metrics(model, val_loader, config, checkpoint_path=None):
    """Evaluate model on validation set"""
    print("=" * 60)
    print("METRICS EVALUATION")
    print("=" * 60)

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

    # Initialize metric accumulators
    total_mae = 0.0
    total_psnr = 0.0
    total_ssim = 0.0
    total_pixel_accuracy = 0.0
    total_images = 0

    print("\nCalculating metrics on validation set...")
    pbar = tqdm(val_loader, desc="Evaluating")

    with torch.no_grad():
        for satellite, target_map in pbar:
            satellite = satellite.to(config.DEVICE)
            target_map = target_map.to(config.DEVICE)

            batch_size = satellite.shape[0]

            # Forward pass
            outputs = model(satellite)
            predicted_map = outputs[0]

            # Calculate metrics
            mae = calculate_mae(predicted_map, target_map)
            psnr_value = calculate_psnr(predicted_map, target_map)
            ssim_value = calculate_ssim_batch(predicted_map, target_map)

            # Convert to 0-255 range for pixel accuracy
            pred_0_255 = ((predicted_map * 0.5 + 0.5) * 255).type(torch.IntTensor)
            target_0_255 = ((target_map * 0.5 + 0.5) * 255).type(torch.IntTensor)
            pixel_accuracy = pixelLevelAcc(pred_0_255, target_0_255)

            # Accumulate
            total_mae += mae
            total_psnr += psnr_value
            total_ssim += ssim_value
            total_pixel_accuracy += pixel_accuracy
            total_images += batch_size

            # Update progress bar
            pbar.set_postfix({
                'Batch MAE': f'{mae/batch_size:.6f}',
                'Batch PSNR': f'{psnr_value/batch_size:.2f}dB',
                'Batch SSIM': f'{ssim_value/batch_size:.4f}',
                'Batch Pixel Acc': f'{pixel_accuracy/batch_size:.2f}%'
            })

    # Calculate averages
    avg_mae = total_mae / total_images
    avg_psnr = total_psnr / total_images
    avg_ssim = total_ssim / total_images
    avg_pixel_accuracy = total_pixel_accuracy / total_images

    # Print results
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    print(f"Mean Absolute Error:        {avg_mae:.6f}")
    print(f"Peak Signal-to-Noise Ratio: {avg_psnr:.2f} dB")
    print(f"Structural Similarity Index: {avg_ssim:.4f}")
    print(f"Pixel Accuracy:             {avg_pixel_accuracy:.2f} %")
    print("=" * 60)

    return {
        'mae': avg_mae,
        'psnr': avg_psnr,
        'ssim': avg_ssim,
        'pixel_accuracy': avg_pixel_accuracy
    }


def main():
    parser = argparse.ArgumentParser(description='Evaluate Swin-UNET model')
    parser.add_argument('--checkpoint', type=str, default=None,
                       help='Path to checkpoint file (default: best_model.pth)')
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
    print("SWIN-UNET EVALUATION")
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

    # Evaluate
    try:
        metrics = evaluate_metrics(model, val_loader, Config, args.checkpoint)
        
        # Save metrics to file
        metrics_file = os.path.join(Config.OUTPUT_DIR, 'evaluation_metrics.txt')
        with open(metrics_file, 'w') as f:
            f.write("EVALUATION METRICS\n")
            f.write("=" * 60 + "\n")
            f.write(f"Checkpoint: {args.checkpoint}\n")
            f.write(f"Mean Absolute Error: {metrics['mae']:.6f}\n")
            f.write(f"Peak Signal-to-Noise Ratio: {metrics['psnr']:.2f} dB\n")
            f.write(f"Structural Similarity Index: {metrics['ssim']:.4f}\n")
            f.write(f"Pixel Accuracy: {metrics['pixel_accuracy']:.2f} %\n")
        
        print(f"\n✓ Metrics saved to: {metrics_file}")
        
    except Exception as e:
        print(f"✗ Error during evaluation: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()