#!/usr/bin/env python3
"""
Main Training Script for Swin-UNET
Run this script to train the model

Usage:
    python train.py                  # Start fresh training
    python train.py --resume         # Resume from latest checkpoint
"""

import argparse
import os
import sys

# Add current directory to path
sys.path.insert(0, os.path.dirname(__file__))

from config import Config
from dataset import get_data_loaders
from model import SwinUNetpp
from train_utils import train_model


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train Swin-UNET for satellite to map translation')
    parser.add_argument('--resume', action='store_true', 
                       help='Resume training from latest checkpoint')
    parser.add_argument('--batch-size', type=int, default=None,
                       help='Batch size (overrides config)')
    parser.add_argument('--epochs', type=int, default=None,
                       help='Number of epochs (overrides config)')
    parser.add_argument('--lr', type=float, default=None,
                       help='Learning rate (overrides config)')
    parser.add_argument('--num-workers', type=int, default=None,
                       help='Number of data loading workers (overrides config)')
    args = parser.parse_args()

    # Override config if arguments provided
    if args.batch_size:
        Config.BATCH_SIZE = args.batch_size
    if args.epochs:
        Config.END_EPOCH = args.epochs
    if args.lr:
        Config.LEARNING_RATE = args.lr
    if args.num_workers is not None:
        Config.NUM_WORKERS = args.num_workers

    # Print header
    print("\n" + "=" * 60)
    print("SWIN-UNET TRAINING")
    print("Satellite to Map Image Translation")
    print("=" * 60)

    # Create directories if they don't exist
    os.makedirs(Config.CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(Config.OUTPUT_DIR, exist_ok=True)
    
    print(f"\n✓ Checkpoint directory: {Config.CHECKPOINT_DIR}")
    print(f"✓ Output directory: {Config.OUTPUT_DIR}")

    # Load data
    print("\n" + "=" * 60)
    print("LOADING DATA")
    print("=" * 60)
    
    try:
        train_loader, val_loader = get_data_loaders(Config)
    except Exception as e:
        print(f"\n✗ Error loading data: {e}")
        print("\nPlease check:")
        print(f"  1. Dataset path exists: {Config.DATASET_PATH}")
        print(f"  2. Train directory exists: {os.path.join(Config.DATASET_PATH, 'train')}")
        print(f"  3. Val directory exists: {os.path.join(Config.DATASET_PATH, 'val')}")
        print(f"  4. Images are in the correct format (.jpg, .png)")
        return

    if len(train_loader.dataset) == 0:
        print("\n✗ Error: No training data found!")
        print(f"Please add images to: {os.path.join(Config.DATASET_PATH, 'train')}")
        return

    # Create model
    print("\n" + "=" * 60)
    print("CREATING MODEL")
    print("=" * 60)
    
    model = SwinUNetpp(Config)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"✓ Model created successfully!")
    print(f"✓ Total parameters: {total_params:,}")
    print(f"✓ Trainable parameters: {trainable_params:,}")
    print(f"✓ Device: {Config.DEVICE}")

    # Training options
    if not args.resume:
        print("\n" + "=" * 60)
        print("TRAINING OPTIONS")
        print("=" * 60)
        print("Starting fresh training")
        print(f"Epochs: {Config.START_EPOCH} to {Config.END_EPOCH}")

    # Start training
    try:
        trained_model = train_model(
            model, 
            train_loader, 
            val_loader, 
            Config, 
            resume=args.resume
        )
        
        print("\n✓ Training completed successfully!")
        print(f"✓ Checkpoints saved in: {Config.CHECKPOINT_DIR}")
        
    except KeyboardInterrupt:
        print("\n\n⚠ Training interrupted by user")
        print(f"✓ Checkpoints saved in: {Config.CHECKPOINT_DIR}")
    except Exception as e:
        print(f"\n✗ Error during training: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()