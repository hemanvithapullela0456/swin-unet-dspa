#!/usr/bin/env python3
"""
Training with Edge-Aware Loss
Train Swin-UNET with edge-aware loss for better road/boundary detection

Usage:
    python train_edge.py                 # Start training with edge-aware loss
    python train_edge.py --resume        # Resume from checkpoint
    python train_edge.py --edge-weight 0.7  # Custom edge weight
"""

import argparse
import os
import sys
import torch
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(__file__))

from config_edge import Config
from dataset import get_data_loaders
from model import SwinUNetpp
from edge_loss import CombinedEdgeLoss
from train_utils import save_checkpoint, load_checkpoint, find_latest_checkpoint


def train_one_epoch_edge(model, train_loader, optimizer, scaler, combined_loss, config, epoch):
    """Train for one epoch with edge-aware loss"""
    model.train()
    running_total_loss = 0.0
    running_l1_loss = 0.0
    running_edge_loss = 0.0
    running_perceptual_loss = 0.0

    pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{config.END_EPOCH} [TRAIN]")

    for batch_idx, (satellite, target_map) in enumerate(pbar):
        satellite = satellite.to(config.DEVICE)
        target_map = target_map.to(config.DEVICE)

        optimizer.zero_grad()

        # Mixed precision training
        with autocast():
            # Forward pass
            outputs = model(satellite)
            predicted_map = outputs[0]

            # Calculate combined loss
            total_loss, (l1_loss, edge_loss, perceptual_loss) = combined_loss(
                predicted_map, target_map
            )

        # Backward pass
        scaler.scale(total_loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # Track losses
        running_total_loss += total_loss.item()
        running_l1_loss += l1_loss.item()
        running_edge_loss += edge_loss.item()
        running_perceptual_loss += perceptual_loss.item()

        # Update progress bar
        pbar.set_postfix({
            'Loss': f'{total_loss.item():.4f}',
            'L1': f'{l1_loss.item():.4f}',
            'Edge': f'{edge_loss.item():.4f}',
            'Perc': f'{perceptual_loss.item():.4f}'
        })

    # Calculate average losses
    avg_total = running_total_loss / len(train_loader)
    avg_l1 = running_l1_loss / len(train_loader)
    avg_edge = running_edge_loss / len(train_loader)
    avg_perc = running_perceptual_loss / len(train_loader)

    return avg_total, avg_l1, avg_edge, avg_perc


def validate_edge(model, val_loader, combined_loss, config, epoch):
    """Validate with edge-aware loss"""
    model.eval()
    running_total_loss = 0.0
    running_l1_loss = 0.0
    running_edge_loss = 0.0
    running_perceptual_loss = 0.0

    pbar = tqdm(val_loader, desc=f"Epoch {epoch}/{config.END_EPOCH} [VAL]")

    with torch.no_grad():
        for satellite, target_map in pbar:
            satellite = satellite.to(config.DEVICE)
            target_map = target_map.to(config.DEVICE)

            # Forward pass
            with autocast():
                outputs = model(satellite)
                predicted_map = outputs[0]

                # Calculate combined loss
                total_loss, (l1_loss, edge_loss, perceptual_loss) = combined_loss(
                    predicted_map, target_map
                )

            # Track losses
            running_total_loss += total_loss.item()
            running_l1_loss += l1_loss.item()
            running_edge_loss += edge_loss.item()
            running_perceptual_loss += perceptual_loss.item()

            # Update progress bar
            pbar.set_postfix({
                'Loss': f'{total_loss.item():.4f}',
                'L1': f'{l1_loss.item():.4f}',
                'Edge': f'{edge_loss.item():.4f}',
                'Perc': f'{perceptual_loss.item():.4f}'
            })

    # Calculate average losses
    avg_total = running_total_loss / len(val_loader)
    avg_l1 = running_l1_loss / len(val_loader)
    avg_edge = running_edge_loss / len(val_loader)
    avg_perc = running_perceptual_loss / len(val_loader)

    return avg_total, avg_l1, avg_edge, avg_perc


def train_model_edge(model, train_loader, val_loader, config, resume=False):
    """Main training loop with edge-aware loss"""
    print("=" * 60)
    print("TRAINING WITH EDGE-AWARE LOSS")
    print("=" * 60)
    config.print_config()

    # Initialize model, optimizer, and loss
    model = model.to(config.DEVICE)
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY
    )
    scaler = GradScaler()

    # Combined loss with edge awareness
    combined_loss = CombinedEdgeLoss(
        l1_weight=config.L1_WEIGHT,
        edge_weight=config.EDGE_WEIGHT,
        perceptual_weight=config.PERCEPTUAL_WEIGHT
    ).to(config.DEVICE)

    start_epoch = config.START_EPOCH

    # Resume from checkpoint if requested
    if resume:
        latest_checkpoint = find_latest_checkpoint(config)
        if latest_checkpoint:
            start_epoch = load_checkpoint(model, optimizer, scaler, config, latest_checkpoint)
        else:
            print("No checkpoint found. Starting from scratch.")
            resume = False

    if not resume:
        print("Starting training from scratch")

    print("=" * 60)

    # Training loop
    best_val_loss = float('inf')

    for epoch in range(start_epoch, config.END_EPOCH + 1):
        print(f"\n{'=' * 60}")
        print(f"EPOCH {epoch}/{config.END_EPOCH}")
        print(f"{'=' * 60}")

        # Train one epoch
        train_loss, train_l1, train_edge, train_perc = train_one_epoch_edge(
            model, train_loader, optimizer, scaler, combined_loss, config, epoch
        )

        # Print training summary
        print(f"\n{'=' * 60}")
        print(f"EPOCH {epoch} TRAINING SUMMARY")
        print(f"{'=' * 60}")
        print(f"Train Loss: {train_loss:.4f}")
        print(f"  - L1:         {train_l1:.4f}")
        print(f"  - Edge:       {train_edge:.4f}")
        print(f"  - Perceptual: {train_perc:.4f}")
        print(f"{'=' * 60}")

        # Validate periodically
        if epoch % 5 == 0 or epoch == config.END_EPOCH:
            val_loss, val_l1, val_edge, val_perc = validate_edge(
                model, val_loader, combined_loss, config, epoch
            )
            
            print(f"\n{'=' * 60}")
            print(f"EPOCH {epoch} VALIDATION SUMMARY")
            print(f"{'=' * 60}")
            print(f"Val Loss: {val_loss:.4f}")
            print(f"  - L1:         {val_l1:.4f}")
            print(f"  - Edge:       {val_edge:.4f}")
            print(f"  - Perceptual: {val_perc:.4f}")
            print(f"{'=' * 60}")

            # Save checkpoint
            save_checkpoint(
                model, optimizer, scaler, epoch, val_loss,
                config, f"checkpoint_epoch_{epoch}.pth"
            )

            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_checkpoint(
                    model, optimizer, scaler, epoch, val_loss,
                    config, "best_model.pth"
                )
                print(f"✓ New best model saved! Val Loss: {val_loss:.4f}")

    print("\n" + "=" * 60)
    print("TRAINING COMPLETED!")
    print("=" * 60)
    print(f"Best Validation Loss: {best_val_loss:.4f}")
    print("=" * 60)

    return model


def main():
    parser = argparse.ArgumentParser(description='Train Swin-UNET with edge-aware loss')
    parser.add_argument('--resume', action='store_true',
                       help='Resume training from latest checkpoint')
    parser.add_argument('--batch-size', type=int, default=None,
                       help='Batch size (overrides config)')
    parser.add_argument('--epochs', type=int, default=None,
                       help='Number of epochs (overrides config)')
    parser.add_argument('--lr', type=float, default=None,
                       help='Learning rate (overrides config)')
    parser.add_argument('--edge-weight', type=float, default=None,
                       help='Edge loss weight (overrides config)')
    parser.add_argument('--num-workers', type=int, default=None,
                       help='Number of data loading workers')
    args = parser.parse_args()

    # Override config if arguments provided
    if args.batch_size:
        Config.BATCH_SIZE = args.batch_size
    if args.epochs:
        Config.END_EPOCH = args.epochs
    if args.lr:
        Config.LEARNING_RATE = args.lr
    if args.edge_weight is not None:
        Config.EDGE_WEIGHT = args.edge_weight
    if args.num_workers is not None:
        Config.NUM_WORKERS = args.num_workers

    # Print header
    print("\n" + "=" * 60)
    print("SWIN-UNET WITH EDGE-AWARE LOSS")
    print("Satellite to Map Translation")
    print("=" * 60)

    # Create directories
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
        return

    if len(train_loader.dataset) == 0:
        print("\n✗ Error: No training data found!")
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

    # Start training
    try:
        trained_model = train_model_edge(
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