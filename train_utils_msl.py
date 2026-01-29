"""
Training utilities for Swin-UNET
Includes loss functions, checkpoint management, and training/validation loops
"""

import os
import re
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
import torchvision.models as models
from tqdm import tqdm


# ============================================================================
# LOSS FUNCTIONS
# ============================================================================

class VGGPerceptualLoss(nn.Module):
    """Perceptual loss using VGG19 features"""
    
    def __init__(self):
        super().__init__()
        # Load pretrained VGG19
        vgg = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1).features

        self.slice1 = nn.Sequential(*list(vgg.children())[:4])
        self.slice2 = nn.Sequential(*list(vgg.children())[4:9])
        self.slice3 = nn.Sequential(*list(vgg.children())[9:18])
        self.slice4 = nn.Sequential(*list(vgg.children())[18:27])

        # Freeze all parameters
        for param in self.parameters():
            param.requires_grad = False

        self.eval()

    def forward(self, x, y):
        # Denormalize from [-1, 1] to [0, 1]
        x = (x + 1) / 2
        y = (y + 1) / 2

        # Normalize for VGG (ImageNet stats)
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(x.device)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(x.device)
        x = (x - mean) / std
        y = (y - mean) / std

        loss = 0.0

        x1 = self.slice1(x)
        y1 = self.slice1(y)
        loss += nn.functional.l1_loss(x1, y1)

        x2 = self.slice2(x1)
        y2 = self.slice2(y1)
        loss += nn.functional.l1_loss(x2, y2)

        x3 = self.slice3(x2)
        y3 = self.slice3(y2)
        loss += nn.functional.l1_loss(x3, y3)

        x4 = self.slice4(x3)
        y4 = self.slice4(y3)
        loss += nn.functional.l1_loss(x4, y4)

        return loss / 4.0


class MultiscaleLoss(nn.Module):
    """
    Multiscale L1 loss at multiple resolutions
    Compares predictions and targets at full, half, and quarter resolution
    """
    
    def __init__(self, scales=[1.0, 0.5, 0.25], weights=[1.0, 0.5, 0.25]):
        """
        Args:
            scales: List of scale factors (1.0 = full res, 0.5 = half, 0.25 = quarter)
            weights: Corresponding weights for each scale
        """
        super().__init__()
        self.scales = scales
        self.weights = weights
        self.l1_loss = nn.L1Loss()
        
        print(f"✓ MultiscaleLoss initialized with scales: {scales}, weights: {weights}")
    
    def forward(self, pred, target):
        """
        Args:
            pred: predicted image [B, C, H, W]
            target: target image [B, C, H, W]
        Returns:
            Weighted sum of L1 losses at multiple scales
        """
        total_loss = 0.0
        
        for scale, weight in zip(self.scales, self.weights):
            if scale == 1.0:
                # Full resolution - no downsampling needed
                scaled_pred = pred
                scaled_target = target
            else:
                # Downscale using bilinear interpolation
                h, w = pred.shape[2], pred.shape[3]
                new_h, new_w = int(h * scale), int(w * scale)
                
                scaled_pred = F.interpolate(
                    pred, 
                    size=(new_h, new_w), 
                    mode='bilinear', 
                    align_corners=False
                )
                scaled_target = F.interpolate(
                    target, 
                    size=(new_h, new_w), 
                    mode='bilinear', 
                    align_corners=False
                )
            
            # Calculate L1 loss at this scale
            loss_at_scale = self.l1_loss(scaled_pred, scaled_target)
            total_loss += weight * loss_at_scale
        
        return total_loss


# ============================================================================
# CHECKPOINT MANAGEMENT
# ============================================================================

def save_checkpoint(model, optimizer, scaler, epoch, loss, config, filename):
    """Save model checkpoint"""
    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scaler_state_dict': scaler.state_dict(),
        'loss': loss,
        'config': {
            'IMG_SIZE': config.IMG_SIZE,
            'EMBED_DIM': config.EMBED_DIM,
            'DEPTHS': config.DEPTHS,
            'NUM_HEADS': config.NUM_HEADS,
            'WINDOW_SIZE': config.WINDOW_SIZE,
        }
    }
    filepath = os.path.join(config.CHECKPOINT_DIR, filename)
    torch.save(checkpoint, filepath)
    print(f"✓ Checkpoint saved: {filepath}")


def load_checkpoint(model, optimizer, scaler, config, filename):
    """Load model checkpoint"""
    filepath = os.path.join(config.CHECKPOINT_DIR, filename)
    if not os.path.exists(filepath):
        print(f"✗ Checkpoint not found: {filepath}")
        return 0

    checkpoint = torch.load(filepath, map_location=config.DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scaler.load_state_dict(checkpoint['scaler_state_dict'])
    epoch = checkpoint['epoch']

    print(f"✓ Checkpoint loaded: {filepath}")
    print(f"✓ Resuming from epoch {epoch + 1}")
    return epoch + 1


def find_latest_checkpoint(config):
    """Find the latest checkpoint in the checkpoint directory"""
    if not os.path.exists(config.CHECKPOINT_DIR):
        return None

    checkpoints = [f for f in os.listdir(config.CHECKPOINT_DIR) if f.endswith('.pth')]
    if not checkpoints:
        return None

    def get_epoch_from_filename(filename):
        match = re.search(r'_epoch_(\d+)\.pth$', filename)
        if match:
            return int(match.group(1))
        return 0

    checkpoints.sort(key=get_epoch_from_filename)
    return checkpoints[-1]


# ============================================================================
# TRAINING & VALIDATION (WITH MULTISCALE SUPPORT)
# ============================================================================

def train_one_epoch(model, train_loader, optimizer, scaler, l1_criterion, 
                    perceptual_criterion, config, epoch, multiscale_criterion=None):
    """Train for one epoch"""
    model.train()
    running_loss = 0.0
    running_l1_loss = 0.0
    running_perceptual_loss = 0.0
    running_multiscale_loss = 0.0

    pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{config.END_EPOCH} [TRAIN]")

    for batch_idx, (satellite, target_map) in enumerate(pbar):
        satellite = satellite.to(config.DEVICE)
        target_map = target_map.to(config.DEVICE)

        optimizer.zero_grad()

        # Mixed precision training
        with autocast():
            # Forward pass
            outputs = model(satellite)
            predicted_map = outputs[0]  # Get first (and only) output

            # Calculate losses
            l1_loss = l1_criterion(predicted_map, target_map)
            perceptual_loss = perceptual_criterion(predicted_map, target_map)
            
            # Calculate multiscale loss if enabled
            if multiscale_criterion is not None:
                multiscale_loss = multiscale_criterion(predicted_map, target_map)
            else:
                multiscale_loss = torch.tensor(0.0).to(config.DEVICE)

            # Combined loss
            total_loss = (config.L1_WEIGHT * l1_loss +
                         config.PERCEPTUAL_WEIGHT * perceptual_loss +
                         config.MULTISCALE_WEIGHT * multiscale_loss)

        # Backward pass
        scaler.scale(total_loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # Track losses
        running_loss += total_loss.item()
        running_l1_loss += l1_loss.item()
        running_perceptual_loss += perceptual_loss.item()
        running_multiscale_loss += multiscale_loss.item()

        # Update progress bar
        if multiscale_criterion is not None:
            pbar.set_postfix({
                'Loss': f'{total_loss.item():.4f}',
                'L1': f'{l1_loss.item():.4f}',
                'Perc': f'{perceptual_loss.item():.4f}',
                'Multi': f'{multiscale_loss.item():.4f}'
            })
        else:
            pbar.set_postfix({
                'Loss': f'{total_loss.item():.4f}',
                'L1': f'{l1_loss.item():.4f}',
                'Perc': f'{perceptual_loss.item():.4f}'
            })

    # Calculate average losses
    avg_loss = running_loss / len(train_loader)
    avg_l1 = running_l1_loss / len(train_loader)
    avg_perc = running_perceptual_loss / len(train_loader)
    avg_multi = running_multiscale_loss / len(train_loader)

    return avg_loss, avg_l1, avg_perc, avg_multi


def validate(model, val_loader, l1_criterion, perceptual_criterion, config, epoch, multiscale_criterion=None):
    """Validate the model"""
    model.eval()
    running_loss = 0.0
    running_l1_loss = 0.0
    running_perceptual_loss = 0.0
    running_multiscale_loss = 0.0

    pbar = tqdm(val_loader, desc=f"Epoch {epoch}/{config.END_EPOCH} [VAL]")

    with torch.no_grad():
        for satellite, target_map in pbar:
            satellite = satellite.to(config.DEVICE)
            target_map = target_map.to(config.DEVICE)

            # Forward pass
            with autocast():
                outputs = model(satellite)
                predicted_map = outputs[0]

                # Calculate losses
                l1_loss = l1_criterion(predicted_map, target_map)
                perceptual_loss = perceptual_criterion(predicted_map, target_map)
                
                # Calculate multiscale loss if enabled
                if multiscale_criterion is not None:
                    multiscale_loss = multiscale_criterion(predicted_map, target_map)
                else:
                    multiscale_loss = torch.tensor(0.0).to(config.DEVICE)

                total_loss = (config.L1_WEIGHT * l1_loss +
                             config.PERCEPTUAL_WEIGHT * perceptual_loss +
                             config.MULTISCALE_WEIGHT * multiscale_loss)

            # Track losses
            running_loss += total_loss.item()
            running_l1_loss += l1_loss.item()
            running_perceptual_loss += perceptual_loss.item()
            running_multiscale_loss += multiscale_loss.item()

            # Update progress bar
            if multiscale_criterion is not None:
                pbar.set_postfix({
                    'Loss': f'{total_loss.item():.4f}',
                    'L1': f'{l1_loss.item():.4f}',
                    'Perc': f'{perceptual_loss.item():.4f}',
                    'Multi': f'{multiscale_loss.item():.4f}'
                })
            else:
                pbar.set_postfix({
                    'Loss': f'{total_loss.item():.4f}',
                    'L1': f'{l1_loss.item():.4f}',
                    'Perc': f'{perceptual_loss.item():.4f}'
                })

    # Calculate average losses
    avg_loss = running_loss / len(val_loader)
    avg_l1 = running_l1_loss / len(val_loader)
    avg_perc = running_perceptual_loss / len(val_loader)
    avg_multi = running_multiscale_loss / len(val_loader)

    return avg_loss, avg_l1, avg_perc, avg_multi


def train_model(model, train_loader, val_loader, config, resume=False):
    """Main training loop"""
    print("=" * 60)
    print("TRAINING CONFIGURATION")
    print("=" * 60)
    config.print_config()

    # Initialize model, optimizer, and loss functions
    model = model.to(config.DEVICE)
    optimizer = optim.AdamW(model.parameters(),
                           lr=config.LEARNING_RATE,
                           weight_decay=config.WEIGHT_DECAY)
    scaler = GradScaler()

    l1_criterion = nn.L1Loss()
    perceptual_criterion = VGGPerceptualLoss().to(config.DEVICE)
    
    # Initialize multiscale loss if weight > 0
    multiscale_criterion = None
    if hasattr(config, 'MULTISCALE_WEIGHT') and config.MULTISCALE_WEIGHT > 0:
        multiscale_criterion = MultiscaleLoss().to(config.DEVICE)
        print(f"✓ Multiscale Loss enabled with weight: {config.MULTISCALE_WEIGHT}")
    else:
        print("✓ Multiscale Loss disabled")

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
        train_loss, train_l1, train_perc, train_multi = train_one_epoch(
            model, train_loader, optimizer, scaler,
            l1_criterion, perceptual_criterion, config, epoch,
            multiscale_criterion=multiscale_criterion
        )

        # Print training summary
        print(f"\n{'=' * 60}")
        print(f"EPOCH {epoch} TRAINING SUMMARY")
        print(f"{'=' * 60}")
        if multiscale_criterion is not None:
            print(f"Train Loss: {train_loss:.4f} | L1: {train_l1:.4f} | Perc: {train_perc:.4f} | Multi: {train_multi:.4f}")
        else:
            print(f"Train Loss: {train_loss:.4f} | L1: {train_l1:.4f} | Perceptual: {train_perc:.4f}")
        print(f"{'=' * 60}")

        # Validate periodically
        val_loss = None
        if epoch % 5 == 0 or epoch == config.END_EPOCH:
            val_loss, val_l1, val_perc, val_multi = validate(
                model, val_loader,
                l1_criterion, perceptual_criterion, config, epoch,
                multiscale_criterion=multiscale_criterion
            )
            print(f"\n{'=' * 60}")
            print(f"EPOCH {epoch} VALIDATION SUMMARY")
            print(f"{'=' * 60}")
            if multiscale_criterion is not None:
                print(f"Val Loss:   {val_loss:.4f} | L1: {val_l1:.4f} | Perc: {val_perc:.4f} | Multi: {val_multi:.4f}")
            else:
                print(f"Val Loss:   {val_loss:.4f} | L1: {val_l1:.4f} | Perceptual: {val_perc:.4f}")
            print(f"{'=' * 60}")

            # Save checkpoint
            save_checkpoint(model, optimizer, scaler, epoch, val_loss,
                          config, f"checkpoint_epoch_{epoch}.pth")

            # Save best model
            if val_loss is not None and val_loss < best_val_loss:
                best_val_loss = val_loss
                save_checkpoint(model, optimizer, scaler, epoch, val_loss,
                              config, "best_model.pth")
                print(f"✓ New best model saved! Val Loss: {val_loss:.4f}")

    print("\n" + "=" * 60)
    print("TRAINING COMPLETED!")
    print("=" * 60)
    print(f"Best Validation Loss: {best_val_loss:.4f}")
    print("=" * 60)

    return model