"""
Configuration file for Swin-UNET training
Edit these settings according to your needs
"""

import torch

class Config:
    """Configuration class for all hyperparameters"""
    
    # ========================================
    # IMAGE SETTINGS
    # ========================================
    IMG_SIZE = 256  # Image size (256x256)
    PATCH_SIZE = 4
    IN_CHANNELS = 3
    OUT_CHANNELS = 3

    # ========================================
    # TRAINING SETTINGS
    # ========================================
    BATCH_SIZE = 4  # Increase if you have enough GPU memory
    NUM_WORKERS = 4  # Use 0 on Windows if you have issues
    START_EPOCH = 1
    END_EPOCH = 100  # Total epochs to train
    
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-4
    SAVE_EVERY = 5  # Save checkpoint every N epochs

    # ========================================
    # MODEL ARCHITECTURE
    # ========================================
    EMBED_DIM = 96
    DEPTHS = [2, 2, 6, 2]
    DEPTHS_DECODER = [2, 2, 6, 2]
    NUM_HEADS = [3, 6, 12, 24]
    WINDOW_SIZE = 8
    MLP_RATIO = 4.0

    # ========================================
    # REGULARIZATION
    # ========================================
    DROP_RATE = 0.0
    ATTN_DROP_RATE = 0.0
    DROP_PATH_RATE = 0.1
    
    QKV_BIAS = True
    PATCH_NORM = True

    # ========================================
    # LOSS WEIGHTS
    # ========================================
    L1_WEIGHT = 1.0
    PERCEPTUAL_WEIGHT = 0.01

    # ========================================
    # PATHS
    # ========================================
    DATASET_PATH = 'data/maps/'
    CHECKPOINT_DIR = 'checkpoints/'
    OUTPUT_DIR = 'outputs/'
    
    # ========================================
    # DEVICE
    # ========================================
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @classmethod
    def print_config(cls):
        """Print all configuration settings"""
        print("=" * 60)
        print("CONFIGURATION")
        print("=" * 60)
        print(f"Image Size: {cls.IMG_SIZE}x{cls.IMG_SIZE}")
        print(f"Batch Size: {cls.BATCH_SIZE}")
        print(f"Num Workers: {cls.NUM_WORKERS}")
        print(f"Learning Rate: {cls.LEARNING_RATE}")
        print(f"Epochs: {cls.START_EPOCH} to {cls.END_EPOCH}")
        print(f"Embed Dim: {cls.EMBED_DIM}")
        print(f"Window Size: {cls.WINDOW_SIZE}")
        print(f"Depths: {cls.DEPTHS}")
        print(f"Num Heads: {cls.NUM_HEADS}")
        print(f"Device: {cls.DEVICE}")
        print(f"Dataset Path: {cls.DATASET_PATH}")
        print(f"Checkpoint Dir: {cls.CHECKPOINT_DIR}")
        print("=" * 60)