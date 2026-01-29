# Swin Transformer for Image-to-Image Translation

This project implements a Swin Transformer-based U-Net for image-to-image translation tasks.

## Model Architecture
- **Encoder**: Swin Transformer blocks
- **Decoder**: Swin Transformer blocks with skip connections
- **Task**: Pix2Pix style image translation (satellite to map)

## Training Configuration
- Image Size: 256x256
- Batch Size: 1
- Total Epochs: 650
- Loss Function: L1 + Perceptual Loss

## Setup
```bash
pip install -r requirements.txt
```

## Usage

See `Copy_of_swin.ipynb` for detailed training and inference code.

## Dataset
Maps dataset (satellite imagery to map translation)

## Results
Sample results are available in the `results/` directory.