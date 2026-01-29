"""
Edge-Aware Loss Implementation
Uses Sobel gradients to emphasize edges, boundaries, and road structures
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class EdgeAwareLoss(nn.Module):
    """
    Edge-Aware Loss using Sobel operators
    
    Emphasizes edge regions (roads, building boundaries) in map translation.
    Calculates gradient magnitude using Sobel operators and compares predicted
    vs ground truth edge maps.
    
    This is particularly useful for satellite-to-map translation where:
    - Roads should have clear, continuous boundaries
    - Building edges should be sharp
    - Map features should have well-defined outlines
    """
    
    def __init__(self, edge_weight=1.0):
        """
        Args:
            edge_weight: Weight for edge loss component (default: 1.0)
        """
        super().__init__()
        self.edge_weight = edge_weight
        
        # Sobel kernels for edge detection
        # Horizontal (Gx) and Vertical (Gy) gradients
        sobel_x = torch.tensor([
            [-1, 0, 1],
            [-2, 0, 2],
            [-1, 0, 1]
        ], dtype=torch.float32).view(1, 1, 3, 3)
        
        sobel_y = torch.tensor([
            [-1, -2, -1],
            [ 0,  0,  0],
            [ 1,  2,  1]
        ], dtype=torch.float32).view(1, 1, 3, 3)
        
        # Register as buffers (not trainable parameters)
        self.register_buffer('sobel_x', sobel_x)
        self.register_buffer('sobel_y', sobel_y)
        
        # Replicate for RGB channels
        self.sobel_x_rgb = None
        self.sobel_y_rgb = None
        
        print(f"✓ EdgeAwareLoss initialized with edge_weight: {edge_weight}")
    
    def _get_sobel_kernels(self, channels):
        """Prepare Sobel kernels for the correct number of channels"""
        if self.sobel_x_rgb is None or self.sobel_x_rgb.shape[0] != channels:
            # Replicate Sobel kernels for each channel
            self.sobel_x_rgb = self.sobel_x.repeat(channels, 1, 1, 1)
            self.sobel_y_rgb = self.sobel_y.repeat(channels, 1, 1, 1)
        return self.sobel_x_rgb, self.sobel_y_rgb
    
    def compute_gradients(self, x):
        """
        Compute gradient magnitude using Sobel operators
        
        Args:
            x: Input tensor [B, C, H, W]
            
        Returns:
            Gradient magnitude [B, C, H, W]
        """
        B, C, H, W = x.shape
        
        # Get Sobel kernels for the right number of channels
        sobel_x, sobel_y = self._get_sobel_kernels(C)
        
        # Compute gradients (groups=C for per-channel convolution)
        grad_x = F.conv2d(x, sobel_x, padding=1, groups=C)
        grad_y = F.conv2d(x, sobel_y, padding=1, groups=C)
        
        # Gradient magnitude: sqrt(Gx^2 + Gy^2)
        grad_magnitude = torch.sqrt(grad_x ** 2 + grad_y ** 2 + 1e-8)
        
        return grad_magnitude
    
    def forward(self, pred, target):
        """
        Compute edge-aware loss
        
        Args:
            pred: Predicted image [B, C, H, W], range [-1, 1]
            target: Target image [B, C, H, W], range [-1, 1]
            
        Returns:
            Edge-aware loss value
        """
        # Denormalize from [-1, 1] to [0, 1] for better gradient computation
        pred = (pred + 1) / 2
        target = (target + 1) / 2
        
        # Compute edge maps
        pred_edges = self.compute_gradients(pred)
        target_edges = self.compute_gradients(target)
        
        # L1 loss on edge magnitudes
        edge_loss = F.l1_loss(pred_edges, target_edges)
        
        return self.edge_weight * edge_loss


class CombinedEdgeLoss(nn.Module):
    """
    Combined loss: L1 + Edge-Aware + Perceptual
    
    This is the complete loss function for training with edge awareness.
    """
    
    def __init__(self, l1_weight=1.0, edge_weight=0.5, perceptual_weight=0.01):
        """
        Args:
            l1_weight: Weight for L1 (pixel-wise) loss
            edge_weight: Weight for edge-aware loss
            perceptual_weight: Weight for perceptual (VGG) loss
        """
        super().__init__()
        self.l1_weight = l1_weight
        self.edge_weight = edge_weight
        self.perceptual_weight = perceptual_weight
        
        # Loss components
        self.l1_loss = nn.L1Loss()
        self.edge_loss = EdgeAwareLoss(edge_weight=edge_weight)
        
        # Import VGG perceptual loss
        from train_utils import VGGPerceptualLoss
        self.perceptual_loss = VGGPerceptualLoss()
        
        print(f"✓ CombinedEdgeLoss initialized:")
        print(f"  - L1 weight: {l1_weight}")
        print(f"  - Edge weight: {edge_weight}")
        print(f"  - Perceptual weight: {perceptual_weight}")
    
    def forward(self, pred, target):
        """
        Compute combined loss
        
        Args:
            pred: Predicted image [B, C, H, W]
            target: Target image [B, C, H, W]
            
        Returns:
            total_loss, (l1_loss, edge_loss, perceptual_loss)
        """
        # Compute individual losses
        l1 = self.l1_loss(pred, target)
        edge = self.edge_loss(pred, target)
        perceptual = self.perceptual_loss(pred, target)
        
        # Combined loss
        total = (self.l1_weight * l1 + 
                self.edge_weight * edge + 
                self.perceptual_weight * perceptual)
        
        return total, (l1, edge, perceptual)


# ============================================================================
# TESTING AND VISUALIZATION
# ============================================================================

def test_edge_loss():
    """Test edge-aware loss implementation"""
    print("=" * 60)
    print("TESTING EDGE-AWARE LOSS")
    print("=" * 60)
    
    # Create dummy data
    batch_size = 2
    channels = 3
    height, width = 256, 256
    
    pred = torch.randn(batch_size, channels, height, width)
    target = torch.randn(batch_size, channels, height, width)
    
    # Test EdgeAwareLoss
    print("\n1. Testing EdgeAwareLoss...")
    edge_loss = EdgeAwareLoss(edge_weight=0.5)
    
    loss = edge_loss(pred, target)
    print(f"✓ Edge loss computed: {loss.item():.6f}")
    
    # Test CombinedEdgeLoss
    print("\n2. Testing CombinedEdgeLoss...")
    combined_loss = CombinedEdgeLoss(
        l1_weight=1.0,
        edge_weight=0.5,
        perceptual_weight=0.01
    )
    
    total, (l1, edge, perc) = combined_loss(pred, target)
    print(f"✓ Total loss: {total.item():.6f}")
    print(f"  - L1: {l1.item():.6f}")
    print(f"  - Edge: {edge.item():.6f}")
    print(f"  - Perceptual: {perc.item():.6f}")
    
    print("\n" + "=" * 60)
    print("EDGE-AWARE LOSS TEST PASSED")
    print("=" * 60)


def visualize_edge_maps(image_path=None):
    """Visualize edge detection on a sample image"""
    import matplotlib.pyplot as plt
    import numpy as np
    from PIL import Image
    
    print("\n" + "=" * 60)
    print("VISUALIZING EDGE DETECTION")
    print("=" * 60)
    
    # Create or load test image
    if image_path and os.path.exists(image_path):
        img = Image.open(image_path).convert('RGB')
        img = img.resize((256, 256))
    else:
        # Create synthetic test image with edges
        img = np.zeros((256, 256, 3), dtype=np.uint8)
        # Add some rectangles (simulating buildings)
        img[50:100, 50:150] = [255, 0, 0]
        img[120:200, 100:200] = [0, 255, 0]
        # Add some lines (simulating roads)
        img[110:120, :] = [255, 255, 0]
        img[:, 180:190] = [0, 0, 255]
        img = Image.fromarray(img)
    
    # Convert to tensor
    img_np = np.array(img).astype(np.float32) / 255.0
    img_tensor = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0)
    
    # Compute edges
    edge_loss = EdgeAwareLoss()
    edge_map = edge_loss.compute_gradients(img_tensor)
    
    # Convert back to numpy
    edge_map_np = edge_map[0].mean(dim=0).detach().numpy()
    
    # Visualize
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    axes[0].imshow(img)
    axes[0].set_title('Original Image', fontsize=14, fontweight='bold')
    axes[0].axis('off')
    
    axes[1].imshow(edge_map_np, cmap='hot')
    axes[1].set_title('Detected Edges (Sobel)', fontsize=14, fontweight='bold')
    axes[1].axis('off')
    
    plt.tight_layout()
    plt.savefig('edge_detection_demo.png', dpi=150, bbox_inches='tight')
    print("\n✓ Edge visualization saved to: edge_detection_demo.png")
    plt.close()
    
    print("=" * 60)


if __name__ == "__main__":
    import os
    
    # Run tests
    test_edge_loss()
    
    # Visualize edge detection
    visualize_edge_maps()