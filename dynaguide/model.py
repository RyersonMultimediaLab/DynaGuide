"""
DynaGuide CNN Architecture

Lightweight segmentation network (106.4K parameters) trained per-image.
Architecture: 3 conv blocks with batch normalization, ReLU, and skip connections.

Reference:
    Guermazi et al., "DynaGuide: A generalizable dynamic guidance framework for 
    zero-shot guided unsupervised semantic segmentation", Image and Vision Computing, 2025.
"""

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class DynaGuideNet(nn.Module):
    """
    Lightweight CNN for unsupervised segmentation with skip connections.
    
    The backbone maintains input resolution throughout, preserving spatial 
    information for accurate boundary delineation. Residual connections 
    address vanishing gradients and facilitate stable convergence.
    
    Architecture:
        - 3 convolutional blocks (Conv + BatchNorm + ReLU)
        - Skip connection from input conv to output
        - 1x1 conv for final response map
    
    Args:
        input_dim: Number of input channels (3 for RGB)
        n_channel: Number of feature/cluster channels (default: 100)
        n_conv: Number of convolutional blocks (default: 3)
    
    Total parameters: ~106.4K (0.11M)
    GFLOPs: ~6.99
    """
    
    def __init__(self, input_dim: int = 3, n_channel: int = 100, n_conv: int = 3):
        super(DynaGuideNet, self).__init__()
        
        self.n_channel = n_channel
        self.n_conv = n_conv
        
        # Initial convolution
        self.conv1 = nn.Conv2d(input_dim, n_channel, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(n_channel)
        
        # Intermediate convolutions (conv blocks 2 to n_conv)
        self.conv_layers = nn.ModuleList()
        self.bn_layers = nn.ModuleList()
        for _ in range(n_conv - 1):
            self.conv_layers.append(
                nn.Conv2d(n_channel, n_channel, kernel_size=3, stride=1, padding=1)
            )
            self.bn_layers.append(nn.BatchNorm2d(n_channel))
        
        # Output 1x1 convolution for response map
        self.conv_out = nn.Conv2d(n_channel, n_channel, kernel_size=1, stride=1, padding=0)
        self.bn_out = nn.BatchNorm2d(n_channel)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with skip connection.
        
        Args:
            x: Input tensor of shape (B, C, H, W)
            
        Returns:
            Normalized response map R̂ of shape (B, n_channel, H, W)
        """
        # Initial conv (also serves as skip connection source)
        residual = self.conv1(x)
        
        x = self.conv1(x)
        x = F.relu(x)
        x = self.bn1(x)
        
        # Intermediate conv blocks
        for conv, bn in zip(self.conv_layers, self.bn_layers):
            x = conv(x)
            x = F.relu(x)
            x = bn(x)
        
        # Skip connection
        x = x + residual
        
        # Output response map
        x = self.conv_out(x)
        x = self.bn_out(x)
        
        return x
    
    def get_cluster_labels(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, int]:
        """
        Get cluster labels from network output.
        
        Implements Equation (1) from the paper:
            Ĉ_i = argmax_c R̂_i,c
        
        Args:
            x: Input image tensor (B, C, H, W)
            
        Returns:
            Tuple of:
                - output_flat: Flattened response map (H*W, n_channel)
                - labels: Cluster labels (H*W,)
                - n_labels: Number of unique clusters K̂
        """
        output = self.forward(x)
        B, C, H, W = output.shape
        
        # Flatten to (H*W, n_channel)
        output_flat = output.permute(0, 2, 3, 1).contiguous().view(-1, C)
        
        # Assign cluster labels (argmax)
        _, labels = torch.max(output_flat, dim=1)
        
        # Count unique labels (dynamic K̂)
        n_labels = len(torch.unique(labels))
        
        return output_flat, labels, n_labels
    
    def get_spatial_output(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get spatially-arranged output for continuity loss computation.
        
        Args:
            x: Input image tensor (B, C, H, W)
            
        Returns:
            Response map reshaped to (H, W, n_channel)
        """
        output = self.forward(x)
        B, C, H, W = output.shape
        
        # Reshape to (H, W, C) for spatial loss computation
        return output[0].permute(1, 2, 0).contiguous()


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters in model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Test model
    model = DynaGuideNet(input_dim=3, n_channel=100, n_conv=3)
    
    # Count parameters
    n_params = count_parameters(model)
    print(f"DynaGuideNet parameters: {n_params:,} ({n_params/1e6:.3f}M)")
    
    # Test forward pass
    x = torch.randn(1, 3, 224, 224)
    output = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    
    # Test cluster labels
    output_flat, labels, n_labels = model.get_cluster_labels(x)
    print(f"Output flat shape: {output_flat.shape}")
    print(f"Labels shape: {labels.shape}")
    print(f"Number of clusters: {n_labels}")
