"""
DynaGuide Loss Functions

Adaptive multi-component loss with dynamic weighting:
    L_total = L_sim + (K̂/β) × L_spatial + L_guide

Components:
    - L_sim: Feature Similarity Loss (CrossEntropy)
    - L_spatial: Huber-smoothed Spatial Continuity (H + V + Diagonals)
    - L_guide: Global Guidance Loss (CrossEntropy with pseudo-labels)

Reference:
    Guermazi et al., "DynaGuide: A generalizable dynamic guidance framework for 
    zero-shot guided unsupervised semantic segmentation", Image and Vision Computing, 2025.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, List


class DynaGuideLoss(nn.Module):
    """
    Dynamic multi-component loss for DynaGuide.
    
    The loss dynamically balances based on the current number of clusters K̂,
    using a fixed scaling factor β=15 that requires no dataset-specific tuning.
    
    Args:
        n_channel: Number of output channels (default: 100)
        beta: Scaling factor for spatial loss (default: 15, fixed across datasets)
        include_diagonal: Include diagonal continuity terms (default: True)
    """
    
    def __init__(
        self,
        n_channel: int = 100,
        beta: float = 15.0,
        include_diagonal: bool = True
    ):
        super(DynaGuideLoss, self).__init__()
        
        self.n_channel = n_channel
        self.beta = beta
        self.include_diagonal = include_diagonal
        
        # Loss functions
        self.ce_loss = nn.CrossEntropyLoss()
        
        # Huber loss for spatial continuity (more robust than L1)
        self.huber_loss = nn.SmoothL1Loss(reduction='mean')
    
    def feature_similarity_loss(
        self, 
        response_map: torch.Tensor, 
        labels: torch.Tensor
    ) -> torch.Tensor:
        """
        Feature Similarity Loss (Equation 3 in paper).
        
        Promotes clustering of pixels with similar features using
        cross-entropy between normalized response map and cluster labels.
        
        Args:
            response_map: R̂ of shape (N, K) where N = H*W
            labels: Ĉ of shape (N,)
            
        Returns:
            L_sim scalar
        """
        return self.ce_loss(response_map, labels)
    
    def spatial_continuity_loss(
        self,
        response_map_spatial: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Spatial Continuity Loss with Huber smoothing (Equations 4-7 in paper).
        
        Enforces smooth segmentation by minimizing differences between
        neighboring pixels in horizontal, vertical, and diagonal directions.
        
        Args:
            response_map_spatial: R̂ of shape (H, W, K)
            
        Returns:
            Tuple of (total_spatial_loss, component_dict)
        """
        H, W, K = response_map_spatial.shape
        device = response_map_spatial.device
        
        # Horizontal continuity: L_HPy (Eq. 4)
        hp_y = response_map_spatial[1:, :, :] - response_map_spatial[:-1, :, :]
        hp_y_target = torch.zeros_like(hp_y)
        loss_hp_y = self.huber_loss(hp_y, hp_y_target)
        
        # Vertical continuity: L_HPz (Eq. 5)
        hp_z = response_map_spatial[:, 1:, :] - response_map_spatial[:, :-1, :]
        hp_z_target = torch.zeros_like(hp_z)
        loss_hp_z = self.huber_loss(hp_z, hp_z_target)
        
        total_loss = loss_hp_y + loss_hp_z
        components = {
            'L_HPy': loss_hp_y.item(),
            'L_HPz': loss_hp_z.item()
        }
        
        # Diagonal continuity (Eqs. 6-7)
        if self.include_diagonal:
            # Main diagonal (↘): L_HD1 (Eq. 6)
            hd1 = response_map_spatial[1:, 1:, :] - response_map_spatial[:-1, :-1, :]
            hd1_target = torch.zeros_like(hd1)
            loss_hd1 = self.huber_loss(hd1, hd1_target)
            
            # Anti-diagonal (↙): L_HD2 (Eq. 7)
            hd2 = response_map_spatial[1:, :-1, :] - response_map_spatial[:-1, 1:, :]
            hd2_target = torch.zeros_like(hd2)
            loss_hd2 = self.huber_loss(hd2, hd2_target)
            
            total_loss = total_loss + loss_hd1 + loss_hd2
            components['L_HD1'] = loss_hd1.item()
            components['L_HD2'] = loss_hd2.item()
        
        return total_loss, components
    
    def global_guidance_loss(
        self,
        response_map: torch.Tensor,
        pseudo_labels: torch.Tensor,
        current_labels: torch.Tensor
    ) -> torch.Tensor:
        """
        Global Guidance Loss (Equation 8 in paper).
        
        Aligns local predictions with global pseudo-labels from
        DiffSeg or SegFormer using cross-entropy.
        
        Args:
            response_map: R̂ of shape (N, K) where N = H*W
            pseudo_labels: Ŝ of shape (H, W) from global generator
            current_labels: Current cluster labels for remapping
            
        Returns:
            L_guide scalar
        """
        # Flatten pseudo-labels
        pseudo_flat = pseudo_labels.view(-1)
        
        # Get unique labels from both sources
        pseudo_unique = torch.unique(pseudo_flat)
        target_unique = torch.unique(current_labels)
        
        # Create mapping from pseudo-label space to current cluster space
        label_mapping = {}
        for idx, pl in enumerate(pseudo_unique):
            label_mapping[pl.item()] = target_unique[idx % len(target_unique)].item()
        
        # Apply mapping
        pseudo_mapped = pseudo_flat.clone()
        for src, dst in label_mapping.items():
            pseudo_mapped[pseudo_flat == src] = dst
        
        return self.ce_loss(response_map, pseudo_mapped)
    
    def forward(
        self,
        response_map: torch.Tensor,
        response_map_spatial: torch.Tensor,
        labels: torch.Tensor,
        n_labels: int,
        pseudo_labels: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute total DynaGuide loss (Equation 2 in paper).
        
        L_total = L_sim + (K̂/β) × L_spatial + L_guide
        
        Args:
            response_map: Flattened response R̂ (N, K)
            response_map_spatial: Spatial response R̂ (H, W, K)
            labels: Cluster labels Ĉ (N,)
            n_labels: Current number of clusters K̂
            pseudo_labels: Optional global pseudo-labels Ŝ (H, W)
            
        Returns:
            Tuple of (total_loss, loss_dict)
        """
        # Feature Similarity Loss
        loss_sim = self.feature_similarity_loss(response_map, labels)
        
        # Spatial Continuity Loss
        loss_spatial, spatial_components = self.spatial_continuity_loss(response_map_spatial)
        
        # Dynamic weight based on cluster count (Eq. 2)
        spatial_weight = n_labels / self.beta
        
        # Global Guidance Loss (if pseudo-labels provided)
        loss_guide = torch.tensor(0.0, device=response_map.device)
        if pseudo_labels is not None:
            loss_guide = self.global_guidance_loss(response_map, pseudo_labels, labels)
        
        # Total loss (Eq. 2)
        total_loss = loss_sim + spatial_weight * loss_spatial + loss_guide
        
        # Compile loss dictionary
        loss_dict = {
            'L_sim': loss_sim.item(),
            'L_spatial': loss_spatial.item(),
            'L_guide': loss_guide.item() if pseudo_labels is not None else 0.0,
            'spatial_weight': spatial_weight,
            'total': total_loss.item(),
            **spatial_components
        }
        
        return total_loss, loss_dict


class HuberSpatialLoss(nn.Module):
    """
    Standalone Huber-smoothed spatial continuity loss.
    
    More robust than L1 loss for handling outliers while
    preserving fine-grained boundary details.
    """
    
    def __init__(self, include_diagonal: bool = True):
        super(HuberSpatialLoss, self).__init__()
        self.huber = nn.SmoothL1Loss(reduction='mean')
        self.include_diagonal = include_diagonal
    
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: Tensor of shape (H, W, C)
            
        Returns:
            Spatial continuity loss scalar
        """
        # Horizontal
        hp_y = features[1:, :, :] - features[:-1, :, :]
        loss = self.huber(hp_y, torch.zeros_like(hp_y))
        
        # Vertical
        hp_z = features[:, 1:, :] - features[:, :-1, :]
        loss = loss + self.huber(hp_z, torch.zeros_like(hp_z))
        
        if self.include_diagonal:
            # Main diagonal
            hd1 = features[1:, 1:, :] - features[:-1, :-1, :]
            loss = loss + self.huber(hd1, torch.zeros_like(hd1))
            
            # Anti-diagonal
            hd2 = features[1:, :-1, :] - features[:-1, 1:, :]
            loss = loss + self.huber(hd2, torch.zeros_like(hd2))
        
        return loss
