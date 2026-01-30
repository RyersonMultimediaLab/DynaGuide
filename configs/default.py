"""
DynaGuide Configuration

Default hyperparameters from the paper:
    - n_channel = 100 (feature/cluster dimension)
    - n_conv = 3 (number of conv blocks)
    - lr = 0.1 (SGD learning rate)
    - momentum = 0.9
    - beta = 15 (fixed spatial loss scaling, no tuning needed)
    - max_iter = 1000

Reference:
    Guermazi et al., "DynaGuide: A generalizable dynamic guidance framework for 
    zero-shot guided unsupervised semantic segmentation", Image and Vision Computing, 2025.
"""

from dataclasses import dataclass, field
from typing import Optional

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal


@dataclass
class ModelConfig:
    """Model architecture configuration."""
    n_channel: int = 100      # Feature/cluster dimension K
    n_conv: int = 2           # Number of conv layers (gives ~106.4K params)
    

@dataclass
class TrainingConfig:
    """Training configuration."""
    max_iter: int = 1000      # Maximum iterations per image
    lr: float = 0.1           # Learning rate (SGD)
    momentum: float = 0.9     # SGD momentum
    weight_decay: float = 0.0 # No weight decay used in paper
    seed: int = 42            # Random seed


@dataclass
class LossConfig:
    """Loss function configuration (Section 3.4 of paper)."""
    # Spatial loss scaling factor β (Eq. 2)
    # Fixed at 15 across all datasets - no tuning required
    beta: float = 15.0
    
    # Include diagonal continuity (L_HD1 + L_HD2)
    include_diagonal: bool = True


@dataclass
class GuidanceConfig:
    """Pseudo-label guidance configuration."""
    method: Literal["diffseg", "segformer", "dino", "none"] = "diffseg"
    
    # SegFormer model (used as frozen prior on unseen data)
    segformer_model: str = "nvidia/segformer-b2-finetuned-ade-512-512"
    
    # DINO model (for attention-based pseudo-labels)
    dino_model: str = "facebook/dino-vits16"


@dataclass
class DynaGuideConfig:
    """Complete DynaGuide configuration."""
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    guidance: GuidanceConfig = field(default_factory=GuidanceConfig)
    
    # I/O
    input_folder: str = ""
    output_folder: str = ""
    
    # Visualization
    visualize: bool = True
    
    # Device
    device: str = "cuda"
    
    @classmethod
    def from_dict(cls, config_dict: dict) -> "DynaGuideConfig":
        """Create config from dictionary."""
        model_cfg = ModelConfig(**config_dict.get("model", {}))
        training_cfg = TrainingConfig(**config_dict.get("training", {}))
        loss_cfg = LossConfig(**config_dict.get("loss", {}))
        guidance_cfg = GuidanceConfig(**config_dict.get("guidance", {}))
        
        return cls(
            model=model_cfg,
            training=training_cfg,
            loss=loss_cfg,
            guidance=guidance_cfg,
            input_folder=config_dict.get("input_folder", ""),
            output_folder=config_dict.get("output_folder", ""),
            visualize=config_dict.get("visualize", True),
            device=config_dict.get("device", "cuda")
        )


# ============================================================================
# Preset configurations for different datasets (Table 1 in paper)
# ============================================================================

BSD500_CONFIG = DynaGuideConfig(
    model=ModelConfig(n_channel=100, n_conv=3),
    training=TrainingConfig(max_iter=1000, lr=0.1),
    loss=LossConfig(beta=15.0, include_diagonal=True),
    guidance=GuidanceConfig(method="diffseg")
)

PASCAL_VOC_CONFIG = DynaGuideConfig(
    model=ModelConfig(n_channel=100, n_conv=3),
    training=TrainingConfig(max_iter=1000, lr=0.1),
    loss=LossConfig(beta=15.0, include_diagonal=True),
    guidance=GuidanceConfig(method="diffseg")
)

COCO_CONFIG = DynaGuideConfig(
    model=ModelConfig(n_channel=100, n_conv=3),
    training=TrainingConfig(max_iter=1000, lr=0.1),
    loss=LossConfig(beta=15.0, include_diagonal=True),
    guidance=GuidanceConfig(method="diffseg")
)

# With SegFormer guidance (higher mIoU, see Table 1)
BSD500_SEGFORMER_CONFIG = DynaGuideConfig(
    model=ModelConfig(n_channel=100, n_conv=3),
    training=TrainingConfig(max_iter=1000, lr=0.1),
    loss=LossConfig(beta=15.0, include_diagonal=True),
    guidance=GuidanceConfig(method="segformer")
)
