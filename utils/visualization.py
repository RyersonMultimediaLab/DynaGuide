"""
Visualization utilities for DynaGuide
"""

import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from typing import Optional, Union
import os


def create_color_palette(n_colors: int = 256, seed: int = 42) -> np.ndarray:
    """
    Create a random color palette for visualization.
    
    Args:
        n_colors: Number of colors to generate
        seed: Random seed for reproducibility
        
    Returns:
        Array of shape (n_colors, 3) with RGB values
    """
    np.random.seed(seed)
    return np.random.randint(0, 255, size=(n_colors, 3), dtype=np.uint8)


def apply_color_palette(
    labels: np.ndarray,
    palette: Optional[np.ndarray] = None,
    n_channel: int = 100
) -> np.ndarray:
    """
    Apply color palette to label map.
    
    Args:
        labels: Label map of shape (H, W)
        palette: Color palette, generated if not provided
        n_channel: Number of channels for palette generation
        
    Returns:
        RGB image of shape (H, W, 3)
    """
    if palette is None:
        palette = create_color_palette(n_channel)
    
    colored = np.array([palette[c % len(palette)] for c in labels.flatten()])
    return colored.reshape(labels.shape[0], labels.shape[1], 3).astype(np.uint8)


def visualize_segmentation(
    image: Union[np.ndarray, Image.Image],
    segmentation: np.ndarray,
    pseudo_labels: Optional[np.ndarray] = None,
    title: str = "DynaGuide Segmentation",
    save_path: Optional[str] = None,
    show: bool = True
) -> None:
    """
    Visualize segmentation results.
    
    Args:
        image: Original image
        segmentation: Segmentation mask
        pseudo_labels: Optional pseudo-labels for comparison
        title: Plot title
        save_path: Path to save figure
        show: Whether to display figure
    """
    if isinstance(image, Image.Image):
        image = np.array(image)
    
    n_plots = 3 if pseudo_labels is not None else 2
    fig, axes = plt.subplots(1, n_plots, figsize=(5 * n_plots, 5))
    
    # Original image
    axes[0].imshow(image)
    axes[0].set_title("Original Image")
    axes[0].axis('off')
    
    # Segmentation
    seg_colored = apply_color_palette(segmentation)
    axes[1].imshow(seg_colored)
    axes[1].set_title(f"Segmentation ({len(np.unique(segmentation))} classes)")
    axes[1].axis('off')
    
    # Pseudo-labels (if provided)
    if pseudo_labels is not None:
        pseudo_colored = apply_color_palette(pseudo_labels)
        axes[2].imshow(pseudo_colored)
        axes[2].set_title(f"Pseudo-labels ({len(np.unique(pseudo_labels))} classes)")
        axes[2].axis('off')
    
    plt.suptitle(title)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    if show:
        plt.show()
    else:
        plt.close()


def save_segmentation(
    segmentation: np.ndarray,
    output_path: str,
    as_colored: bool = True,
    palette: Optional[np.ndarray] = None
) -> None:
    """
    Save segmentation mask to file.
    
    Args:
        segmentation: Segmentation mask (H, W)
        output_path: Output file path
        as_colored: Save as colored RGB image
        palette: Color palette for visualization
    """
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    
    if as_colored:
        colored = apply_color_palette(segmentation, palette)
        cv2.imwrite(output_path, cv2.cvtColor(colored, cv2.COLOR_RGB2BGR))
    else:
        cv2.imwrite(output_path, segmentation)


def create_overlay(
    image: np.ndarray,
    segmentation: np.ndarray,
    alpha: float = 0.5,
    palette: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Create overlay of segmentation on original image.
    
    Args:
        image: Original image (H, W, 3)
        segmentation: Segmentation mask (H, W)
        alpha: Blend factor
        palette: Color palette
        
    Returns:
        Blended image
    """
    seg_colored = apply_color_palette(segmentation, palette)
    
    if image.shape[:2] != segmentation.shape:
        seg_colored = cv2.resize(seg_colored, (image.shape[1], image.shape[0]))
    
    overlay = cv2.addWeighted(image, 1 - alpha, seg_colored, alpha, 0)
    return overlay


def plot_training_progress(
    losses: list,
    n_labels: list,
    save_path: Optional[str] = None,
    show: bool = True
) -> None:
    """
    Plot training progress.
    
    Args:
        losses: List of loss values
        n_labels: List of label counts
        save_path: Path to save figure
        show: Whether to display
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    ax1.plot(losses)
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training Loss')
    ax1.grid(True, alpha=0.3)
    
    ax2.plot(n_labels)
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('Number of Labels')
    ax2.set_title('Label Count')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    if show:
        plt.show()
    else:
        plt.close()
