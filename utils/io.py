"""
I/O utilities for DynaGuide
"""

import os
import json
import numpy as np
import cv2
from PIL import Image
import torch
from typing import Union, Optional, Tuple


SUPPORTED_FORMATS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}


def load_image(
    path: str,
    as_tensor: bool = False,
    device: Optional[torch.device] = None
) -> Union[np.ndarray, torch.Tensor, Tuple[np.ndarray, Image.Image]]:
    """
    Load an image from disk.
    
    Args:
        path: Path to image file
        as_tensor: Return as PyTorch tensor normalized to [0, 1]
        device: Device to move tensor to
        
    Returns:
        Image as numpy array, PIL Image, or tensor
    """
    # Load with OpenCV (BGR format)
    im_cv = cv2.imread(path)
    if im_cv is None:
        raise ValueError(f"Could not load image: {path}")
    
    # Also load with PIL for compatibility
    im_pil = Image.open(path)
    if im_pil.mode != 'RGB':
        im_pil = im_pil.convert('RGB')
    
    if as_tensor:
        # Convert to tensor (C, H, W) normalized to [0, 1]
        tensor = torch.from_numpy(
            im_cv.transpose(2, 0, 1).astype('float32') / 255.0
        ).unsqueeze(0)
        
        if device is not None:
            tensor = tensor.to(device)
        
        return tensor, im_cv, im_pil
    
    return im_cv, im_pil


def get_image_files(folder: str) -> list:
    """
    Get all image files in a folder.
    
    Args:
        folder: Path to folder
        
    Returns:
        List of image file paths
    """
    files = []
    for f in os.listdir(folder):
        ext = os.path.splitext(f)[1].lower()
        if ext in SUPPORTED_FORMATS:
            # Skip scribble files
            if '_scribble' not in f:
                files.append(os.path.join(folder, f))
    return sorted(files)


def save_results(
    output_folder: str,
    filename: str,
    segmentation: np.ndarray,
    metadata: Optional[dict] = None
) -> str:
    """
    Save segmentation results.
    
    Args:
        output_folder: Output folder path
        filename: Base filename
        segmentation: Segmentation mask
        metadata: Optional metadata to save
        
    Returns:
        Path to saved segmentation
    """
    os.makedirs(output_folder, exist_ok=True)
    
    # Save segmentation as PNG
    base_name = os.path.splitext(os.path.basename(filename))[0]
    output_path = os.path.join(output_folder, f"{base_name}.png")
    
    cv2.imwrite(output_path, segmentation)
    
    # Save metadata if provided
    if metadata is not None:
        meta_path = os.path.join(output_folder, f"{base_name}_meta.json")
        with open(meta_path, 'w') as f:
            json.dump(metadata, f, indent=2)
    
    return output_path


def setup_output_folders(base_folder: str) -> dict:
    """
    Setup output folder structure.
    
    Args:
        base_folder: Base output folder
        
    Returns:
        Dictionary of folder paths
    """
    folders = {
        'segmentations': os.path.join(base_folder, 'segmentations'),
        'visualizations': os.path.join(base_folder, 'visualizations'),
        'logs': os.path.join(base_folder, 'logs')
    }
    
    for folder in folders.values():
        os.makedirs(folder, exist_ok=True)
    
    return folders
