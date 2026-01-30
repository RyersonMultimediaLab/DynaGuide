#!/usr/bin/env python3
"""
DynaGuide: Zero-Shot Guided Unsupervised Semantic Segmentation

Main training script implementing Algorithm 1 from the paper.

Reference:
    Guermazi et al., "DynaGuide: A generalizable dynamic guidance framework for 
    zero-shot guided unsupervised semantic segmentation", Image and Vision Computing, 2025.
    DOI: 10.1016/j.imavis.2025.105770

Usage:
    python train.py --input_folder /path/to/images --output_folder /path/to/output --guidance diffseg
"""

import os
import argparse
import gc
import random
import numpy as np
import cv2
import torch
import torch.optim as optim
from typing import Tuple
from PIL import Image

from dynaguide import DynaGuideNet
from dynaguide.losses import DynaGuideLoss
from dynaguide.pseudo_labels import get_pseudo_label_generator
from configs import DynaGuideConfig, BSD500_CONFIG, PASCAL_VOC_CONFIG, COCO_CONFIG
from utils.visualization import create_color_palette
from utils.io import get_image_files


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def check_gpu():
    """Check GPU availability."""
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        return True
    print("Using CPU")
    return False


def train_single_image(
    image_path: str,
    config: DynaGuideConfig,
    pseudo_label_generator=None,
    device: torch.device = None
) -> Tuple[np.ndarray, dict]:
    """
    Train DynaGuide on a single image (Algorithm 1 in paper).
    
    Args:
        image_path: Path to input image
        config: DynaGuide configuration
        pseudo_label_generator: Global pseudo-label generator (DiffSeg/SegFormer)
        device: Torch device
        
    Returns:
        Tuple of (segmentation_mask, training_metadata)
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    use_cuda = device.type == "cuda"
    
    # Clear memory
    if use_cuda:
        torch.cuda.empty_cache()
        gc.collect()
    
    # Set seed
    set_seed(config.training.seed)
    
    # Load image
    im_cv = cv2.imread(image_path)
    im_pil = Image.open(image_path).convert('RGB')
    H, W = im_cv.shape[:2]
    
    # Prepare input tensor
    data = torch.from_numpy(
        im_cv.transpose(2, 0, 1).astype('float32') / 255.0
    ).unsqueeze(0).to(device)
    
    # Generate global pseudo-labels Ŝ (frozen, static throughout training)
    pseudo_labels = None
    pseudo_mask_tensor = None
    if pseudo_label_generator is not None:
        pseudo_labels = pseudo_label_generator.generate(im_pil)
        pseudo_mask_tensor = torch.from_numpy(pseudo_labels).long().to(device)
    
    # Initialize CNN model (106.4K parameters)
    model = DynaGuideNet(
        input_dim=data.size(1),
        n_channel=config.model.n_channel,
        n_conv=config.model.n_conv
    ).to(device)
    model.train()
    
    # Initialize loss function
    loss_fn = DynaGuideLoss(
        n_channel=config.model.n_channel,
        beta=config.loss.beta,
        include_diagonal=config.loss.include_diagonal
    )
    
    # Optimizer (SGD with momentum, no weight decay)
    optimizer = optim.SGD(
        model.parameters(),
        lr=config.training.lr,
        momentum=config.training.momentum,
        weight_decay=config.training.weight_decay
    )
    
    # Color palette for visualization
    label_colours = create_color_palette(config.model.n_channel)
    
    # Training metadata
    metadata = {
        'losses': [],
        'n_labels': [],
        'final_iter': 0
    }
    
    # Training loop (Algorithm 1)
    for iteration in range(config.training.max_iter):
        optimizer.zero_grad()
        
        # Forward pass: get response map R̂
        output = model(data)[0]  # (n_channel, H, W)
        
        # Reshape for losses
        output_flat = output.permute(1, 2, 0).contiguous().view(-1, config.model.n_channel)  # (H*W, K)
        output_spatial = output.permute(1, 2, 0).contiguous()  # (H, W, K)
        
        # Get cluster labels Ĉ and count K̂
        _, labels = torch.max(output_flat, 1)
        n_labels = len(torch.unique(labels))
        
        # Compute loss (Equation 2)
        loss, loss_dict = loss_fn(
            response_map=output_flat,
            response_map_spatial=output_spatial,
            labels=labels,
            n_labels=n_labels,
            pseudo_labels=pseudo_mask_tensor
        )
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Record metadata
        metadata['losses'].append(loss_dict['total'])
        metadata['n_labels'].append(n_labels)
        
        # Visualization
        if config.visualize and iteration % 50 == 0:
            im_target = labels.cpu().numpy()
            im_target_rgb = np.array([label_colours[c % config.model.n_channel] for c in im_target])
            im_target_rgb = im_target_rgb.reshape(H, W, 3).astype(np.uint8)
            cv2.imshow("DynaGuide Output", im_target_rgb)
            cv2.waitKey(10)
        
        # Memory cleanup
        if iteration % 10 == 0 and use_cuda:
            torch.cuda.empty_cache()
        
        # Print progress
        if iteration % 100 == 0:
            print(f"  Iter {iteration}/{config.training.max_iter} | K̂={n_labels} | Loss={loss_dict['total']:.4f}")
    
    metadata['final_iter'] = config.training.max_iter
    
    # Final segmentation
    model.eval()
    with torch.no_grad():
        output = model(data)[0]
        output_flat = output.permute(1, 2, 0).contiguous().view(-1, config.model.n_channel)
        _, labels = torch.max(output_flat, 1)
        im_target = labels.cpu().numpy()
    
    # Create colored segmentation
    im_target_rgb = np.array([label_colours[c % config.model.n_channel] for c in im_target])
    segmentation = im_target_rgb.reshape(H, W, 3).astype(np.uint8)
    
    cv2.destroyAllWindows()
    
    return segmentation, metadata


def main():
    parser = argparse.ArgumentParser(
        description='DynaGuide: Zero-Shot Guided Unsupervised Semantic Segmentation',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # I/O
    parser.add_argument('--input_folder', type=str, required=True,
                        help='Input image folder')
    parser.add_argument('--output_folder', type=str, required=True,
                        help='Output folder for segmentations')
    
    # Preset
    parser.add_argument('--preset', type=str, default=None,
                        choices=['bsd500', 'pascal_voc', 'coco'],
                        help='Use preset configuration')
    
    # Model
    parser.add_argument('--n_channel', type=int, default=100,
                        help='Feature/cluster dimension K')
    parser.add_argument('--n_conv', type=int, default=3,
                        help='Number of conv blocks')
    
    # Training
    parser.add_argument('--max_iter', type=int, default=1000,
                        help='Maximum iterations per image')
    parser.add_argument('--lr', type=float, default=0.1,
                        help='Learning rate')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    
    # Loss
    parser.add_argument('--beta', type=float, default=15.0,
                        help='Spatial loss scaling factor (fixed across datasets)')
    parser.add_argument('--no_diagonal', action='store_true',
                        help='Disable diagonal continuity loss')
    
    # Guidance
    parser.add_argument('--guidance', type=str, default='diffseg',
                        choices=['diffseg', 'segformer', 'dino', 'none'],
                        help='Global pseudo-label source')
    
    # Visualization
    parser.add_argument('--visualize', action='store_true',
                        help='Show visualization during training')
    
    args = parser.parse_args()
    
    # Build configuration
    if args.preset == 'bsd500':
        config = BSD500_CONFIG
    elif args.preset == 'pascal_voc':
        config = PASCAL_VOC_CONFIG
    elif args.preset == 'coco':
        config = COCO_CONFIG
    else:
        config = DynaGuideConfig()
    
    # Override with CLI args
    config.model.n_channel = args.n_channel
    config.model.n_conv = args.n_conv
    config.training.max_iter = args.max_iter
    config.training.lr = args.lr
    config.training.seed = args.seed
    config.loss.beta = args.beta
    config.loss.include_diagonal = not args.no_diagonal
    config.guidance.method = args.guidance
    config.visualize = args.visualize
    config.input_folder = args.input_folder
    config.output_folder = args.output_folder
    
    # Setup
    os.makedirs(args.output_folder, exist_ok=True)
    use_cuda = check_gpu()
    device = torch.device("cuda" if use_cuda else "cpu")
    
    # Initialize pseudo-label generator
    pseudo_gen = None
    if args.guidance != 'none':
        print(f"Loading {args.guidance} global generator...")
        pseudo_gen = get_pseudo_label_generator(args.guidance, device=device)
    
    # Get image files
    image_files = get_image_files(args.input_folder)
    print(f"Found {len(image_files)} images")
    print(f"Config: K={config.model.n_channel}, β={config.loss.beta}, diagonal={config.loss.include_diagonal}")
    
    # Process each image
    for idx, image_path in enumerate(image_files):
        filename = os.path.basename(image_path)
        print(f"\n[{idx+1}/{len(image_files)}] {filename}")
        
        try:
            segmentation, metadata = train_single_image(
                image_path=image_path,
                config=config,
                pseudo_label_generator=pseudo_gen,
                device=device
            )
            
            # Save
            output_path = os.path.join(
                args.output_folder,
                os.path.splitext(filename)[0] + '.png'
            )
            cv2.imwrite(output_path, segmentation)
            print(f"  Saved: {output_path} (K̂={metadata['n_labels'][-1]})")
            
        except Exception as e:
            print(f"  Error: {e}")
            continue
    
    print("\nDone!")


if __name__ == "__main__":
    main()
