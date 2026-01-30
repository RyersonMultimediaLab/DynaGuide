#!/usr/bin/env python3
"""
DynaGuide Demo Script

Simple example showing how to use DynaGuide for single image segmentation.
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt

from dynaguide import DynaGuideNet
from dynaguide.pseudo_labels import get_pseudo_label_generator
from configs import DynaGuideConfig
from utils.visualization import visualize_segmentation, create_color_palette


def demo_single_image(image_path: str, guidance: str = "dino"):
    """
    Run DynaGuide on a single image.
    
    Args:
        image_path: Path to input image
        guidance: Pseudo-label method ('segformer', 'dino', 'dinov2', 'none')
    """
    print(f"Processing: {image_path}")
    print(f"Guidance method: {guidance}")
    
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # Load image
    im_cv = cv2.imread(image_path)
    im_pil = Image.open(image_path).convert('RGB')
    
    # Initialize pseudo-label generator
    pseudo_gen = None
    pseudo_labels = None
    if guidance != "none":
        print(f"Loading {guidance} model...")
        pseudo_gen = get_pseudo_label_generator(guidance, device=device)
        pseudo_labels = pseudo_gen.generate(im_pil)
        print(f"Generated pseudo-labels with {len(np.unique(pseudo_labels))} classes")
    
    # Configuration
    config = DynaGuideConfig()
    config.model.n_channel = 100
    config.training.max_iter = 200  # Fewer iterations for demo
    config.training.min_labels = 3
    
    # Prepare input
    data = torch.from_numpy(
        im_cv.transpose(2, 0, 1).astype('float32') / 255.0
    ).unsqueeze(0).to(device)
    
    # Initialize model
    model = DynaGuideNet(
        input_dim=3,
        n_channel=config.model.n_channel,
        n_conv=config.model.n_conv
    ).to(device)
    
    # Loss and optimizer
    loss_fn = torch.nn.CrossEntropyLoss()
    loss_cont = torch.nn.SmoothL1Loss(reduction='mean')
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
    
    H, W = im_cv.shape[:2]
    n_channel = config.model.n_channel
    
    # Continuity targets
    HPy_target = torch.zeros(H - 1, W, n_channel, device=device)
    HPz_target = torch.zeros(H, W - 1, n_channel, device=device)
    
    # Training
    print("\nTraining...")
    model.train()
    
    losses = []
    n_labels_history = []
    
    for iteration in range(config.training.max_iter):
        optimizer.zero_grad()
        
        output = model(data)[0]
        output_flat = output.permute(1, 2, 0).contiguous().view(-1, n_channel)
        output_spatial = output_flat.view(H, W, n_channel)
        
        # Losses
        _, target = torch.max(output_flat, 1)
        loss_sim = loss_fn(output_flat, target)
        
        HPy = output_spatial[1:, :, :] - output_spatial[:-1, :, :]
        HPz = output_spatial[:, 1:, :] - output_spatial[:, :-1, :]
        loss_con = loss_cont(HPy, HPy_target) + loss_cont(HPz, HPz_target)
        
        n_labels = len(torch.unique(target))
        loss = loss_sim + (n_labels / 15.0) * loss_con
        
        loss.backward()
        optimizer.step()
        
        losses.append(loss.item())
        n_labels_history.append(n_labels)
        
        if iteration % 20 == 0:
            print(f"  Iter {iteration}: loss={loss.item():.4f}, labels={n_labels}")
        
        if n_labels <= config.training.min_labels:
            print(f"  Early stopping at iter {iteration}")
            break
    
    # Get final segmentation
    model.eval()
    with torch.no_grad():
        output = model(data)[0]
        output_flat = output.permute(1, 2, 0).contiguous().view(-1, n_channel)
        _, labels = torch.max(output_flat, 1)
        labels = labels.cpu().numpy()
    
    # Colorize
    palette = create_color_palette(n_channel)
    segmentation = np.array([palette[c % n_channel] for c in labels])
    segmentation = segmentation.reshape(H, W, 3)
    
    # Visualize results
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Original
    axes[0, 0].imshow(im_pil)
    axes[0, 0].set_title("Original Image")
    axes[0, 0].axis('off')
    
    # Pseudo-labels
    if pseudo_labels is not None:
        pseudo_colored = np.array([palette[c % n_channel] for c in pseudo_labels.flatten()])
        pseudo_colored = pseudo_colored.reshape(H, W, 3)
        axes[0, 1].imshow(pseudo_colored)
        axes[0, 1].set_title(f"Pseudo-labels ({guidance})")
    else:
        axes[0, 1].text(0.5, 0.5, "No guidance", ha='center', va='center')
    axes[0, 1].axis('off')
    
    # Segmentation
    axes[1, 0].imshow(segmentation)
    axes[1, 0].set_title(f"DynaGuide ({len(np.unique(labels))} segments)")
    axes[1, 0].axis('off')
    
    # Training curves
    ax_loss = axes[1, 1]
    ax_loss.plot(losses, 'b-', label='Loss')
    ax_loss.set_xlabel('Iteration')
    ax_loss.set_ylabel('Loss', color='b')
    ax_loss.tick_params(axis='y', labelcolor='b')
    
    ax_labels = ax_loss.twinx()
    ax_labels.plot(n_labels_history, 'r-', label='Labels')
    ax_labels.set_ylabel('# Labels', color='r')
    ax_labels.tick_params(axis='y', labelcolor='r')
    
    axes[1, 1].set_title("Training Progress")
    
    plt.tight_layout()
    plt.savefig("demo_output.png", dpi=150)
    plt.show()
    
    print(f"\nDone! Output saved to demo_output.png")
    print(f"Final segments: {len(np.unique(labels))}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="DynaGuide Demo")
    parser.add_argument("--image", type=str, required=True, help="Input image path")
    parser.add_argument("--guidance", type=str, default="dino",
                        choices=["segformer", "dino", "dinov2", "diffseg", "none"],
                        help="Pseudo-label guidance method")
    
    args = parser.parse_args()
    
    demo_single_image(args.image, args.guidance)
