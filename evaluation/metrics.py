"""
Evaluation metrics for DynaGuide segmentation
"""

import os
import numpy as np
from PIL import Image
from sklearn.metrics import confusion_matrix
from typing import Optional
import warnings


def calculate_iou(gt_mask: np.ndarray, pred_mask: np.ndarray) -> float:
    """
    Calculate Intersection over Union for binary masks.
    
    Args:
        gt_mask: Ground truth binary mask
        pred_mask: Predicted binary mask
        
    Returns:
        IoU score
    """
    intersection = np.sum(gt_mask & pred_mask)
    union = np.sum(gt_mask | pred_mask)
    return intersection / union if union != 0 else 0.0


def calculate_miou(gt_mask: np.ndarray, pred_mask: np.ndarray) -> float:
    """
    Calculate mean Intersection over Union across all classes.
    
    Args:
        gt_mask: Ground truth segmentation mask
        pred_mask: Predicted segmentation mask
        
    Returns:
        mIoU score
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        conf_matrix = confusion_matrix(gt_mask.flatten(), pred_mask.flatten())
    
    # IoU per class
    iou_per_class = (
        np.diag(conf_matrix) / 
        (np.sum(conf_matrix, axis=1) + np.sum(conf_matrix, axis=0) - np.diag(conf_matrix))
    )
    
    return np.nanmean(iou_per_class)


def match_classes(gt_mask: np.ndarray, pred_mask: np.ndarray) -> np.ndarray:
    """
    Match predicted classes to ground truth classes using Hungarian algorithm.
    Uses IoU-based greedy matching.
    
    Args:
        gt_mask: Ground truth segmentation mask
        pred_mask: Predicted segmentation mask
        
    Returns:
        Remapped prediction mask with matched classes
    """
    class_mapping = {}
    
    for pred_class in np.unique(pred_mask):
        pred_mask_binary = (pred_mask == pred_class)
        
        best_matching_class = None
        highest_iou = 0.0
        
        for gt_class in np.unique(gt_mask):
            gt_mask_binary = (gt_mask == gt_class)
            iou = calculate_iou(gt_mask_binary, pred_mask_binary)
            
            if iou > highest_iou:
                highest_iou = iou
                best_matching_class = gt_class
        
        class_mapping[pred_class] = best_matching_class
    
    # Apply mapping
    matched_mask = np.vectorize(class_mapping.get)(pred_mask)
    
    return matched_mask


def evaluate_single_image(
    gt_path: str,
    pred_path: str,
    match_classes_first: bool = True
) -> dict:
    """
    Evaluate a single image prediction.
    
    Args:
        gt_path: Path to ground truth mask
        pred_path: Path to predicted mask
        match_classes_first: Whether to match classes before computing mIoU
        
    Returns:
        Dictionary with evaluation metrics
    """
    gt_image = Image.open(gt_path)
    pred_image = Image.open(pred_path)
    
    gt_mask = np.array(gt_image.convert("L"))
    pred_mask = np.array(pred_image.convert("L"))
    
    if gt_mask.shape != pred_mask.shape:
        # Resize prediction to match ground truth
        pred_image = pred_image.resize(gt_image.size, Image.NEAREST)
        pred_mask = np.array(pred_image.convert("L"))
    
    if match_classes_first:
        matched_mask = match_classes(gt_mask, pred_mask)
    else:
        matched_mask = pred_mask
    
    miou = calculate_miou(gt_mask, matched_mask)
    n_classes_gt = len(np.unique(gt_mask))
    n_classes_pred = len(np.unique(pred_mask))
    
    return {
        "miou": miou,
        "n_classes_gt": n_classes_gt,
        "n_classes_pred": n_classes_pred
    }


def evaluate_folder(
    gt_folder: str,
    pred_folder: str,
    output_file: Optional[str] = None,
    match_classes_first: bool = True
) -> dict:
    """
    Evaluate all predictions in a folder.
    
    Args:
        gt_folder: Path to ground truth masks folder
        pred_folder: Path to predicted masks folder
        output_file: Optional path to save results
        match_classes_first: Whether to match classes before computing mIoU
        
    Returns:
        Dictionary with aggregate metrics
    """
    gt_files = set(os.listdir(gt_folder))
    pred_files = os.listdir(pred_folder)
    
    results = []
    total_miou = 0.0
    processed = 0
    
    for pred_file in pred_files:
        # Try to find matching ground truth file
        gt_file = None
        
        # Exact match
        if pred_file in gt_files:
            gt_file = pred_file
        else:
            # Try without extension
            pred_base = os.path.splitext(pred_file)[0]
            for gf in gt_files:
                gt_base = os.path.splitext(gf)[0]
                if pred_base == gt_base:
                    gt_file = gf
                    break
        
        if gt_file is None:
            continue
        
        gt_path = os.path.join(gt_folder, gt_file)
        pred_path = os.path.join(pred_folder, pred_file)
        
        try:
            metrics = evaluate_single_image(gt_path, pred_path, match_classes_first)
            results.append({
                "filename": pred_file,
                **metrics
            })
            total_miou += metrics["miou"]
            processed += 1
            print(f"{pred_file}: mIoU = {metrics['miou']:.4f}")
        except Exception as e:
            print(f"Error processing {pred_file}: {e}")
    
    avg_miou = total_miou / processed if processed > 0 else 0.0
    
    summary = {
        "average_miou": avg_miou,
        "total_images": processed,
        "results": results
    }
    
    print(f"\n{'='*50}")
    print(f"Average mIoU: {avg_miou:.4f}")
    print(f"Total images evaluated: {processed}")
    
    if output_file:
        import json
        with open(output_file, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"Results saved to: {output_file}")
    
    return summary


def compute_pixel_accuracy(gt_mask: np.ndarray, pred_mask: np.ndarray) -> float:
    """
    Compute pixel-wise accuracy.
    
    Args:
        gt_mask: Ground truth mask
        pred_mask: Predicted mask (should be class-matched)
        
    Returns:
        Pixel accuracy
    """
    correct = np.sum(gt_mask == pred_mask)
    total = gt_mask.size
    return correct / total


def compute_dice_coefficient(gt_mask: np.ndarray, pred_mask: np.ndarray) -> float:
    """
    Compute Dice coefficient (F1 score for segmentation).
    
    Args:
        gt_mask: Ground truth binary mask
        pred_mask: Predicted binary mask
        
    Returns:
        Dice coefficient
    """
    intersection = np.sum(gt_mask & pred_mask)
    return (2 * intersection) / (np.sum(gt_mask) + np.sum(pred_mask) + 1e-8)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate DynaGuide predictions")
    parser.add_argument("--gt_folder", required=True, help="Ground truth folder")
    parser.add_argument("--pred_folder", required=True, help="Predictions folder")
    parser.add_argument("--output", default=None, help="Output JSON file")
    
    args = parser.parse_args()
    
    evaluate_folder(args.gt_folder, args.pred_folder, args.output)
