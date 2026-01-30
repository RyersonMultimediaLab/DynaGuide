"""
Pseudo-label generators for DynaGuide

Supports multiple guidance sources:
1. SegFormer (supervised pretrained)
2. DINO/DINOv2 (self-supervised)
3. DiffSeg (diffusion-based unsupervised)
"""

import torch
import torch.nn as nn
import numpy as np
import cv2
from PIL import Image
from abc import ABC, abstractmethod
from typing import Optional, Union


class PseudoLabelGenerator(ABC):
    """Base class for pseudo-label generators."""
    
    @abstractmethod
    def generate(self, image: Union[Image.Image, np.ndarray]) -> np.ndarray:
        """Generate pseudo-labels for an image."""
        pass
    
    def to_tensor(self, mask: np.ndarray, device: torch.device) -> torch.Tensor:
        """Convert numpy mask to tensor."""
        return torch.from_numpy(mask).long().to(device)


class SegFormerGuide(PseudoLabelGenerator):
    """
    SegFormer-based pseudo-label generator.
    Uses a pretrained semantic segmentation model as frozen prior.
    """
    
    def __init__(
        self,
        model_name: str = "nvidia/segformer-b2-finetuned-ade-512-512",
        device: Optional[torch.device] = None
    ):
        from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor
        
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = SegformerImageProcessor.from_pretrained(model_name)
        self.model = SegformerForSemanticSegmentation.from_pretrained(model_name)
        self.model.to(self.device).eval()
        
        # Freeze model
        for param in self.model.parameters():
            param.requires_grad = False
    
    def generate(self, image: Union[Image.Image, np.ndarray]) -> np.ndarray:
        """Generate semantic segmentation pseudo-labels."""
        if isinstance(image, np.ndarray):
            image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        original_size = image.size  # (W, H)
        
        inputs = self.processor(images=image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
        
        # Upsample to original size
        upsampled = nn.functional.interpolate(
            logits,
            size=(original_size[1], original_size[0]),  # (H, W)
            mode='bilinear',
            align_corners=False
        )
        
        # Get predictions
        pred = upsampled.argmax(dim=1).squeeze().cpu().numpy()
        
        return pred.astype(np.uint8)


class DINOGuide(PseudoLabelGenerator):
    """
    DINO/DINOv2 attention-based pseudo-label generator.
    Extracts attention maps from self-supervised ViT.
    """
    
    def __init__(
        self,
        model_name: str = "facebook/dino-vits16",
        device: Optional[torch.device] = None,
        threshold: str = "otsu"  # 'otsu' or float value
    ):
        from transformers import ViTFeatureExtractor, ViTModel
        
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.feature_extractor = ViTFeatureExtractor.from_pretrained(model_name)
        self.model = ViTModel.from_pretrained(model_name)
        self.model.to(self.device).eval()
        self.threshold = threshold
        
        # Freeze model
        for param in self.model.parameters():
            param.requires_grad = False
    
    def generate(self, image: Union[Image.Image, np.ndarray]) -> np.ndarray:
        """Generate attention-based pseudo-labels."""
        if isinstance(image, np.ndarray):
            image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        original_size = image.size  # (W, H)
        
        inputs = self.feature_extractor(images=image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(**inputs, output_attentions=True)
        
        # Get attention from last layer, CLS token to patches
        attentions = outputs.attentions[-1]  # (1, n_heads, seq_len, seq_len)
        cls_attn = attentions[0, :, 0, 1:]  # Remove CLS-to-CLS attention
        
        # Average across heads
        cls_attn = cls_attn.mean(dim=0)
        
        # Reshape to spatial grid
        size = int(np.sqrt(cls_attn.shape[0]))
        attention_map = cls_attn.reshape(size, size).cpu().numpy()
        
        # Normalize
        attention_map = (attention_map - attention_map.min()) / (attention_map.max() - attention_map.min() + 1e-8)
        attention_map = (attention_map * 255).astype(np.uint8)
        
        # Resize to original
        attention_map = cv2.resize(attention_map, original_size, interpolation=cv2.INTER_LINEAR)
        
        # Apply Gaussian blur
        attention_map = cv2.GaussianBlur(attention_map, (3, 3), 0)
        
        # Threshold
        if self.threshold == "otsu":
            _, mask = cv2.threshold(attention_map, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        else:
            _, mask = cv2.threshold(attention_map, int(self.threshold * 255), 255, cv2.THRESH_BINARY)
        
        # Morphological cleanup
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        return mask.astype(np.uint8)


class DINOv2Guide(PseudoLabelGenerator):
    """
    DINOv2-based pseudo-label generator with improved features.
    """
    
    def __init__(
        self,
        model_name: str = "facebook/dinov2-base",
        device: Optional[torch.device] = None,
        n_clusters: int = 5
    ):
        from transformers import AutoImageProcessor, AutoModel
        from sklearn.cluster import KMeans
        
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.to(self.device).eval()
        self.n_clusters = n_clusters
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        
        # Freeze model
        for param in self.model.parameters():
            param.requires_grad = False
    
    def generate(self, image: Union[Image.Image, np.ndarray]) -> np.ndarray:
        """Generate cluster-based pseudo-labels from DINOv2 features."""
        if isinstance(image, np.ndarray):
            image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        original_size = image.size  # (W, H)
        
        inputs = self.processor(images=image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            features = outputs.last_hidden_state[:, 1:, :]  # Remove CLS token
        
        # Reshape features
        B, N, C = features.shape
        size = int(np.sqrt(N))
        features_np = features.squeeze().cpu().numpy()
        
        # K-means clustering
        labels = self.kmeans.fit_predict(features_np)
        labels = labels.reshape(size, size)
        
        # Resize to original
        labels = cv2.resize(labels.astype(np.float32), original_size, interpolation=cv2.INTER_NEAREST)
        
        return labels.astype(np.uint8)


class DiffSegGuide(PseudoLabelGenerator):
    """
    DiffSeg-style diffusion-based pseudo-label generator.
    Fully unsupervised approach using diffusion model features.
    
    Note: This is a simplified version. Full DiffSeg requires
    the complete diffusion model setup.
    """
    
    def __init__(
        self,
        model_name: str = "stabilityai/stable-diffusion-2-1",
        device: Optional[torch.device] = None,
        n_clusters: int = 5,
        timestep: int = 50
    ):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.n_clusters = n_clusters
        self.timestep = timestep
        self._model_loaded = False
        self.model_name = model_name
        
        # Lazy loading to avoid memory issues
        self.pipe = None
    
    def _load_model(self):
        """Lazy load the diffusion model."""
        if self._model_loaded:
            return
        
        try:
            from diffusers import StableDiffusionPipeline
            self.pipe = StableDiffusionPipeline.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16
            )
            self.pipe.to(self.device)
            self._model_loaded = True
        except ImportError:
            print("Warning: diffusers not installed. Using fallback clustering.")
            self._model_loaded = False
    
    def generate(self, image: Union[Image.Image, np.ndarray]) -> np.ndarray:
        """Generate diffusion-based pseudo-labels."""
        if isinstance(image, np.ndarray):
            image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        original_size = image.size
        
        # Fallback: use color-based clustering if diffusion model not available
        self._load_model()
        
        if not self._model_loaded:
            return self._fallback_clustering(image, original_size)
        
        # TODO: Implement full DiffSeg feature extraction
        # This requires extracting intermediate features from the diffusion process
        return self._fallback_clustering(image, original_size)
    
    def _fallback_clustering(self, image: Image.Image, original_size: tuple) -> np.ndarray:
        """Fallback using superpixel + feature clustering."""
        from sklearn.cluster import KMeans
        
        img_np = np.array(image)
        
        # Convert to LAB color space for better clustering
        img_lab = cv2.cvtColor(img_np, cv2.COLOR_RGB2LAB)
        
        # Flatten and cluster
        pixels = img_lab.reshape(-1, 3).astype(np.float32)
        kmeans = KMeans(n_clusters=self.n_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(pixels)
        
        # Reshape to image
        mask = labels.reshape(img_np.shape[0], img_np.shape[1])
        
        return mask.astype(np.uint8)


def get_pseudo_label_generator(
    method: str = "segformer",
    **kwargs
) -> PseudoLabelGenerator:
    """
    Factory function to get pseudo-label generator.
    
    Args:
        method: One of 'segformer', 'dino', 'dinov2', 'diffseg'
        **kwargs: Additional arguments for the generator
        
    Returns:
        PseudoLabelGenerator instance
    """
    generators = {
        "segformer": SegFormerGuide,
        "dino": DINOGuide,
        "dinov2": DINOv2Guide,
        "diffseg": DiffSegGuide
    }
    
    if method not in generators:
        raise ValueError(f"Unknown method: {method}. Choose from {list(generators.keys())}")
    
    return generators[method](**kwargs)
