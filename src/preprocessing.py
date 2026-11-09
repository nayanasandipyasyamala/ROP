"""
Stage 4: Standardized Image Preprocessing & Augmentation Pipeline.

Features:
- Robust retinal field detection & black border cropping (thresholding > 15).
- Aspect-ratio preserving square padding with black borders.
- Standardized resizing (default 256x256, model input 224x224).
- Safe, clinically motivated domain augmentations (training only):
  * Horizontal and vertical flips.
  * Mild rotation (-15 to +15 deg).
  * Color jitter (brightness, contrast, saturation).
  * Mild Gaussian blur (simulating media opacity / camera focus shifts).
- Strict deterministic transforms for validation and testing (NO augmentation).
- ImageNet normalization.
"""

import sys
from pathlib import Path
from typing import Tuple, Optional

import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image

sys.path.append(str(Path(__file__).resolve().parent.parent))
from configs import config


def crop_retinal_field(im: Image.Image, threshold: int = 15) -> Image.Image:
    """
    Remove unnecessary black surrounding borders while preserving
    the full circular retinal fundus field.
    """
    arr = np.asarray(im)
    if arr.ndim == 3:
        mask = arr.max(axis=2) > threshold
    else:
        mask = arr > threshold
        
    ys, xs = np.where(mask)
    if len(ys) > 0 and len(xs) > 0:
        ymin, ymax = ys.min(), ys.max() + 1
        xmin, xmax = xs.min(), xs.max() + 1
        return im.crop((xmin, ymin, xmax, ymax))
    return im


def pad_to_square(im: Image.Image, fill: Tuple[int, int, int] = (0, 0, 0)) -> Image.Image:
    """
    Pad rectangular image to a square canvas preserving aspect ratio.
    """
    w, h = im.size
    s = max(w, h)
    canvas = Image.new("RGB", (s, s), fill)
    canvas.paste(im, ((s - w) // 2, (s - h) // 2))
    return canvas


def preprocess_fundus_image(
    image_input,
    target_size: int = config.IMG_SIZE,
    crop_borders: bool = True
) -> Image.Image:
    """
    Full standard preprocessing:
    1. Read RGB.
    2. Crop background black borders.
    3. Pad to square.
    4. Lanczos high-quality resize.
    """
    if isinstance(image_input, (str, Path)):
        im = Image.open(image_input).convert("RGB")
    elif isinstance(image_input, Image.Image):
        im = image_input.convert("RGB")
    else:
        raise ValueError(f"Unsupported image input type: {type(image_input)}")

    if crop_borders:
        im = crop_retinal_field(im)

    im = pad_to_square(im)
    im = im.resize((target_size, target_size), Image.Resampling.LANCZOS)
    return im


class PreprocessingTransform:
    """Callable transform wrapping retinal crop and pad before tensor conversion."""
    def __init__(self, target_size: int = config.IMG_SIZE):
        self.target_size = target_size

    def __call__(self, img: Image.Image) -> Image.Image:
        return preprocess_fundus_image(img, target_size=self.target_size)


def get_transforms(
    img_size: int = config.IMG_SIZE,
    is_training: bool = True,
    strong_domain_aug: bool = False
) -> T.Compose:
    """
    Returns torchvision transforms for training vs validation/testing.
    Validation/Testing: ALWAYS deterministic (NO random augmentation).
    Training: Safe clinical augmentations.
    """
    normalize = T.Normalize(mean=config.IMAGENET_MEAN, std=config.IMAGENET_STD)

    if not is_training:
        return T.Compose([
            PreprocessingTransform(target_size=img_size),
            T.ToTensor(),
            normalize
        ])

    # Standard training augmentations
    aug_list = [
        PreprocessingTransform(target_size=img_size),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomVerticalFlip(p=0.5),
        T.RandomRotation(degrees=15, interpolation=T.InterpolationMode.BILINEAR),
        T.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15, hue=0.02),
    ]

    # Stronger domain-oriented augmentations (for Experiment Run 4)
    if strong_domain_aug:
        aug_list.extend([
            T.RandomApply([T.GaussianBlur(kernel_size=(3, 3), sigma=(0.1, 1.5))], p=0.3),
            T.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.25),
        ])

    aug_list.extend([
        T.ToTensor(),
        normalize
    ])

    return T.Compose(aug_list)
