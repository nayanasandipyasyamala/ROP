"""
Stage 4: PyTorch Dataset and DataLoader Module.

Implements:
- ROPDataset: PyTorch Dataset for retinal fundus images with metadata.
- Support for Task A (Binary screening: Normal=0, ROP=1).
- Support for Task B (Multiclass severity: mild=0, moderate=1, severe=2, and binary non-severe vs severe).
- Leak-free DataLoader construction for:
  * Experiment A1: Pooled 5-fold cross-validation.
  * Experiment A2: ROP-VL -> Ostrava (cross-dataset domain shift).
  * Experiment A3: Ostrava -> ROP-VL (cross-dataset domain shift).
  * Experiment A4: Leave-one-camera-out on Ostrava.
  * Experiment A5: External evaluation on HVDROPDB.
- Imbalance handling: WeightedRandomSampler for class-balanced sampling.
"""

import sys
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, List

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from PIL import Image

sys.path.append(str(Path(__file__).resolve().parent.parent))
from configs import config
from src.preprocessing import get_transforms

SEVERITY_MAP = {"mild": 0, "moderate": 1, "severe": 2}
SEVERITY_BINARY_MAP = {"mild": 0, "moderate": 0, "severe": 1}  # non-severe (0) vs severe (1)


class ROPDataset(Dataset):
    """
    Standard PyTorch Dataset for ROP images.
    """
    def __init__(
        self,
        df: pd.DataFrame,
        transform=None,
        task: str = "binary"  # 'binary', 'severity_3class', 'severity_binary'
    ):
        self.df = df.reset_index(drop=True)
        self.transform = transform
        self.task = task

        # Prepare labels
        self.labels = []
        for _, r in self.df.iterrows():
            if self.task == "binary":
                self.labels.append(int(r["binary_label"]))
            elif self.task == "severity_3class":
                self.labels.append(SEVERITY_MAP[str(r["severity_label"])])
            elif self.task == "severity_binary":
                self.labels.append(SEVERITY_BINARY_MAP[str(r["severity_label"])])
            else:
                raise ValueError(f"Unknown task: {self.task}")

        self.labels = np.array(self.labels, dtype=np.int64)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, Dict[str, Any]]:
        row = self.df.iloc[idx]
        img_path = row["image_path"]

        # Read fundus image
        im = Image.open(img_path).convert("RGB")

        if self.transform is not None:
            im = self.transform(im)

        label = self.labels[idx]

        meta = {
            "image_path": img_path,
            "patient_uid": row.get("unique_patient_id", "Unknown"),
            "dataset_source": row.get("dataset_source", "Unknown"),
            "camera_device": row.get("camera_device", "Unknown"),
            "original_label": row.get("original_label", "Unknown")
        }

        return im, label, meta

    def get_class_counts(self) -> Dict[int, int]:
        """Return counts per class."""
        unique, counts = np.unique(self.labels, return_counts=True)
        return dict(zip(unique.tolist(), counts.tolist()))

    def get_sample_weights(self) -> torch.DoubleTensor:
        """Calculate per-sample weights inversely proportional to class frequencies."""
        class_counts = self.get_class_counts()
        total_samples = len(self.labels)
        class_weights = {cls: total_samples / (len(class_counts) * count) for cls, count in class_counts.items()}
        weights = [class_weights[y] for y in self.labels]
        return torch.DoubleTensor(weights)


def build_dataloaders(
    task_df: pd.DataFrame,
    val_fold: int,
    batch_size: int = config.BATCH_SIZE,
    num_workers: int = config.NUM_WORKERS,
    task: str = "binary",
    use_balanced_sampler: bool = False,
    strong_domain_aug: bool = False
) -> Tuple[DataLoader, DataLoader]:
    """
    Build leak-free train and validation DataLoaders for a specific cross-validation fold.
    Uses patient-level partition column ('fold_taskA' or 'fold_taskB').
    """
    fold_col = "fold_taskA" if task == "binary" else "fold_taskB"

    # Core training images: all folds except val_fold (excluding test set fold = -1)
    train_mask = (task_df[fold_col] != val_fold) & (task_df[fold_col] >= 0)
    val_mask = task_df[fold_col] == val_fold

    train_df = task_df[train_mask].reset_index(drop=True)
    val_df = task_df[val_mask].reset_index(drop=True)

    # Leakage check assertion
    train_pts = set(train_df["unique_patient_id"])
    val_pts = set(val_df["unique_patient_id"])
    assert len(train_pts & val_pts) == 0, f"DATA LEAKAGE: {len(train_pts & val_pts)} patients overlap in fold {val_fold}!"

    # Datasets
    train_tf = get_transforms(img_size=config.IMG_SIZE, is_training=True, strong_domain_aug=strong_domain_aug)
    val_tf = get_transforms(img_size=config.IMG_SIZE, is_training=False)

    train_dataset = ROPDataset(train_df, transform=train_tf, task=task)
    val_dataset = ROPDataset(val_df, transform=val_tf, task=task)

    # Sampler
    sampler = None
    shuffle = True
    if use_balanced_sampler:
        weights = train_dataset.get_sample_weights()
        sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
        shuffle = False

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )

    return train_loader, val_loader


def build_cross_dataset_dataloaders(
    task_df: pd.DataFrame,
    train_source: str = "ROPVL",
    test_source: str = "OSTRAVA",
    batch_size: int = config.BATCH_SIZE,
    num_workers: int = config.NUM_WORKERS,
    task: str = "binary",
    use_balanced_sampler: bool = False,
    strong_domain_aug: bool = False
) -> Tuple[DataLoader, DataLoader]:
    """
    Build cross-dataset train and test DataLoaders for domain-shift experiments (A2 and A3).
    """
    train_df = task_df[task_df["dataset_source"] == train_source].reset_index(drop=True)
    test_df = task_df[task_df["dataset_source"] == test_source].reset_index(drop=True)

    train_tf = get_transforms(img_size=config.IMG_SIZE, is_training=True, strong_domain_aug=strong_domain_aug)
    test_tf = get_transforms(img_size=config.IMG_SIZE, is_training=False)

    train_dataset = ROPDataset(train_df, transform=train_tf, task=task)
    test_dataset = ROPDataset(test_df, transform=test_tf, task=task)

    sampler = None
    shuffle = True
    if use_balanced_sampler:
        weights = train_dataset.get_sample_weights()
        sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
        shuffle = False

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, sampler=sampler, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, test_loader


def build_leave_one_camera_out_dataloaders(
    task_df: pd.DataFrame,
    test_camera: str,
    batch_size: int = config.BATCH_SIZE,
    num_workers: int = config.NUM_WORKERS,
    use_balanced_sampler: bool = False
) -> Tuple[DataLoader, DataLoader]:
    """
    Leave-one-camera-out cross-validation on Ostrava (Experiment A4).
    """
    ostrava_df = task_df[task_df["dataset_source"] == "OSTRAVA"].copy()
    train_df = ostrava_df[ostrava_df["camera_device"] != test_camera].reset_index(drop=True)
    test_df = ostrava_df[ostrava_df["camera_device"] == test_camera].reset_index(drop=True)

    train_tf = get_transforms(img_size=config.IMG_SIZE, is_training=True)
    test_tf = get_transforms(img_size=config.IMG_SIZE, is_training=False)

    train_dataset = ROPDataset(train_df, transform=train_tf, task="binary")
    test_dataset = ROPDataset(test_df, transform=test_tf, task="binary")

    sampler = None
    shuffle = True
    if use_balanced_sampler:
        weights = train_dataset.get_sample_weights()
        sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
        shuffle = False

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, sampler=sampler, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, test_loader


def build_external_test_dataloader(
    task_df: pd.DataFrame,
    external_source: str = "HVDROPDB",
    batch_size: int = config.BATCH_SIZE,
    num_workers: int = config.NUM_WORKERS
) -> DataLoader:
    """
    Build DataLoader for external test benchmark (Experiment A5: HVDROPDB).
    """
    ext_df = task_df[task_df["dataset_source"] == external_source].reset_index(drop=True)
    ext_tf = get_transforms(img_size=config.IMG_SIZE, is_training=False)
    ext_dataset = ROPDataset(ext_df, transform=ext_tf, task="binary")
    return DataLoader(ext_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
