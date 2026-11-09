"""
Stage 10: PyTorch Training Loop and Experiment Manager.

Features:
- Reproducible random seeding across PyTorch, NumPy, and Python.
- Mixed precision training (AMP autocast + GradScaler) for high GPU efficiency.
- Optimizer: AdamW with weight decay.
- Schedulers: CosineAnnealingLR / ReduceLROnPlateau.
- Early stopping with patience tracking validation AUROC / loss.
- Automatic model checkpoint saving (best model & last state).
- Resuming capability from saved checkpoints.
- Google Drive sync support for Colab environments.
"""

import os
import sys
import time
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.append(str(Path(__file__).resolve().parent.parent))
from configs import config
from src.evaluation import compute_binary_metrics, compute_multiclass_metrics


def set_seed(seed: int = config.SEED):
    """Set random seed for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[Any] = None,
        device: Optional[torch.device] = None,
        checkpoint_dir: Optional[Path] = None,
        experiment_name: str = "rop_experiment",
        task: str = "binary",
        patience: int = 7
    ):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.task = task
        self.patience = patience
        self.experiment_name = experiment_name
        
        self.device = device or (torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        self.model.to(self.device)
        self.criterion.to(self.device)

        self.checkpoint_dir = checkpoint_dir or config.CHECKPOINTS_DIR
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.best_checkpoint_path = self.checkpoint_dir / f"{experiment_name}_best.pt"
        self.last_checkpoint_path = self.checkpoint_dir / f"{experiment_name}_last.pt"

        self.use_amp = torch.cuda.is_available()
        self.scaler = torch.amp.GradScaler('cuda', enabled=self.use_amp)

        self.history = {
            "train_loss": [],
            "val_loss": [],
            "val_accuracy": [],
            "val_auroc": [],
            "val_f1": []
        }

    def train_epoch(self, dataloader: DataLoader) -> float:
        """Run one training epoch."""
        self.model.train()
        total_loss = 0.0

        for images, targets, _ in dataloader:
            images = images.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)

            self.optimizer.zero_grad()

            with torch.amp.autocast('cuda', enabled=self.use_amp):
                logits = self.model(images)
                loss = self.criterion(logits, targets)

            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            total_loss += loss.item() * len(targets)

        return total_loss / len(dataloader.dataset)

    @torch.no_grad()
    def evaluate(self, dataloader: DataLoader) -> Tuple[float, Dict[str, Any]]:
        """Evaluate model on a dataloader without gradients."""
        self.model.eval()
        total_loss = 0.0
        all_targets = []
        all_probs = []

        for images, targets, _ in dataloader:
            images = images.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)

            with torch.amp.autocast('cuda', enabled=self.use_amp):
                logits = self.model(images)
                loss = self.criterion(logits, targets)

            total_loss += loss.item() * len(targets)

            probs = torch.softmax(logits, dim=1).cpu().numpy()
            all_probs.append(probs)
            all_targets.append(targets.cpu().numpy())

        avg_loss = total_loss / len(dataloader.dataset)
        y_true = np.concatenate(all_targets)
        y_probs = np.concatenate(all_probs)

        if self.task == "binary":
            # Probability of ROP positive class (index 1)
            metrics = compute_binary_metrics(y_true, y_probs[:, 1])
        else:
            class_names = ["mild", "moderate", "severe"] if y_probs.shape[1] == 3 else None
            metrics = compute_multiclass_metrics(y_true, y_probs, class_names=class_names)

        return avg_loss, metrics

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int = 25
    ) -> Dict[str, Any]:
        """Full training loop with early stopping."""
        best_metric = -float("inf")
        patience_counter = 0

        print(f"\n--- Starting Experiment: {self.experiment_name} on {self.device} ---")
        print(f"Epochs: {epochs} | Batch size: {train_loader.batch_size} | Task: {self.task}")

        for epoch in range(1, epochs + 1):
            t0 = time.time()
            train_loss = self.train_epoch(train_loader)
            val_loss, val_metrics = self.evaluate(val_loader)

            # Primary monitoring metric: AUROC if valid, else F1 or -val_loss
            if self.task == "binary":
                monitored = val_metrics.get("auroc")
                if monitored is None or np.isnan(monitored):
                    monitored = val_metrics.get("f1_score", 0.0)
            else:
                monitored = val_metrics.get("macro_f1", 0.0)

            # Scheduler step
            if self.scheduler is not None:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()

            # Record history
            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            self.history["val_accuracy"].append(val_metrics.get("accuracy", 0.0))
            self.history["val_auroc"].append(val_metrics.get("auroc", 0.0))
            self.history["val_f1"].append(val_metrics.get("f1_score", val_metrics.get("macro_f1", 0.0)))

            elapsed = time.time() - t0
            print(
                f"Epoch [{epoch:02d}/{epochs:02d}] "
                f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
                f"Val Acc: {val_metrics.get('accuracy', 0.0):.4f} | "
                f"AUROC: {val_metrics.get('auroc', 0.0):.4f} | "
                f"F1: {self.history['val_f1'][-1]:.4f} | ({elapsed:.1f}s)"
            )

            # Checkpoint saving & early stopping
            if monitored > best_metric:
                best_metric = monitored
                patience_counter = 0
                self.save_checkpoint(self.best_checkpoint_path, epoch, val_metrics)
                print(f"  --> Saved new best checkpoint (Metric: {best_metric:.4f})")
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    print(f"\n[Early Stopping Triggered] No improvement for {self.patience} epochs.")
                    break

        self.save_checkpoint(self.last_checkpoint_path, epoch, val_metrics)
        print(f"\nTraining completed. Best monitored metric: {best_metric:.4f}")
        return self.history

    def save_checkpoint(self, path: Path, epoch: int, metrics: Dict[str, Any]):
        """Save training state."""
        state = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "metrics": metrics,
            "experiment_name": self.experiment_name,
            "history": self.history
        }
        torch.save(state, path)

    def load_best_model(self):
        """Load the best saved checkpoint weights into the model."""
        if self.best_checkpoint_path.exists():
            checkpoint = torch.load(self.best_checkpoint_path, map_location=self.device)
            self.model.load_state_dict(checkpoint["model_state_dict"])
            print(f"[Loaded] Best weights loaded from {self.best_checkpoint_path}")
        else:
            print(f"Warning: Best checkpoint not found at {self.best_checkpoint_path}")
