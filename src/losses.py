"""
Stage 8: Imbalance-Aware Loss Functions for Rare & Severe ROP Detection.

Implements:
1. Standard Cross-Entropy Loss.
2. Class-Weighted Cross-Entropy Loss (dynamically computed or user-specified).
3. Multi-Class / Binary Focal Loss (Lin et al.).
4. Loss factory helper: get_loss_function(name, class_counts, device).
"""

from typing import Optional, List, Union
import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Focal Loss for addressing extreme class imbalance in severe ROP categories.
    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    """
    def __init__(
        self,
        alpha: Optional[Union[float, List[float], torch.Tensor]] = None,
        gamma: float = 2.0,
        reduction: str = "mean"
    ):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.reduction = reduction

        if alpha is not None:
            if isinstance(alpha, (list, tuple)):
                self.alpha = torch.tensor(alpha, dtype=torch.float32)
            elif isinstance(alpha, (float, int)):
                self.alpha = torch.tensor([1.0 - alpha, alpha], dtype=torch.float32)
            elif isinstance(alpha, torch.Tensor):
                self.alpha = alpha.float()
            else:
                self.alpha = None
        else:
            self.alpha = None

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        inputs: (batch_size, num_classes) raw logits
        targets: (batch_size,) integer class labels
        """
        ce_loss = F.cross_entropy(inputs, targets, reduction="none")
        p = torch.exp(-ce_loss)
        focal_loss = ((1.0 - p) ** self.gamma) * ce_loss

        if self.alpha is not None:
            if self.alpha.device != inputs.device:
                self.alpha = self.alpha.to(inputs.device)
            alpha_t = self.alpha[targets]
            focal_loss = alpha_t * focal_loss

        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        return focal_loss


def get_loss_function(
    loss_name: str = "cross_entropy",
    class_counts: Optional[dict] = None,
    gamma: float = 2.0,
    device: Optional[torch.device] = None
) -> nn.Module:
    """
    Loss factory helper.
    Options: 'cross_entropy', 'weighted_ce', 'focal'
    """
    loss_name = loss_name.lower().replace("-", "_")

    weights = None
    if class_counts is not None and len(class_counts) > 0:
        total = sum(class_counts.values())
        num_classes = len(class_counts)
        # Inverse class frequency weighting
        raw_weights = [total / (num_classes * class_counts[i]) for i in sorted(class_counts.keys())]
        weights = torch.tensor(raw_weights, dtype=torch.float32)
        if device is not None:
            weights = weights.to(device)

    if loss_name == "cross_entropy":
        return nn.CrossEntropyLoss()

    elif loss_name == "weighted_ce":
        if weights is None:
            raise ValueError("class_counts required to compute class-weighted cross-entropy")
        return nn.CrossEntropyLoss(weight=weights)

    elif loss_name == "focal":
        return FocalLoss(alpha=weights, gamma=gamma, reduction="mean")

    else:
        raise ValueError(f"Unknown loss: {loss_name}. Supported: 'cross_entropy', 'weighted_ce', 'focal'")
