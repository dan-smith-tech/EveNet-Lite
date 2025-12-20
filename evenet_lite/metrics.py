from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn.functional as F

try:  # Optional dependency
    from sklearn import metrics as sk_metrics
except Exception:  # pragma: no cover - fallback when sklearn unavailable
    sk_metrics = None


def compute_loss(logits: torch.Tensor, targets: torch.Tensor, weights: Optional[torch.Tensor]) -> torch.Tensor:
    per_sample = F.cross_entropy(logits, targets, reduction="none")
    if weights is not None:
        weights = weights.to(per_sample.device)
        per_sample = per_sample * weights
    return per_sample.mean()


def compute_accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    preds = logits.argmax(dim=1)
    correct = (preds == targets).sum().item()
    return correct / max(1, targets.numel())


def compute_auc(logits: torch.Tensor, targets: torch.Tensor) -> Optional[float]:
    if sk_metrics is None:
        return None
    if logits.shape[1] != 2:
        return None
    probs = torch.softmax(logits, dim=1)[:, 1].detach().cpu().numpy()
    y_true = targets.detach().cpu().numpy()
    try:
        return float(sk_metrics.roc_auc_score(y_true, probs))
    except Exception:
        return None


def summarize_metrics(accumulator: Dict[str, float], counts: Dict[str, int]) -> Dict[str, float]:
    return {name: accumulator[name] / max(1, counts[name]) for name in accumulator}
