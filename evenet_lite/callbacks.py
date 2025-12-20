from __future__ import annotations

import copy
from typing import Any, Dict, Iterable, Optional

import torch


class Callback:
    """Base class for Evenet-Lite callbacks.

    Subclasses can override any hook to inject custom behavior.
    """

    def on_train_start(self, trainer: "Trainer") -> None:  # pragma: no cover - interface
        pass

    def on_epoch_start(self, trainer: "Trainer", epoch: int) -> None:  # pragma: no cover - interface
        pass

    def on_batch_end(
        self, trainer: "Trainer", epoch: int, batch_idx: int, batch: Dict[str, Any], loss: float
    ) -> None:  # pragma: no cover - interface
        pass

    def on_epoch_end(self, trainer: "Trainer", epoch: int, metrics: Dict[str, float]) -> None:  # pragma: no cover - interface
        pass

    def on_train_end(self, trainer: "Trainer") -> None:  # pragma: no cover - interface
        pass


class EvenetLiteNormalizer:
    """Simple feature-wise normalizer.

    Computes mean and standard deviation for each feature tensor and applies
    normalization during training and evaluation. The normalizer is stateless
    w.r.t. gradients and safe to share across dataloaders.
    """

    def __init__(self) -> None:
        self._stats: Dict[str, Dict[str, torch.Tensor]] = {}

    def fit(self, data: Dict[str, torch.Tensor], feature_names: Dict[str, Iterable[str]]) -> None:
        """Fit the normalizer on the provided training data.

        Args:
            data: Mapping of feature names to tensors. Values should be shaped
                ``(N, ...)`` where the last dimension corresponds to features.
            feature_names: Human-readable feature names per tensor. Only used
                for validation and traceability.
        """

        self._stats = {}
        for key, tensor in data.items():
            if key not in feature_names:
                # Unknown feature container, skip normalization
                continue
            if not torch.is_tensor(tensor):
                tensor = torch.as_tensor(tensor)
            if tensor.numel() == 0:
                continue
            flat = tensor.float().view(tensor.shape[0], -1)
            mean = flat.mean(dim=0)
            std = flat.std(dim=0, unbiased=False).clamp_min(1e-6)
            self._stats[key] = {"mean": mean, "std": std, "shape": flat.shape[1:]}

    def transform(self, data: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Normalize inputs using previously computed statistics."""

        if not self._stats:
            return data

        normalized: Dict[str, torch.Tensor] = {}
        for key, tensor in data.items():
            if key not in self._stats:
                normalized[key] = tensor
                continue
            stats = self._stats[key]
            orig_shape = tensor.shape
            reshaped = tensor.float().view(orig_shape[0], -1)
            normed = (reshaped - stats["mean"]) / stats["std"]
            normalized[key] = normed.view(orig_shape)
        return normalized

    def state_dict(self) -> Dict[str, Any]:
        return copy.deepcopy(self._stats)

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        self._stats = copy.deepcopy(state_dict)


class NormalizationCallback(Callback):
    """Callback that handles fitting and applying a normalizer.

    The normalizer is fitted on the training data at the start of training and
    then attached to all datasets managed by the trainer.
    """

    def __init__(self, normalizer: Optional[EvenetLiteNormalizer] = None) -> None:
        self.normalizer = normalizer or EvenetLiteNormalizer()

    def on_train_start(self, trainer: "Trainer") -> None:
        train_data = trainer.train_dataset.raw_features
        feature_names = trainer.feature_names
        self.normalizer.fit(train_data, feature_names)
        trainer.attach_normalizer(self.normalizer)

    def state_dict(self) -> Dict[str, Any]:
        return {"normalizer": self.normalizer.state_dict()}

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        if "normalizer" in state_dict:
            self.normalizer.load_state_dict(state_dict["normalizer"])
