from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Tuple

import torch

from .callbacks import Callback, EvenetLiteNormalizer, NormalizationCallback
from .data import EvenetTensorDataset
from .trainer import Trainer, TrainerConfig


class EvenetLiteClassifier:
    """High-level classifier API for Evenet-Lite."""

    def __init__(
        self,
        model: torch.nn.Module,
        num_classes: int,
        device: str = "auto",
        lr: float = 1e-3,
        weight_decay: float = 0.01,
        grad_clip: Optional[float] = None,
        scheduler_fn: Optional[callable] = None,
    ) -> None:
        self.model = model
        self.num_classes = num_classes
        self.config = TrainerConfig(
            device=device,
            lr=lr,
            weight_decay=weight_decay,
            grad_clip=grad_clip,
            scheduler_fn=scheduler_fn,
        )
        self.trainer: Optional[Trainer] = None
        self.normalizer: Optional[EvenetLiteNormalizer] = None
        self.feature_names: Optional[Dict[str, Iterable[str]]] = None

    def fit(
        self,
        train_data: Tuple[Dict[str, torch.Tensor], torch.Tensor, Optional[torch.Tensor]],
        val_data: Optional[Tuple[Dict[str, torch.Tensor], torch.Tensor, Optional[torch.Tensor]]] = None,
        feature_names: Optional[Dict[str, Iterable[str]]] = None,
        callbacks: Optional[List[Callback]] = None,
        epochs: int = 10,
        batch_size: int = 256,
        sampler: Optional[str] = None,
        epoch_size: Optional[int] = None,
        checkpoint_path: Optional[str] = None,
        resume_from: Optional[str] = None,
        checkpoint_every: int = 1,
    ) -> None:
        if feature_names is None:
            feature_names = {k: [] for k in train_data[0]}
        self.feature_names = feature_names
        callback_list = callbacks or []
        if not any(isinstance(cb, NormalizationCallback) for cb in callback_list):
            callback_list = [NormalizationCallback()] + callback_list

        self.config.checkpoint_path = checkpoint_path
        self.config.resume_from = resume_from
        self.config.checkpoint_every = checkpoint_every

        self.trainer = Trainer(self.model, feature_names, self.config, callback_list)
        self.trainer.train(train_data, val_data, None, epochs, batch_size, sampler, epoch_size)
        norm_callback = next(cb for cb in self.trainer.callbacks if isinstance(cb, NormalizationCallback))
        self.normalizer = norm_callback.normalizer

    def predict(self, X: Dict[str, torch.Tensor], batch_size: int = 256) -> torch.Tensor:
        if self.trainer is None:
            raise RuntimeError("Model must be fitted before predicting")
        dataset = EvenetTensorDataset(X, torch.zeros(len(next(iter(X.values())))), None, self.normalizer)
        return self.trainer.predict(dataset, batch_size)

    def evaluate(
        self,
        X: Dict[str, torch.Tensor],
        y: torch.Tensor,
        weights: Optional[torch.Tensor] = None,
        batch_size: int = 256,
    ) -> Dict[str, float]:
        if self.trainer is None:
            raise RuntimeError("Model must be fitted before evaluation")
        dataset = EvenetTensorDataset(X, y, weights, self.normalizer)
        return self.trainer.evaluate(dataset, batch_size)

    def load_checkpoint(
        self,
        path: str,
        feature_names: Optional[Dict[str, Iterable[str]]] = None,
        map_location: Optional[str] = None,
    ) -> None:
        feature_names = feature_names or self.feature_names
        if feature_names is None:
            raise ValueError("feature_names must be provided when loading a checkpoint before fitting.")
        self.feature_names = feature_names
        if self.trainer is None:
            callbacks: List[Callback] = [NormalizationCallback()]
            self.trainer = Trainer(self.model, feature_names, self.config, callbacks)
        self.trainer.restore_checkpoint(path, map_location=map_location)
        norm_callback = next((cb for cb in self.trainer.callbacks if isinstance(cb, NormalizationCallback)), None)
        if norm_callback is not None:
            self.normalizer = norm_callback.normalizer

    def save_checkpoint(self, path: str) -> None:
        if self.trainer is None:
            raise RuntimeError("Model must be fitted before saving checkpoints")
        self.trainer.save_checkpoint(path)
