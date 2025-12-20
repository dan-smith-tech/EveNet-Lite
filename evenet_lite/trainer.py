from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple
import os

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

from .callbacks import Callback, NormalizationCallback
from .data import EvenetTensorDataset, build_sampler, DistributedWeightedSampler
from .checkpoint import load_checkpoint, save_checkpoint
from .metrics import compute_accuracy, compute_auc, compute_loss, summarize_metrics


@dataclass
class TrainerConfig:
    device: str = "auto"
    lr: float = 1e-3
    weight_decay: float = 0.01
    grad_clip: Optional[float] = None
    num_workers: int = 2
    scheduler_fn: Optional[Any] = None
    checkpoint_path: Optional[str] = None
    checkpoint_every: int = 1
    resume_from: Optional[str] = None


class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        feature_names: Dict[str, Iterable[str]],
        config: TrainerConfig,
        callbacks: Optional[List[Callback]] = None,
    ) -> None:
        self.model = model
        self.feature_names = feature_names
        self.config = config
        self.callbacks: List[Callback] = callbacks or []
        self._init_distributed()
        self.device = self._resolve_device(config.device)

        self.train_dataset: EvenetTensorDataset
        self.val_dataset: Optional[EvenetTensorDataset] = None
        self.test_dataset: Optional[EvenetTensorDataset] = None

        self.optimizer: torch.optim.Optimizer
        self.scheduler: Optional[Any] = None

    def _init_distributed(self) -> None:
        if dist.is_available() and not dist.is_initialized():
            world_size = int(os.environ.get("WORLD_SIZE", "1"))
            if world_size > 1:
                backend = "nccl" if torch.cuda.is_available() else "gloo"
                dist.init_process_group(backend=backend, init_method="env://")
        self.local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        self.global_rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0

    @property
    def rank(self) -> int:
        if dist.is_available() and dist.is_initialized():
            return dist.get_rank()
        return 0

    @property
    def world_size(self) -> int:
        if dist.is_available() and dist.is_initialized():
            return dist.get_world_size()
        return 1

    def is_rank_zero(self) -> bool:
        return self.rank == 0

    def _resolve_device(self, device: str) -> torch.device:
        if device == "auto":
            if torch.cuda.is_available():
                if dist.is_available() and dist.is_initialized():
                    torch.cuda.set_device(self.local_rank)
                    return torch.device(f"cuda:{self.local_rank}")
                return torch.device("cuda")
            return torch.device("cpu")
        return torch.device(device)

    def attach_normalizer(self, normalizer: Any) -> None:
        if hasattr(self, "train_dataset") and self.train_dataset is not None:
            self.train_dataset.set_normalizer(normalizer)
        if getattr(self, "val_dataset", None):
            self.val_dataset.set_normalizer(normalizer)
        if getattr(self, "test_dataset", None):
            self.test_dataset.set_normalizer(normalizer)

    def attach_normalizer_state(self, state: Dict[str, Any]) -> None:
        for cb in self.callbacks:
            if isinstance(cb, NormalizationCallback) and cb.normalizer is not None:
                cb.normalizer.load_state_dict(state)
                self.attach_normalizer(cb.normalizer)
                break

    def _current_normalizer_state(self) -> Optional[Dict[str, Any]]:
        for cb in self.callbacks:
            if isinstance(cb, NormalizationCallback) and cb.normalizer is not None:
                return cb.normalizer.state_dict()
        return None

    def setup_datasets(
        self,
        train_data: Tuple[Dict[str, torch.Tensor], torch.Tensor, Optional[torch.Tensor]],
        val_data: Optional[Tuple[Dict[str, torch.Tensor], torch.Tensor, Optional[torch.Tensor]]],
        test_data: Optional[Tuple[Dict[str, torch.Tensor], torch.Tensor, Optional[torch.Tensor]]],
    ) -> None:
        X_train, y_train, w_train = train_data
        self.train_dataset = EvenetTensorDataset(X_train, y_train, w_train)
        self.val_dataset = EvenetTensorDataset(*val_data) if val_data is not None else None
        self.test_dataset = EvenetTensorDataset(*test_data) if test_data is not None else None

    def _maybe_wrap_ddp(self) -> torch.nn.Module:
        if self.world_size > 1:
            if self.device.type == "cuda":
                return DDP(self.model.to(self.device), device_ids=[self.device])
            return DDP(self.model.to(self.device))
        return self.model.to(self.device)

    def train(
        self,
        train_data: Tuple[Dict[str, torch.Tensor], torch.Tensor, Optional[torch.Tensor]],
        val_data: Optional[Tuple[Dict[str, torch.Tensor], torch.Tensor, Optional[torch.Tensor]]],
        test_data: Optional[Tuple[Dict[str, torch.Tensor], torch.Tensor, Optional[torch.Tensor]]],
        epochs: int,
        batch_size: int,
        sampler: Optional[str],
        epoch_size: Optional[int] = None,
    ) -> None:
        self.setup_datasets(train_data, val_data, test_data)
        # Insert normalization callback by default
        if not any(isinstance(cb, NormalizationCallback) for cb in self.callbacks):
            self.callbacks.insert(0, NormalizationCallback())

        wrapped_model = self._maybe_wrap_ddp()
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.config.lr, weight_decay=self.config.weight_decay)
        if self.config.scheduler_fn:
            self.scheduler = self.config.scheduler_fn(self.optimizer)

        if self.config.resume_from:
            self.restore_checkpoint(self.config.resume_from)

        for cb in self.callbacks:
            cb.on_train_start(self)

        sampler_obj = build_sampler(sampler, self.train_dataset, self.train_dataset.sample_weights, epoch_size)
        if sampler_obj is None and self.world_size > 1:
            sampler_obj = DistributedSampler(self.train_dataset)

        train_loader = DataLoader(
            self.train_dataset,
            batch_size=batch_size,
            sampler=sampler_obj,
            shuffle=sampler_obj is None,
            num_workers=self.config.num_workers,
            pin_memory=True,
            drop_last=False,
        )

        val_loader = None
        if self.val_dataset is not None:
            val_sampler = DistributedSampler(self.val_dataset, shuffle=False) if self.world_size > 1 else None
            val_loader = DataLoader(
                self.val_dataset,
                batch_size=batch_size,
                sampler=val_sampler,
                shuffle=False,
                num_workers=self.config.num_workers,
            )

        for epoch in range(epochs):
            if isinstance(sampler_obj, (DistributedSampler, DistributedWeightedSampler)):
                sampler_obj.set_epoch(epoch)
            if isinstance(val_loader, DataLoader) and isinstance(val_loader.sampler, DistributedSampler):
                val_loader.sampler.set_epoch(epoch)

            for cb in self.callbacks:
                cb.on_epoch_start(self, epoch)

            train_metrics = self._run_epoch(wrapped_model, train_loader, epoch, training=True)
            if val_loader is not None:
                val_metrics = self._run_epoch(wrapped_model, val_loader, epoch, training=False)
            else:
                val_metrics = {}

            if self.scheduler:
                self.scheduler.step()

            if self.is_rank_zero():
                merged = {f"train_{k}": v for k, v in train_metrics.items()}
                merged.update({f"val_{k}": v for k, v in val_metrics.items()})
            else:
                merged = {}

            if self.config.checkpoint_path and self.is_rank_zero():
                if (epoch + 1) % max(1, self.config.checkpoint_every) == 0:
                    extra = {"epoch": epoch}
                    if self.scheduler:
                        extra["scheduler"] = self.scheduler.state_dict()
                    self.save_checkpoint(self.config.checkpoint_path, extra)

            for cb in self.callbacks:
                cb.on_epoch_end(self, epoch, merged)

        for cb in self.callbacks:
            cb.on_train_end(self)

    def save_checkpoint(self, path: str, extra: Optional[Dict[str, Any]] = None) -> None:
        normalizer_state = self._current_normalizer_state()
        save_checkpoint(
            path,
            model_state=self.model.state_dict(),
            optimizer_state=self.optimizer.state_dict(),
            normalizer_state=normalizer_state,
            extra=extra,
        )

    def restore_checkpoint(self, path: str, map_location: Optional[str] = None) -> None:
        checkpoint = load_checkpoint(path, map_location=map_location)
        self.model.load_state_dict(checkpoint["model"])
        if "optimizer" in checkpoint and hasattr(self, "optimizer"):
            self.optimizer.load_state_dict(checkpoint["optimizer"])
        if "normalizer" in checkpoint and checkpoint["normalizer"]:
            self.attach_normalizer_state(checkpoint["normalizer"])
        if self.config.scheduler_fn and "extra" in checkpoint:
            scheduler_state = checkpoint["extra"].get("scheduler") if checkpoint.get("extra") else None
            if scheduler_state and self.scheduler:
                self.scheduler.load_state_dict(scheduler_state)

    def _forward(self, model: torch.nn.Module, features: Dict[str, torch.Tensor]) -> torch.Tensor:
        try:
            return model(**features)
        except TypeError:
            return model(features)

    def _run_epoch(
        self,
        model: torch.nn.Module,
        loader: DataLoader,
        epoch: int,
        training: bool,
    ) -> Dict[str, float]:
        if training:
            model.train()
        else:
            model.eval()

        metric_sum: Dict[str, float] = {"loss": 0.0, "accuracy": 0.0}
        metric_count: Dict[str, int] = {"loss": 0, "accuracy": 0}
        auc_values: List[float] = []

        for batch_idx, (features, targets, weights) in enumerate(loader):
            features = {k: v.to(self.device) for k, v in features.items()}
            targets = targets.to(self.device)
            weight_tensor = weights.to(self.device) if weights is not None else None

            with torch.set_grad_enabled(training):
                logits = self._forward(model, features)
                loss = compute_loss(logits, targets, weight_tensor)
                if training:
                    self.optimizer.zero_grad()
                    loss.backward()
                    if self.config.grad_clip:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), self.config.grad_clip)
                    self.optimizer.step()

            metric_sum["loss"] += loss.item() * targets.size(0)
            metric_count["loss"] += targets.size(0)

            acc = compute_accuracy(logits, targets)
            metric_sum["accuracy"] += acc * targets.size(0)
            metric_count["accuracy"] += targets.size(0)

            auc_val = compute_auc(logits, targets)
            if auc_val is not None:
                auc_values.append(auc_val)

            for cb in self.callbacks:
                cb.on_batch_end(self, epoch, batch_idx, {"features": features, "targets": targets}, loss.item())

        metrics = summarize_metrics(metric_sum, metric_count)
        if auc_values:
            metrics["auc"] = sum(auc_values) / len(auc_values)
        return metrics

    def predict(self, dataset: EvenetTensorDataset, batch_size: int = 256) -> torch.Tensor:
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=self.config.num_workers)
        self.model.to(self.device)
        self.model.eval()
        outputs: List[torch.Tensor] = []
        with torch.no_grad():
            for features, _, _ in loader:
                features = {k: v.to(self.device) for k, v in features.items()}
                logits = self._forward(self.model, features)
                outputs.append(torch.softmax(logits, dim=1).cpu())
        return torch.cat(outputs, dim=0)

    def evaluate(
        self,
        dataset: EvenetTensorDataset,
        batch_size: int = 256,
    ) -> Dict[str, float]:
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=self.config.num_workers)
        self.model.to(self.device)
        self.model.eval()
        metric_sum: Dict[str, float] = {"loss": 0.0, "accuracy": 0.0}
        metric_count: Dict[str, int] = {"loss": 0, "accuracy": 0}
        auc_values: List[float] = []
        with torch.no_grad():
            for features, targets, weights in loader:
                features = {k: v.to(self.device) for k, v in features.items()}
                targets = targets.to(self.device)
                weight_tensor = weights.to(self.device) if weights is not None else None

                logits = self._forward(self.model, features)
                loss = compute_loss(logits, targets, weight_tensor)
                metric_sum["loss"] += loss.item() * targets.size(0)
                metric_count["loss"] += targets.size(0)

                acc = compute_accuracy(logits, targets)
                metric_sum["accuracy"] += acc * targets.size(0)
                metric_count["accuracy"] += targets.size(0)

                auc_val = compute_auc(logits, targets)
                if auc_val is not None:
                    auc_values.append(auc_val)
        metrics = summarize_metrics(metric_sum, metric_count)
        if auc_values:
            metrics["auc"] = sum(auc_values) / len(auc_values)
        return metrics
