from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple
import logging
import os

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

from .callbacks import Callback, NormalizationCallback
from .data import EvenetTensorDataset, build_sampler, DistributedWeightedSampler
from .checkpoint import load_checkpoint, save_checkpoint
from .metrics import calculate_physics_metrics, compute_accuracy, compute_loss, summarize_metrics


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
    use_wandb: bool = False
    wandb: Optional[Dict[str, Any]] = None
    compute_physics_metrics: bool = True
    physics_bins: int = 1000


class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        feature_names: Dict[str, Iterable[str]],
        config: TrainerConfig,
        callbacks: Optional[List[Callback]] = None,
        class_labels: Optional[List[str]] = None,
    ) -> None:
        self.model = model
        self.feature_names = feature_names
        self.config = config
        self.callbacks: List[Callback] = callbacks or []
        self._init_distributed()
        self.device = self._resolve_device(config.device)

        self.global_step = 0

        self.class_labels = class_labels
        self.num_classes = len(class_labels) if class_labels is not None else self._infer_num_classes(model)
        self.train_accuracy = None
        self.val_accuracy = None
        self._init_metrics()
        self.wandb_run = None
        self._maybe_init_wandb()
        self._warned_non_binary_physics = False

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

    def _infer_num_classes(self, model: torch.nn.Module) -> Optional[int]:
        if hasattr(model, "num_classes"):
            classes = getattr(model, "num_classes")
            if isinstance(classes, dict):
                return next(iter(classes.values()), None)
            if isinstance(classes, int):
                return classes
        return None

    def _init_metrics(self) -> None:
        if self.num_classes is None:
            logging.info("Skipping torchmetrics initialization because num_classes could not be inferred.")
            return
        try:
            from torchmetrics import Accuracy  # type: ignore

            self.train_accuracy = Accuracy(
                task="multiclass",
                num_classes=self.num_classes,
                compute_on_cpu=True,
                sync_on_compute=True,
            ).to(self.device)
            self.val_accuracy = Accuracy(
                task="multiclass",
                num_classes=self.num_classes,
                compute_on_cpu=True,
                sync_on_compute=True,
            ).to(self.device)
            logging.info("Initialized torchmetrics Accuracy with num_classes=%s", self.num_classes)
        except Exception as exc:  # pragma: no cover - optional dependency
            logging.warning("torchmetrics is unavailable; falling back to manual accuracy. Error: %s", exc)

    def _maybe_init_wandb(self) -> None:
        if not self.config.use_wandb:
            return
        try:
            import wandb
        except Exception as exc:  # pragma: no cover - optional dependency
            logging.warning("Weights & Biases requested but import failed: %s", exc)
            return

        if not self.is_rank_zero():
            logging.info("Skipping Weights & Biases init on non-zero rank %s", self.rank)
            return

        wandb_settings = self.config.wandb or {}
        self.wandb_run = wandb.init(
            project=wandb_settings.get("project"),
            name=wandb_settings.get("name"),
            config=wandb_settings.get("config", {}),
            entity=wandb_settings.get("entity"),
            mode=wandb_settings.get("mode"),
            group=wandb_settings.get("group"),
            job_type=wandb_settings.get("job_type"),
            tags=wandb_settings.get("tags"),
            notes=wandb_settings.get("notes"),
            reinit=True,
        )
        logging.info("Initialized Weights & Biases run: %s", self.wandb_run.name if self.wandb_run else "<none>")

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

    def _all_gather_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        if self.world_size <= 1:
            return tensor
        tensor_list = [torch.zeros_like(tensor) for _ in range(self.world_size)]
        dist.all_gather(tensor_list, tensor)
        return torch.cat(tensor_list, dim=0)

    def _all_reduce_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        if self.world_size > 1:
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        return tensor

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

    def _describe_sampler(self, sampler_obj: Optional[Any]) -> str:
        if sampler_obj is None:
            return "None (shuffled)"
        details: List[str] = [sampler_obj.__class__.__name__]
        if isinstance(sampler_obj, DistributedWeightedSampler):
            details.append(f"epoch_size={sampler_obj.epoch_size}")
            details.append(f"replacement={sampler_obj.replacement}")
        if isinstance(sampler_obj, DistributedSampler):
            details.append(f"num_replicas={sampler_obj.num_replicas}")
            details.append(f"rank={sampler_obj.rank}")
        return ", ".join(details)

    def _class_stats(self, dataset: EvenetTensorDataset) -> Tuple[int, torch.Tensor, torch.Tensor]:
        labels = dataset.labels.long()
        inferred_classes = self.num_classes if self.num_classes is not None else int(labels.max().item() + 1)
        num_classes = max(inferred_classes, int(labels.max().item() + 1))
        counts = torch.bincount(labels, minlength=num_classes)

        if dataset.sample_weights is None:
            weights = torch.ones_like(labels, dtype=torch.float32)
        else:
            weights = torch.as_tensor(dataset.sample_weights, dtype=torch.float32)
            finite_mask = torch.isfinite(weights)
            weights = torch.where(finite_mask, weights, torch.zeros_like(weights))

        weight_sums = torch.zeros(num_classes, dtype=torch.float32)
        for cls_idx in range(num_classes):
            mask = labels == cls_idx
            if mask.any():
                weight_sums[cls_idx] = weights[mask].sum()

        return num_classes, counts, weight_sums

    def _log_class_distribution(self, name: str, dataset: EvenetTensorDataset) -> None:
        if not self.is_rank_zero():
            return
        num_classes, counts, weight_sums = self._class_stats(dataset)
        total = counts.sum().item() or 1.0
        total_weight = weight_sums.sum().item() or 1.0
        class_names = self.class_labels or [str(i) for i in range(num_classes)]
        parts = []
        for idx in range(num_classes):
            label = class_names[idx] if idx < len(class_names) else str(idx)
            frac = counts[idx].item() / total
            w_frac = weight_sums[idx].item() / total_weight
            parts.append(
                f"{label}: count={counts[idx].item()} (frac={frac:.3f}), weight_sum={weight_sums[idx].item():.3f} (frac={w_frac:.3f})"
            )
        logging.info("Class distribution for %s -> %s", name, " | ".join(parts))

    def _log_training_overview(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader],
        train_sampler: Optional[Any],
        val_sampler: Optional[Any],
        epochs: int,
    ) -> None:
        if not self.is_rank_zero():
            return
        logging.info(
            "Training setup: epochs=%d, batch_size=%d, world_size=%d, device=%s",
            epochs,
            train_loader.batch_size,
            self.world_size,
            self.device,
        )
        logging.info(
            "Train loader: size=%d, steps_per_epoch=%d, sampler=%s",
            len(train_loader.dataset),
            len(train_loader),
            self._describe_sampler(train_sampler),
        )
        self._log_class_distribution("train", train_loader.dataset)
        if val_loader is not None:
            logging.info(
                "Val loader: size=%d, steps_per_epoch=%d, sampler=%s",
                len(val_loader.dataset),
                len(val_loader),
                self._describe_sampler(val_sampler),
            )
            self._log_class_distribution("val", val_loader.dataset)
        else:
            logging.info("Validation loader: None")

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
        self.global_step = 0
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

        self._log_training_overview(train_loader, val_loader, sampler_obj, val_loader.sampler if val_loader else None, epochs)

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
                self._log_epoch_stdout(epoch, epochs, merged)
                if self.wandb_run is not None:
                    self.wandb_run.log({"epoch": epoch + 1, **merged})
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

        if self.wandb_run is not None and self.is_rank_zero():
            self.wandb_run.finish()

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

    def _prepare_features(self, features: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        prepared: Dict[str, torch.Tensor] = {}
        for name, tensor in features.items():
            tensor = tensor.to(self.device)
            if name in {"x", "globals"}:
                tensor = tensor.float()
            prepared[name] = tensor
        return prepared

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
        epoch_probs: List[torch.Tensor] = []
        epoch_targets: List[torch.Tensor] = []
        epoch_weights: List[torch.Tensor] = []

        metric_tracker = self.train_accuracy if training else self.val_accuracy
        if metric_tracker is not None:
            metric_tracker.reset()

        progress = None
        if self.is_rank_zero():
            try:
                from tqdm.auto import tqdm  # type: ignore

                progress = tqdm(
                    total=len(loader),
                    desc=f"{'Train' if training else 'Val'} Epoch {epoch + 1}",
                    leave=False,
                )
            except Exception as exc:  # pragma: no cover - optional dependency
                logging.debug("Progress bar unavailable: %s", exc)

        for batch_idx, (features, targets, weights) in enumerate(loader):
            features = self._prepare_features(features)
            targets = targets.long().to(self.device)
            weight_tensor: Optional[torch.Tensor] = None
            if weights is not None:
                weights = weights.to(self.device)
                weight_tensor = None if torch.any(torch.isinf(weights)) else weights

            with torch.set_grad_enabled(training):
                outputs = self._forward(model, features)
                loss = compute_loss(outputs, targets, weight_tensor)
                if training:
                    self.optimizer.zero_grad()
                    loss.backward()
                    if self.config.grad_clip:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), self.config.grad_clip)
                    self.optimizer.step()

            metric_sum["loss"] += loss.item() * targets.size(0)
            metric_count["loss"] += targets.size(0)

            preds = torch.argmax(outputs, dim=1)
            batch_accuracy = compute_accuracy(outputs, targets)
            if metric_tracker is not None:
                metric_tracker.update(preds, targets)
            else:
                metric_sum["accuracy"] += batch_accuracy * targets.size(0)
                metric_count["accuracy"] += targets.size(0)

            if self.config.compute_physics_metrics:
                epoch_probs.append(outputs.detach())
                epoch_targets.append(targets.detach())
                metric_weights = (
                    weight_tensor.detach()
                    if weight_tensor is not None
                    else torch.ones_like(targets, dtype=torch.float32, device=self.device)
                )
                epoch_weights.append(metric_weights)

            for cb in self.callbacks:
                cb.on_batch_end(self, epoch, batch_idx, {"features": features, "targets": targets}, loss.item())

            if training:
                self.global_step += 1
                self._log_train_step(self.global_step, loss.item(), batch_accuracy, epoch)

            if progress is not None:
                progress.set_postfix({"loss": f"{loss.item():.4f}"}, refresh=False)
                progress.update(1)

        metric_sum_tensor = torch.tensor([metric_sum["loss"], metric_sum["accuracy"]], device=self.device)
        metric_count_tensor = torch.tensor([metric_count["loss"], metric_count["accuracy"]], device=self.device)
        metric_sum_tensor = self._all_reduce_tensor(metric_sum_tensor)
        metric_count_tensor = self._all_reduce_tensor(metric_count_tensor)

        metric_sum["loss"], metric_sum["accuracy"] = metric_sum_tensor.tolist()
        metric_count["loss"] = int(metric_count_tensor[0].item())
        metric_count["accuracy"] = int(metric_count_tensor[1].item())

        metrics = summarize_metrics(metric_sum, metric_count)
        if metric_tracker is not None:
            metrics["accuracy"] = float(metric_tracker.compute().item())

        if self.config.compute_physics_metrics and epoch_probs:
            metrics.update(self._compute_epoch_physics_metrics(epoch_probs, epoch_targets, epoch_weights))
        if progress is not None:
            progress.close()
        return metrics

    def _log_train_step(self, step: int, loss: float, accuracy: float, epoch: int) -> None:
        if self.wandb_run is None or not self.is_rank_zero():
            return
        self.wandb_run.log(
            {"train/loss": loss, "train/accuracy": accuracy, "epoch": epoch + 1},
            step=step,
        )

    def _compute_epoch_physics_metrics(
        self,
        probs_list: List[torch.Tensor],
        targets_list: List[torch.Tensor],
        weights_list: List[torch.Tensor],
    ) -> Dict[str, float]:
        if self.num_classes and self.num_classes != 2 and not self._warned_non_binary_physics:
            logging.warning(
                "Default physics metrics assume binary classification with signal label = 1. "
                "For multiclass tasks, please supply a custom Callback (override on_epoch_end) to compute metrics instead."
            )
            self._warned_non_binary_physics = True

        probs = self._all_gather_tensor(torch.cat(probs_list).detach())
        targets = self._all_gather_tensor(torch.cat(targets_list).detach())
        weights = self._all_gather_tensor(torch.cat(weights_list).detach())

        if not self.is_rank_zero():
            return {}

        metrics = calculate_physics_metrics(
            probs=probs.cpu().numpy(),
            targets=targets.cpu().numpy(),
            weights=weights.cpu().numpy(),
            bins=self.config.physics_bins,
        )
        return {"auc": metrics["auc"], "max_sic": metrics["max_sic"], "max_sic_unc": metrics["max_sic_unc"]}

    def _log_epoch_stdout(self, epoch: int, total_epochs: int, metrics: Dict[str, float]) -> None:
        msg_parts = [f"Epoch {epoch + 1}/{total_epochs}"]
        for key in [
            "train_loss",
            "train_accuracy",
            "train_auc",
            "train_max_sic",
            "val_loss",
            "val_accuracy",
            "val_auc",
            "val_max_sic",
        ]:
            if key in metrics:
                msg_parts.append(f"{key}={metrics[key]:.4f}")
        logging.info(" | ".join(msg_parts))

    def predict(self, dataset: EvenetTensorDataset, batch_size: int = 256) -> torch.Tensor:
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=self.config.num_workers)
        self.model.to(self.device)
        self.model.eval()
        outputs: List[torch.Tensor] = []
        with torch.no_grad():
            for features, _, _ in loader:
                features = self._prepare_features(features)
                outputs.append(self._forward(self.model, features).cpu())
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
        probs_accum: List[torch.Tensor] = []
        targets_accum: List[torch.Tensor] = []
        weights_accum: List[torch.Tensor] = []
        with torch.no_grad():
            for features, targets, weights in loader:
                features = self._prepare_features(features)
                targets = targets.long().to(self.device)
                weight_tensor = weights.to(self.device) if weights is not None else None
                if weight_tensor is not None and torch.any(torch.isinf(weight_tensor)):
                    weight_tensor = None

                outputs = self._forward(self.model, features)
                loss = compute_loss(outputs, targets, weight_tensor)
                metric_sum["loss"] += loss.item() * targets.size(0)
                metric_count["loss"] += targets.size(0)

                acc = compute_accuracy(outputs, targets)
                metric_sum["accuracy"] += acc * targets.size(0)
                metric_count["accuracy"] += targets.size(0)

                if self.config.compute_physics_metrics:
                    probs_accum.append(outputs.cpu())
                    targets_accum.append(targets.cpu())
                    weights_accum.append(
                        torch.ones_like(targets, dtype=torch.float32)
                        if weight_tensor is None
                        else weight_tensor.detach().cpu()
                    )
        metrics = summarize_metrics(metric_sum, metric_count)
        if self.config.compute_physics_metrics and probs_accum:
            metrics.update(
                self._compute_epoch_physics_metrics(
                    [p.to(self.device) for p in probs_accum],
                    [t.to(self.device) for t in targets_accum],
                    [w.to(self.device) for w in weights_accum],
                )
            )
        return metrics
