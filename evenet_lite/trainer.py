import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

from .callbacks import Callback, NormalizationCallback, DebugCallback
from .data import EvenetTensorDataset, build_sampler, DistributedWeightedSampler
from .checkpoint import load_checkpoint, save_checkpoint
from .metrics import calculate_physics_metrics, compute_accuracy, compute_loss, summarize_metrics
from .optim import (
    build_optimizers_and_schedulers,
    DEFAULT_LR_GROUPS,
    DEFAULT_MODULE_GROUPS,
    DEFAULT_WEIGHT_DECAY,
)


def format_metrics_for_logging(
        metrics: Dict[str, Any],
        *,
        exclude_keys: Optional[Iterable[str]] = None,
        float_fmt: str = ".5f",
) -> str:
    """Format evaluation metrics for clean logging.

    - Excludes selected keys
    - Nicely formats scalars
    - Summarizes non-scalars
    """
    exclude_keys = set(exclude_keys or [])

    lines = []
    for key in sorted(metrics.keys()):
        if key in exclude_keys:
            continue

        val = metrics[key]

        # Scalars
        if isinstance(val, (int, float)):
            lines.append(f"{key:>24s} : {val:{float_fmt}}")

        # 0-dim tensors / numpy scalars
        elif hasattr(val, "item") and callable(val.item):
            try:
                lines.append(f"{key:>24s} : {val.item():{float_fmt}}")
            except Exception:
                lines.append(f"{key:>24s} : <tensor>")

        # Arrays / tensors
        elif hasattr(val, "shape"):
            lines.append(f"{key:>24s} : array{tuple(val.shape)}")

        # Everything else
        else:
            lines.append(f"{key:>24s} : {type(val).__name__}")

    return "\n".join(lines)


@dataclass
class TrainerConfig:
    device: str = "auto"
    lr: List[float] = field(default_factory=lambda: list(DEFAULT_LR_GROUPS))
    weight_decay: List[float] = field(default_factory=lambda: [DEFAULT_WEIGHT_DECAY] * len(DEFAULT_LR_GROUPS))
    module_lists: List[List[str]] = field(default_factory=lambda: [list(group) for group in DEFAULT_MODULE_GROUPS])
    grad_clip: Optional[float] = None
    num_workers: int = 2
    scheduler_fn: Optional[Any] = None
    optimizer_fn: Optional[Any] = None
    warmup_epochs: Optional[int] = 1
    warmup_ratio: float = 0.1
    warmup_start_factor: float = 0.1
    min_lr: float = 0.0
    checkpoint_path: Optional[str] = None
    checkpoint_every: int = 1
    resume_from: Optional[str] = None
    use_wandb: bool = False
    wandb: Optional[Dict[str, Any]] = None
    compute_physics_metrics: bool = True
    physics_bins: int = 1000
    sic_min_bkg_events: int = 100
    eval_batch_size: Optional[int] = None
    eval_output_path: Optional[str] = None
    save_top_k: int = 0
    monitor_metric: str = "val_loss"
    minimize_metric: bool = True
    early_stop_metric: str = "val_loss"
    early_stop_minimize: bool = True
    early_stop_patience: int = 0
    find_unused_parameters: bool = True


class Trainer:
    def __init__(
            self,
            model: torch.nn.Module,
            feature_names: Dict[str, Iterable[str]],
            config: TrainerConfig,
            callbacks: Optional[List[Callback]] = None,
            class_labels: Optional[List[str]] = None,
            debug: bool = False,
    ) -> None:
        self.model = model
        self.feature_names = feature_names
        self.config = config
        self.callbacks: List[Callback] = callbacks or []
        self.debug = debug
        if self.debug and not any(isinstance(cb, DebugCallback) for cb in self.callbacks):
            self.callbacks.append(DebugCallback())
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

        self.optimizer: Optional[torch.optim.Optimizer] = None
        self.optimizers: List[torch.optim.Optimizer] = []
        self.optimizer_tags: List[str] = []
        self.scheduler: Optional[Any] = None
        self.schedulers: List[Any] = []
        self._best_checkpoints: List[Tuple[float, str]] = []
        self.train_sampler: Optional[Any] = None
        self.val_sampler: Optional[Any] = None
        self.train_loader: Optional[DataLoader] = None
        self.val_loader: Optional[DataLoader] = None

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

    def _reduce_mean_scalar(self, value: float) -> float:
        tensor = torch.tensor(value, device=self.device)
        tensor = self._all_reduce_tensor(tensor)
        tensor = tensor / max(1, self.world_size)
        return tensor.item()

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
        if isinstance(self.model, DDP):
            return self.model

        model = self.model.to(self.device)
        if self.world_size > 1:
            if self.device.type == "cuda":
                device_id = self.device.index if self.device.index is not None else self.local_rank
                model = DDP(
                    model,
                    device_ids=[device_id],
                    find_unused_parameters=self.config.find_unused_parameters,
                )
            else:
                model = DDP(model, find_unused_parameters=self.config.find_unused_parameters)
        return model

    def _unwrap_model(self) -> torch.nn.Module:
        if isinstance(self.model, DDP):
            return self.model.module
        return self.model

    def _load_model_state(self, state: Optional[Dict[str, torch.Tensor]]) -> None:
        if self.world_size > 1:
            # payload = [state]
            # dist.broadcast_object_list(payload, src=0)
            # state = payload[0]

            payload = [None]
            if dist.get_rank() == 0:
                payload[0] = state  # must be CPU objects for object broadcast

            dist.broadcast_object_list(payload, src=0)
            state = payload[0]
        if state is None:
            return
        self._unwrap_model().load_state_dict(state)

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

    def _setup_optimizers_and_schedulers(self, epochs: int, steps_per_epoch: int) -> None:
        self.optimizers, self.schedulers, self.optimizer_tags = build_optimizers_and_schedulers(
            self.model,
            self.config,
            epochs,
            world_size=self.world_size,
            steps_per_epoch=steps_per_epoch,
        )
        self.optimizer = self.optimizers[0] if self.optimizers else None
        self.scheduler = self.schedulers[0] if self.schedulers else None

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
        try:
            self.global_step = 0
            self.setup_datasets(train_data, val_data, test_data)
            # Insert normalization callback by default
            if not any(isinstance(cb, NormalizationCallback) for cb in self.callbacks):
                self.callbacks.insert(0, NormalizationCallback())

            self.model = self._maybe_wrap_ddp()
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
            val_sampler = None
            if self.val_dataset is not None:
                val_sampler = DistributedSampler(self.val_dataset, shuffle=False) if self.world_size > 1 else None
                val_loader = DataLoader(
                    self.val_dataset,
                    batch_size=batch_size,
                    sampler=val_sampler,
                    shuffle=False,
                    num_workers=self.config.num_workers,
                )

            self.train_sampler = sampler_obj
            self.val_sampler = val_loader.sampler if val_loader else None
            self.train_loader = train_loader
            self.val_loader = val_loader

            steps_per_epoch = max(1, len(train_loader))
            self._setup_optimizers_and_schedulers(epochs, steps_per_epoch)

            if self.config.resume_from:
                self.restore_checkpoint(self.config.resume_from)

            for cb in self.callbacks:
                cb.on_train_start(self)

            self._log_training_overview(train_loader, val_loader, sampler_obj, val_sampler, epochs)

            best_metric: Optional[float] = None
            best_model_state: Optional[Dict[str, torch.Tensor]] = None
            best_epoch: Optional[int] = None
            epochs_since_improve = 0

            for epoch in range(epochs):
                if isinstance(sampler_obj, (DistributedSampler, DistributedWeightedSampler)):
                    sampler_obj.set_epoch(epoch)
                if isinstance(val_loader, DataLoader) and isinstance(val_loader.sampler, DistributedSampler):
                    val_loader.sampler.set_epoch(epoch)

                for cb in self.callbacks:
                    cb.on_epoch_start(self, epoch)

                train_metrics = self._run_epoch(self.model, train_loader, epoch, training=True)
                if val_loader is not None:
                    val_metrics = self._run_epoch(self.model, val_loader, epoch, training=False)
                else:
                    val_metrics = {}

                if self.is_rank_zero():
                    merged = {f"train_{k}": v for k, v in train_metrics.items()}
                    merged.update({f"val_{k}": v for k, v in val_metrics.items()})
                    self._log_epoch_stdout(epoch, epochs, merged)
                    if self.wandb_run is not None:
                        wandb_payload = {
                            "epoch": epoch + 1,
                            **self._format_wandb_epoch_metrics(train_metrics, val_metrics),
                        }
                        # self.wandb_run.log(wandb_payload)
                        self.wandb_run.log(wandb_payload, step=self.global_step)
                else:
                    merged = {}

                if self.config.checkpoint_path and self.is_rank_zero():
                    self._save_epoch_checkpoint(merged, epoch)

                for cb in self.callbacks:
                    cb.on_epoch_end(self, epoch, merged)

                stop_training = False
                metric_value = merged.get(self.config.early_stop_metric) if self.is_rank_zero() else None
                improved = False
                if metric_value is not None:
                    if best_metric is None:
                        improved = True
                    elif self.config.early_stop_minimize:
                        improved = metric_value < best_metric
                    else:
                        improved = metric_value > best_metric

                if improved:
                    best_metric = metric_value
                    best_epoch = epoch
                    epochs_since_improve = 0
                    if self.is_rank_zero():
                        best_model_state = {
                            k: v.detach().cpu().clone() if torch.is_tensor(v) else v
                            for k, v in self._unwrap_model().state_dict().items()
                        }
                elif (
                        self.config.early_stop_patience > 0
                        and best_metric is not None
                        and metric_value is not None
                ):
                    epochs_since_improve += 1
                    if epochs_since_improve >= self.config.early_stop_patience:
                        stop_training = True

                stop_tensor = torch.tensor(1 if stop_training else 0, device=self.device)
                if self.world_size > 1:
                    dist.broadcast(stop_tensor, src=0)
                if stop_tensor.item() == 1:
                    logging.info(
                        "Early stopping triggered after %d epochs without improvement on %s",
                        self.config.early_stop_patience,
                        self.config.early_stop_metric,
                    )
                    break

            # ALL ranks must participate in the broadcast inside _load_model_state
            self._load_model_state(best_model_state)

            if self.is_rank_zero() and best_model_state is not None:
                logging.info(
                    "Restored best model from epoch %d based on %s=%.4f",
                    (best_epoch or 0) + 1,
                    self.config.early_stop_metric,
                    best_metric if best_metric is not None else float("nan"),
                )

            for cb in self.callbacks:
                cb.on_train_end(self)

            logging.info("Training finished")

            if self.test_dataset is not None:
                eval_batch_size = self.config.eval_batch_size or batch_size
                logging.info("Evaluating on test set")
                eval_metrics = self.evaluate(
                    self.test_dataset,
                    batch_size=eval_batch_size,
                    output_path=self.config.eval_output_path,
                )
                if self.is_rank_zero():
                    logging.info(
                        "Evaluation finished for test split; metrics available under keys: %s",
                        ", ".join(sorted(eval_metrics.keys())),
                    )

                    allowed_keys = {
                        "auc",
                        "max_sic",
                        "max_sic_unc",
                        "accuracy",
                        "loss",
                    }

                    logging.info(
                        "Evaluation metrics (test split):\n%s",
                        format_metrics_for_logging(
                            {k: eval_metrics[k] for k in allowed_keys if k in eval_metrics},
                        ),
                    )
        finally:
            self._finalize_training()

    def save_checkpoint(self, path: str, extra: Optional[Dict[str, Any]] = None) -> None:
        normalizer_state = self._current_normalizer_state()
        extra_payload = dict(extra or {})
        if self.schedulers:
            scheduler_states = [s.state_dict() for s in self.schedulers if hasattr(s, "state_dict")]
            if scheduler_states:
                extra_payload["schedulers"] = scheduler_states if len(scheduler_states) > 1 else scheduler_states[0]
        logging.info(
            "Saving checkpoint to %s (includes normalizer=%s, extra keys=%s)",
            path,
            normalizer_state is not None,
            list(extra_payload.keys()),
        )
        if len(self.optimizers) > 1:
            optimizer_state = [opt.state_dict() for opt in self.optimizers]
        elif self.optimizer is not None:
            optimizer_state = self.optimizer.state_dict()
        else:
            optimizer_state = {}
        save_checkpoint(
            path,
            model_state=self._unwrap_model().state_dict(),
            optimizer_state=optimizer_state,
            normalizer_state=normalizer_state,
            extra=extra_payload,
        )

    def _extract_monitored_metric(self, metrics: Dict[str, float]) -> Optional[float]:
        if not metrics:
            return None
        value = metrics.get(self.config.monitor_metric)
        if value is None:
            return None
        if isinstance(value, float):
            return value
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    def _finalize_training(self) -> None:
        if self.wandb_run is not None and self.is_rank_zero():
            try:
                self.wandb_run.finish()
            except Exception as exc:  # pragma: no cover - best-effort cleanup
                logging.warning("Failed to close Weights & Biases run cleanly: %s", exc)
            finally:
                self.wandb_run = None

        if dist.is_available() and dist.is_initialized():
            try:
                if self.world_size > 1:
                    dist.barrier()
            except Exception as exc:  # pragma: no cover - best-effort cleanup
                logging.warning("Distributed barrier failed during shutdown: %s", exc)
            try:
                dist.destroy_process_group()
            except Exception as exc:  # pragma: no cover - best-effort cleanup
                logging.warning("Failed to destroy process group cleanly: %s", exc)

    def _should_replace_worst(self, metric: float) -> bool:
        if len(self._best_checkpoints) < self.config.save_top_k:
            return True
        worst_metric, _ = self._worst_checkpoint()
        if self.config.minimize_metric:
            return metric < worst_metric
        return metric > worst_metric

    def _worst_checkpoint(self) -> Tuple[float, str]:
        key_fn = (lambda item: item[0]) if self.config.minimize_metric else (lambda item: -item[0])
        return max(self._best_checkpoints, key=key_fn)

    def _ensure_dir_like_base(self, base: Path) -> Path:
        if base.exists() and base.is_file():
            raise ValueError(
                f"Checkpoint path {base} is a file but is treated as a directory; "
                "please provide a filename with an extension or remove the file."
            )
        base.mkdir(parents=True, exist_ok=True)
        return base

    def _checkpoint_filename(self, base: Path, epoch: int, metric: Optional[float]) -> Path:
        suffix = base.suffix or ".pt"
        stem = base.stem if base.suffix else (base.name or "checkpoint")
        base_is_dir = base.is_dir() or base.suffix == ""

        if base_is_dir:
            base = self._ensure_dir_like_base(base)

        if metric is None:
            filename = f"{stem}-epoch{epoch + 1:04d}{suffix}"
            return base / filename if base_is_dir else base.with_name(filename)

        safe_metric = self.config.monitor_metric.replace("/", "_")
        filename = f"{stem}-{safe_metric}-epoch{epoch + 1:04d}-{metric:.4f}{suffix}"
        if base_is_dir:
            return base / filename
        return base.with_name(filename)

    def _save_epoch_checkpoint(self, metrics: Dict[str, float], epoch: int) -> None:
        if self.config.save_top_k > 0:
            self._maybe_save_best_checkpoint(metrics, epoch)
            return

        if (epoch + 1) % max(1, self.config.checkpoint_every) != 0:
            return

        extra = {"epoch": epoch}
        periodic_path = self._checkpoint_filename(Path(self.config.checkpoint_path), epoch, metric=None)
        self.save_checkpoint(str(periodic_path), extra)

    def _maybe_save_best_checkpoint(self, metrics: Dict[str, float], epoch: int) -> bool:
        metric_value = self._extract_monitored_metric(metrics)
        if metric_value is None or not self.config.checkpoint_path:
            return False

        if not self._should_replace_worst(metric_value):
            return False

        checkpoint_base = Path(self.config.checkpoint_path)
        checkpoint_path = self._checkpoint_filename(checkpoint_base, epoch, metric_value)

        extra = {"epoch": epoch, "monitored_metric": metric_value}
        self.save_checkpoint(str(checkpoint_path), extra)
        logging.info(
            "Saved top-k checkpoint at %s for %s=%.4f (k=%d/%d)",
            checkpoint_path,
            self.config.monitor_metric,
            metric_value,
            len(self._best_checkpoints) + 1,
            self.config.save_top_k,
        )

        self._best_checkpoints.append((metric_value, str(checkpoint_path)))
        if len(self._best_checkpoints) > self.config.save_top_k:
            worst_metric, worst_path = self._worst_checkpoint()
            try:
                Path(worst_path).unlink(missing_ok=True)
            except OSError:
                pass
            logging.info(
                "Removed checkpoint %s to maintain top-k=%d (dropped metric %.4f)",
                worst_path,
                self.config.save_top_k,
                worst_metric,
            )
            self._best_checkpoints = [(m, p) for m, p in self._best_checkpoints if (m, p) != (worst_metric, worst_path)]

        return True

    def restore_checkpoint(self, path: str, map_location: Optional[str] = None) -> None:
        logging.info("Restoring checkpoint from %s", path)
        checkpoint = load_checkpoint(path, map_location=map_location)
        self._unwrap_model().load_state_dict(checkpoint["model"])
        if "optimizer" in checkpoint:
            opt_state = checkpoint["optimizer"]
            if isinstance(opt_state, list) and self.optimizers:
                for optimizer, state in zip(self.optimizers, opt_state):
                    optimizer.load_state_dict(state)
            elif self.optimizer is not None and isinstance(opt_state, dict):
                self.optimizer.load_state_dict(opt_state)
        if "normalizer" in checkpoint and checkpoint["normalizer"]:
            self.attach_normalizer_state(checkpoint["normalizer"])
            logging.info("Restored normalizer state from checkpoint")
        if "extra" in checkpoint:
            scheduler_state = checkpoint["extra"].get("schedulers") if checkpoint.get("extra") else None
            if scheduler_state:
                states = scheduler_state if isinstance(scheduler_state, list) else [scheduler_state]
                for scheduler, state in zip(self.schedulers, states):
                    if scheduler:
                        scheduler.load_state_dict(state)

    def _forward(self, model: torch.nn.Module, features: Dict[str, torch.Tensor]) -> torch.Tensor:
        try:
            return model(**features)
        except TypeError:
            return model(features)

    def _maybe_concat_parameters(self, features: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        if "params" not in features:
            return features

        merged = dict(features)
        params = merged.pop("params")
        globals_tensor = merged.get("globals")
        if globals_tensor is None:
            merged["globals"] = params
            return merged

        if globals_tensor.shape[0] != params.shape[0]:
            logging.warning(
                "Batch globals and params have mismatched batch sizes (%d vs %d); skipping parameter concatenation.",
                globals_tensor.shape[0],
                params.shape[0],
            )
            return merged

        merged["globals"] = torch.cat([globals_tensor, params], dim=-1)
        return merged

    def _warn_if_global_dim_mismatch(self, globals_tensor: torch.Tensor) -> None:
        model = self._unwrap_model()
        expected_dim = getattr(model, "global_input_dim", None)
        if expected_dim is None or hasattr(self, "_warned_global_dim") and self._warned_global_dim:
            return
        actual_dim = globals_tensor.shape[-1]
        if actual_dim != expected_dim:
            logging.error(
                "Global feature dimension (%d) does not match model expectation (%d). "
                "If you added parameterized inputs, update global_input_dim accordingly.",
                actual_dim,
                expected_dim,
            )
            self._warned_global_dim = True

            exit(1)

    def _prepare_features(self, features: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        merged_features = self._maybe_concat_parameters(features)
        prepared: Dict[str, torch.Tensor] = {}
        for name, tensor in merged_features.items():
            tensor = tensor.to(self.device)
            if name in {"x", "globals"}:
                tensor = tensor.float()
            prepared[name] = tensor

        if "globals" in prepared:
            self._warn_if_global_dim_mismatch(prepared["globals"])
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
            batch_payload = {"features": features, "targets": targets, "weights": weights}
            for cb in self.callbacks:
                cb.on_batch_start(self, epoch, batch_idx, batch_payload, training)

            features = self._prepare_features(batch_payload["features"])
            targets = batch_payload["targets"].long().to(self.device)
            weights = batch_payload["weights"]
            weight_tensor: Optional[torch.Tensor] = None
            if weights is not None:
                weights = weights.to(self.device)
                finite_mask = torch.isfinite(weights)
                # Zero-out any non-finite weights to avoid NaNs in the loss
                weight_tensor = torch.where(finite_mask, weights, torch.zeros_like(weights))
                if not torch.all(finite_mask):
                    logging.debug("Non-finite weights detected; treating them as zero during loss computation.")

            with torch.set_grad_enabled(training):
                outputs = self._forward(model, features)
                if not torch.isfinite(outputs).all():
                    logging.debug("Non-finite outputs detected; treating them as zero."
                                  f"[Rank {self.rank}] Non-finite logits detected\n"
                                  f"min={outputs.min().item()}, max={outputs.max().item()}"
                                  )
                loss = compute_loss(outputs, targets, weight_tensor)
                if training:
                    optimizers = self.optimizers or ([self.optimizer] if self.optimizer else [])
                    for optimizer in optimizers:
                        optimizer.zero_grad()
                    loss.backward()
                    if self.config.grad_clip:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), self.config.grad_clip)
                    for optimizer in optimizers:
                        optimizer.step()
                    for scheduler in self.schedulers:
                        if scheduler:
                            scheduler.step()

            loss_value = loss.detach()

            if torch.isfinite(loss_value):
                metric_sum["loss"] += loss_value.item() * targets.size(0)
                metric_count["loss"] += targets.size(0)
            # else:
            #     # Optional: track how often this happens
            #     metric_sum["nan_loss"] += 1
            metric_count["loss"] += targets.size(0)

            logits_for_metrics = outputs.mean(dim=0) if outputs.dim() == 3 else outputs
            preds = torch.argmax(logits_for_metrics, dim=1)
            batch_accuracy = compute_accuracy(outputs, targets)
            if metric_tracker is not None:
                metric_tracker.update(preds, targets)
            else:
                metric_sum["accuracy"] += batch_accuracy * targets.size(0)
                metric_count["accuracy"] += targets.size(0)

            reduced_loss = self._reduce_mean_scalar(loss.item())
            reduced_accuracy = self._reduce_mean_scalar(batch_accuracy)

            if self.config.compute_physics_metrics:
                epoch_probs.append(logits_for_metrics.detach())
                epoch_targets.append(targets.detach())
                metric_weights = (
                    weight_tensor.detach()
                    if weight_tensor is not None
                    else torch.ones_like(targets, dtype=torch.float32, device=self.device)
                )
                epoch_weights.append(metric_weights)

            batch_metrics = {"loss": loss.item(), "accuracy": batch_accuracy}
            for cb in self.callbacks:
                cb.on_batch_end(
                    self, epoch, batch_idx, {"features": features, "targets": targets}, loss.item(), batch_metrics
                )

            if training:
                self.global_step += 1
                self._log_train_step(self.global_step, reduced_loss, reduced_accuracy, epoch)

            if progress is not None:
                progress.set_postfix({"loss": f"{reduced_loss:.4f}"}, refresh=False)
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
            metrics.update(
                self._compute_epoch_physics_metrics(epoch_probs, epoch_targets, epoch_weights, training=training)
            )
        if progress is not None:
            progress.close()
        return metrics

    def _log_train_step(self, step: int, loss: float, accuracy: float, epoch: int) -> None:
        if self.wandb_run is None or not self.is_rank_zero():
            return
        self.wandb_run.log(
            {
                "train/loss": loss,
                "metrics/train_accuracy": accuracy,
                "epoch": epoch + 1,
                **self._optimizer_learning_rates(),
            },
            step=step,
        )

    def _format_metric_group(self, metrics: Dict[str, float], prefix: str) -> Dict[str, float]:
        formatted: Dict[str, float] = {}
        for name, value in metrics.items():
            key = f"{prefix}/{name}" if name in {"loss", "auc"} else f"metrics/{prefix}_{name}"
            formatted[key] = value
        return formatted

    def _format_wandb_epoch_metrics(
            self, train_metrics: Dict[str, float], val_metrics: Dict[str, float]
    ) -> Dict[str, float]:
        payload: Dict[str, float] = {}
        payload.update(self._format_metric_group(train_metrics, "train"))
        if val_metrics:
            payload.update(self._format_metric_group(val_metrics, "val"))
        return payload

    def _optimizer_learning_rates(self) -> Dict[str, float]:
        lr_logs: Dict[str, float] = {}
        for tag, optimizer in zip(self.optimizer_tags, self.optimizers):
            if optimizer is None:
                continue
            lrs = [group.get("lr") for group in optimizer.param_groups if "lr" in group]
            for idx, lr in enumerate(lrs):
                if lr is None:
                    continue
                key = f"Optimizer/{tag}-lr" if len(lrs) == 1 else f"Optimizer/{tag}-lr-{idx}"
                lr_logs[key] = float(lr)
        return lr_logs

    def _compute_epoch_physics_metrics(
            self,
            probs_list: List[torch.Tensor],
            targets_list: List[torch.Tensor],
            weights_list: List[torch.Tensor],
            training: bool,
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
            logits=probs.cpu().numpy(),
            targets=targets.cpu().numpy(),
            weights=weights.cpu().numpy(),
            training=training,
            bins=self.config.physics_bins,
            min_bkg_events=self.config.sic_min_bkg_events,
            log_plots=self.wandb_run is not None,
            wandb_run=self.wandb_run,
            log_step=self.global_step,
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

    def _collect_predictions(
            self, dataset: EvenetTensorDataset, batch_size: int = 256
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        logging.info("Starting _collect_predictions")
        logging.info("Dataset size = %d", len(dataset))
        logging.info("Batch size = %d", batch_size)

        original_flag = getattr(dataset, "include_indices", False)
        dataset.include_indices = True

        sampler = DistributedSampler(dataset, shuffle=False) if self.world_size > 1 else None
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=sampler,
            shuffle=False,
            num_workers=self.config.num_workers,
        )

        if self.debug:
            logging.info(
                "world_size=%d, sampler=%s, num_workers=%d",
                self.world_size,
                "DistributedSampler" if sampler else "None",
                self.config.num_workers,
            )

        self.model.to(self.device)
        self.model.eval()

        local_outputs: List[torch.Tensor] = []
        local_indices: List[torch.Tensor] = []

        with torch.no_grad():
            for step, batch in enumerate(loader):
                features, _, _, *maybe_idx = batch
                batch_indices = maybe_idx[0] if maybe_idx else None

                if step == 0 and self.debug:
                    logging.info("First batch received")
                    if batch_indices is not None:
                        if self.debug:
                            logging.info(
                                "batch_indices shape=%s, min=%d, max=%d",
                                tuple(batch_indices.shape),
                                batch_indices.min().item(),
                                batch_indices.max().item(),
                            )

                features = self._prepare_features(features)
                outputs = self._forward(self.model, features)

                if step == 0 and self.debug:
                    logging.info("Raw outputs shape = %s", tuple(outputs.shape))

                # Ensemble case: [E, B, C] → [B, C]
                outputs = outputs.mean(dim=0) if outputs.dim() == 3 else outputs
                outputs = outputs.detach().cpu()

                local_outputs.append(outputs)
                if batch_indices is not None:
                    local_indices.append(batch_indices.cpu())

        dataset.include_indices = original_flag

        preds_tensor = torch.cat(local_outputs, dim=0) if local_outputs else torch.empty((0,))
        index_tensor = (
            torch.cat(local_indices, dim=0)
            if local_indices
            else torch.empty((0,), dtype=torch.long)
        )

        if self.debug:
            logging.info(
                "Local outputs: preds=%s, indices=%s",
                tuple(preds_tensor.shape),
                tuple(index_tensor.shape),
            )

        # ------------------------
        # DDP gather
        # ------------------------
        if self.world_size > 1:
            gathered_indices: List[Optional[torch.Tensor]] = [None for _ in range(self.world_size)]
            gathered_preds: List[Optional[torch.Tensor]] = [None for _ in range(self.world_size)]

            if self.debug:
                if dist.get_rank() == 0:
                    logging.info("default pg backend=%s world_size=%d", dist.get_backend(), dist.get_world_size())

                r = dist.get_rank()
                logging.info("[Rank %d] Before barrier", r)
                torch.cuda.synchronize()
                dist.barrier()
                logging.info("[Rank %d] After barrier", r)

                logging.info("rank=%s backend=%s preds_device=%s idx_device=%s",
                             dist.get_rank(), dist.get_backend(),
                             preds_tensor.device, index_tensor.device)

            dist.all_gather_object(gathered_indices, index_tensor)
            dist.all_gather_object(gathered_preds, preds_tensor)

            if self.debug:
                sizes = [
                    gi.shape if gi is not None else None
                    for gi in gathered_indices
                ]
                logging.info("Gathered index shapes per rank = %s", sizes)

            index_tensor = torch.cat([g for g in gathered_indices if g is not None], dim=0)
            preds_tensor = torch.cat([g for g in gathered_preds if g is not None], dim=0)

            if self.debug:
                logging.info(
                    "After gather: preds=%s, indices=%s",
                    tuple(preds_tensor.shape),
                    tuple(index_tensor.shape),
                )

            if index_tensor.numel() > 0:
                order = torch.argsort(index_tensor)
                index_tensor = index_tensor[order]
                preds_tensor = preds_tensor[order]

                unique_mask = torch.ones_like(index_tensor, dtype=torch.bool)
                unique_mask[1:] = index_tensor[1:] != index_tensor[:-1]
                unique_positions = torch.nonzero(unique_mask, as_tuple=False).squeeze(1)

                removed = index_tensor.numel() - unique_positions.numel()
                logging.info("Removed %d duplicated entries", removed)

                index_tensor = index_tensor[unique_positions]
                preds_tensor = preds_tensor[unique_positions]

        logging.info(
            "Finished prediction collection: preds=%s, indices=%s",
            tuple(preds_tensor.shape),
            tuple(index_tensor.shape),
        )

        return preds_tensor, index_tensor

    def predict(self, dataset: EvenetTensorDataset, batch_size: int = 256) -> torch.Tensor:
        preds, _ = self._collect_predictions(dataset, batch_size)
        return preds

    def evaluate(
            self,
            dataset: EvenetTensorDataset,
            batch_size: int = 256,
            output_path: Optional[str] = None,
    ) -> Dict[str, float]:
        preds, indices = self._collect_predictions(dataset, batch_size)

        if not self.is_rank_zero():
            return {}

        labels = dataset.labels[indices] if indices.numel() > 0 else dataset.labels
        weights = dataset.sample_weights[
            indices] if dataset.sample_weights is not None and indices.numel() > 0 else dataset.sample_weights
        raw_features = (
            {name: tensor[indices] for name, tensor in dataset.raw_features.items()}
            if indices.numel() > 0
            else dataset.raw_features
        )

        valid_weight_tensor = None
        if weights is not None:
            finite_mask = torch.isfinite(weights)
            valid_weight_tensor = weights if torch.all(finite_mask) else None

        loss = compute_loss(preds, labels, valid_weight_tensor)
        accuracy = compute_accuracy(preds, labels)
        metrics: Dict[str, float] = {"loss": float(loss.item()), "accuracy": float(accuracy)}

        if self.config.compute_physics_metrics and preds.numel() > 0:
            metrics.update(
                calculate_physics_metrics(
                    logits=preds.numpy(),
                    targets=labels.numpy(),
                    weights=(
                        weights.numpy()
                        if weights is not None
                        else torch.ones_like(labels, dtype=torch.float32).numpy()
                    ),
                    training=False,
                    bins=self.config.physics_bins,
                    min_bkg_events=self.config.sic_min_bkg_events,
                    log_plots=self.wandb_run is not None,
                    wandb_run=self.wandb_run,
                    f_name=Path(output_path) / "eval.png",
                )
            )

        saved_description = None
        if output_path:
            resolved_base = self._resolve_eval_base_path(Path(output_path))
            saved_description = f"{resolved_base.parent} ({resolved_base.stem}-sig/-bkg{resolved_base.suffix})"

            self._export_evaluation(
                base_path=resolved_base,
                preds=preds,
                labels=labels,
                weights=weights,
                raw_features=raw_features,
                metrics=metrics,
            )

        total_entries = int(labels.shape[0])
        sig_entries = int((labels == 1).sum().item()) if labels.numel() > 0 else 0
        bkg_entries = total_entries - sig_entries

        logging.info(
            "Evaluation completed on %d entries (signal=%d, background=%d). Metrics saved%s",
            total_entries,
            sig_entries,
            bkg_entries,
            f" to {saved_description}" if saved_description else " in-memory",
        )

        return metrics

    def _ensure_np_suffix(self, path: Path) -> Path:
        if path.suffix.lower() not in {".npz", ".npy"}:
            return path.with_suffix(path.suffix + ".npz") if path.suffix else path.with_suffix(".npz")
        return path

    def _resolve_eval_base_path(self, provided: Path) -> Path:
        """Resolve eval output path to a base NPZ path.

        If ``provided`` includes a numpy suffix, use it directly. Otherwise, treat it as a
        directory and drop files named ``eval_output-sig.npz``/``eval_output-bkg.npz`` inside.
        """

        if provided.suffix.lower() in {".npz", ".npy"}:
            return self._ensure_np_suffix(provided)
        return provided.joinpath("eval_output.npz")

    def _export_evaluation(
            self,
            *,
            base_path: Path,
            preds: torch.Tensor,
            labels: torch.Tensor,
            weights: Optional[torch.Tensor],
            raw_features: Dict[str, torch.Tensor],
            metrics: Dict[str, float],
    ) -> None:
        base_path = self._ensure_np_suffix(base_path)
        base_path.parent.mkdir(parents=True, exist_ok=True)

        preds_np = preds.cpu().numpy()
        labels_np = labels.cpu().numpy()
        weights_np = weights.cpu().numpy() if weights is not None else None
        features_np = {k: v.cpu().numpy() for k, v in raw_features.items()}

        expected_len = labels_np.shape[0]
        for name, arr in features_np.items():
            if arr.shape[0] != expected_len:
                raise ValueError(
                    f"Feature '{name}' has length {arr.shape[0]} but expected {expected_len} to match labels"
                )

        metric_arrays = {f"metric_{k}": np.array(v, dtype=np.float32) for k, v in metrics.items()}

        sig_mask = labels_np == 1
        bkg_mask = labels_np == 0
        suffixes = [("sig", sig_mask), ("bkg", bkg_mask)]
        for name, mask in suffixes:
            if mask.any():
                class_payload: Dict[str, np.ndarray] = {
                    **{k: v[mask] for k, v in features_np.items()},
                    "predictions": preds_np[mask],
                    "labels": labels_np[mask],
                    **({"sample_weights": weights_np[mask]} if weights_np is not None else {}),
                    "num_entries": np.array(mask.sum(), dtype=np.int64),
                }
                class_payload.update(metric_arrays)

                class_path = base_path.with_name(f"{base_path.stem}-{name}{base_path.suffix}")
                np.savez(class_path, **class_payload)
