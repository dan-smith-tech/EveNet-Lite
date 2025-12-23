
import copy
import logging
import math
from typing import Any, Dict, Iterable, List, Optional, Sequence

import torch
from torch.distributions import Normal


class Callback:
    """Base class for Evenet-Lite callbacks.

    Subclasses can override any hook to inject custom behavior.
    """

    def on_train_start(self, trainer: "Trainer") -> None:  # pragma: no cover - interface
        pass

    def on_batch_start(
        self, trainer: "Trainer", epoch: int, batch_idx: int, batch: Dict[str, Any], training: bool
    ) -> None:  # pragma: no cover - interface
        pass

    def on_epoch_start(self, trainer: "Trainer", epoch: int) -> None:  # pragma: no cover - interface
        pass

    def on_batch_end(
        self, trainer: "Trainer", epoch: int, batch_idx: int, batch: Dict[str, Any], loss: float, metrics: Optional[Dict[str, float]] = None,
    ) -> None:  # pragma: no cover - interface
        pass

    def on_epoch_end(self, trainer: "Trainer", epoch: int, metrics: Dict[str, float]) -> None:  # pragma: no cover - interface
        pass

    def on_train_end(self, trainer: "Trainer") -> None:  # pragma: no cover - interface
        pass


class ParameterRandomizationCallback(Callback):
    """Dynamically randomize parameter inputs for background events.

    During training/validation, background samples (identified via
    ``background_label``) receive fresh parameter values sampled uniformly from
    the provided ``min_values``/``max_values`` range. Signal samples keep their
    original parameter values. Optionally, ranges are inferred from the training
    dataset if not supplied.
    """

    def __init__(
        self,
        param_key: str = "params",
        background_label: int = 0,
        min_values: Optional[Sequence[float]] = None,
        max_values: Optional[Sequence[float]] = None,
        apply_to_validation: bool = True,
    ) -> None:
        self.param_key = param_key
        self.background_label = background_label
        self._configured_min = min_values
        self._configured_max = max_values
        self.apply_to_validation = apply_to_validation

        self._resolved_min: Optional[torch.Tensor] = None
        self._resolved_max: Optional[torch.Tensor] = None
        self._warned_missing_params = False
        self._warned_bounds = False

    def _maybe_warn(self, message: str) -> None:
        if not self._warned_missing_params:
            logging.warning(message)
            self._warned_missing_params = True

    def _resolve_bounds(self, param_dim: int, device: torch.device) -> Optional[torch.Tensor]:
        def _to_tensor(values: Optional[Sequence[float]], name: str) -> Optional[torch.Tensor]:
            if values is None:
                return None
            tensor = torch.as_tensor(values, dtype=torch.float32, device=device)
            if tensor.numel() == 1 and param_dim > 1:
                tensor = tensor.expand(param_dim)
            if tensor.numel() != param_dim:
                if not self._warned_bounds:
                    logging.warning(
                        "ParameterRandomizationCallback %s size %d does not match param_dim=%d; skipping randomization.",
                        name,
                        tensor.numel(),
                        param_dim,
                    )
                    self._warned_bounds = True
                return None
            return tensor

        min_values = self._resolved_min if self._resolved_min is not None else _to_tensor(self._configured_min, "min_values")
        max_values = self._resolved_max if self._resolved_max is not None else _to_tensor(self._configured_max, "max_values")

        if min_values is None or max_values is None:
            return None

        if min_values.shape != max_values.shape:
            if not self._warned_bounds:
                logging.warning(
                    "ParameterRandomizationCallback min/max shapes differ (%s vs %s); skipping randomization.",
                    tuple(min_values.shape),
                    tuple(max_values.shape),
                )
                self._warned_bounds = True
            return None

        return torch.stack([min_values, max_values])

    def on_train_start(self, trainer: "Trainer") -> None:
        if (self._configured_min is None or self._configured_max is None) and hasattr(
            trainer, "train_dataset"
        ):
            raw_params = trainer.train_dataset.raw_features.get(self.param_key)
            if raw_params is not None and raw_params.numel() > 0:
                self._resolved_min = torch.as_tensor(raw_params).amin(dim=0).float()
                self._resolved_max = torch.as_tensor(raw_params).amax(dim=0).float()
            else:
                self._maybe_warn(
                    "ParameterRandomizationCallback could not infer parameter bounds because the training dataset "
                    f"does not include '{self.param_key}'."
                )

    def on_batch_start(
        self, trainer: "Trainer", epoch: int, batch_idx: int, batch: Dict[str, Any], training: bool
    ) -> None:
        if not training and not self.apply_to_validation:
            return

        features = batch.get("features", {})
        targets = batch.get("targets")
        if targets is None:
            return

        if self.param_key not in features:
            if training:
                self._maybe_warn(
                    f"ParameterRandomizationCallback skipped because '{self.param_key}' is absent in batch features."
                )
            return

        params = features[self.param_key]
        if params.dim() < 2:
            return

        bounds = self._resolve_bounds(params.shape[-1], params.device)
        if bounds is None:
            return

        min_values, max_values = bounds[0], bounds[1]
        bkg_mask = targets == self.background_label
        if not torch.any(bkg_mask):
            return

        random_raw = torch.rand((int(bkg_mask.sum().item()), params.shape[-1]), device=params.device, dtype=params.dtype)
        replacement = random_raw * (max_values - min_values) + min_values

        normalizer = next((cb.normalizer for cb in trainer.callbacks if hasattr(cb, "normalizer")), None)
        if normalizer is not None and hasattr(normalizer, "transform"):
            try:
                replacement = normalizer.transform({self.param_key: replacement})[self.param_key]
            except Exception:
                pass

        params = params.clone()
        params[bkg_mask] = replacement
        features[self.param_key] = params


class EvenetLiteNormalizer:
    """Feature-wise normalizer with configurable rules.

    Supported rules
    --------------
    ``normalize``
        Standard score normalization ``(x - mean) / std``.
    ``log_normalize``
        Apply ``log1p`` before standardization to stabilize long-tailed
        distributions.
    ``normalize_uniform``
        Map standardized values to a unit Gaussian through a uniform
        distribution, matching the behavior used in the reference EveNet code.
    ``none``
        Skip normalization for the feature.
    """

    def __init__(self, normalization_rules: Optional[Dict[str, Dict[str, str]]] = None) -> None:
        self.normalization_rules: Dict[str, Dict[str, str]] = normalization_rules or {}
        self._stats: Dict[str, Dict[str, torch.Tensor]] = {}
        self._resolved_rules: Dict[str, List[str]] = {}
        self._feature_names: Dict[str, List[str]] = {}
        self._normal = Normal(0, 1)

    def set_rules(self, normalization_rules: Optional[Dict[str, Dict[str, str]]]) -> None:
        """Update normalization rules.

        Args:
            normalization_rules: Mapping from feature group (e.g., ``"x"`` or
                ``"globals"``) to per-feature rule names.
        """

        self.normalization_rules = normalization_rules or {}

    def _resolve_feature_names(self, key: str, tensor: torch.Tensor, provided: Iterable[str]) -> List[str]:
        names = list(provided)
        feature_dim = tensor.shape[-1] if tensor.dim() > 1 else 1
        if len(names) < feature_dim:
            names = names + [f"feature_{i}" for i in range(len(names), feature_dim)]
        elif len(names) > feature_dim:
            names = names[:feature_dim]
        return names

    def fit(self, data: Dict[str, torch.Tensor], feature_names: Dict[str, Iterable[str]]) -> None:
        """Compute statistics from the training data.

        Args:
            data: Mapping of raw feature tensors with leading batch dimension.
            feature_names: Human-readable feature names per tensor.
        """

        self._stats = {}
        self._resolved_rules = {}
        self._feature_names = {}

        for key, tensor in data.items():
            if not torch.is_tensor(tensor):
                tensor = torch.as_tensor(tensor)
            if tensor.numel() == 0 or tensor.dtype not in (torch.float32, torch.float64, torch.float16, torch.bfloat16):
                continue

            names = self._resolve_feature_names(key, tensor, feature_names.get(key, []))
            rules_for_key = self.normalization_rules.get(key, {})
            rules: List[str] = []
            feature_dim = tensor.shape[-1]
            flat = tensor.float().view(-1, feature_dim)

            mask = None
            mask_key = f"{key}_mask"
            if mask_key in data:
                mask = torch.as_tensor(data[mask_key]).float().view(-1)
                if mask.shape[0] == flat.shape[0]:
                    flat = flat[mask > 0]

            if flat.numel() == 0:
                continue

            mean = torch.zeros(feature_dim, device=flat.device)
            std = torch.ones(feature_dim, device=flat.device)

            for idx in range(feature_dim):
                name = names[idx]
                rule = rules_for_key.get(name, "none")
                rules.append(rule)
                if rule == "none":
                    continue
                feature_values = flat[:, idx]
                mean[idx] = feature_values.mean()
                std[idx] = feature_values.std(unbiased=False).clamp_min(1e-6)

            self._stats[key] = {"mean": mean, "std": std}
            self._resolved_rules[key] = rules
            self._feature_names[key] = names

    def apply_user_stats(
        self,
        data: Dict[str, torch.Tensor],
        feature_names: Dict[str, Iterable[str]],
        normalization_stats: Dict[str, Any],
    ) -> None:
        """Load externally-provided statistics, filling gaps with defaults.

        The provided ``normalization_stats`` is expected to mirror the structure
        of :meth:`state_dict`, containing mean/std tensors per feature group.
        When tensors are missing or have mismatched dimensions, missing values
        default to mean ``0`` and std ``1`` so that inputs remain unchanged.
        """

        self._stats = {}
        self._resolved_rules = {}
        self._feature_names = {}

        normalization_stats = normalization_stats or {}
        if not normalization_stats:
            logging.info(
                "No normalization statistics provided; defaulting to identity normalization (mean=0, std=1)."
            )

        for key, tensor in data.items():
            if not torch.is_tensor(tensor):
                tensor = torch.as_tensor(tensor)
            if tensor.numel() == 0 or tensor.dtype not in (torch.float32, torch.float64, torch.float16, torch.bfloat16):
                continue

            names = self._resolve_feature_names(key, tensor, feature_names.get(key, []))
            rules_for_key = self.normalization_rules.get(key, {})
            rules: List[str] = []
            feature_dim = tensor.shape[-1]

            mean = torch.zeros(feature_dim, device=tensor.device, dtype=torch.float32)
            std = torch.ones(feature_dim, device=tensor.device, dtype=torch.float32)
            provided = normalization_stats.get(key, {}) if normalization_stats is not None else {}

            for idx in range(feature_dim):
                name = names[idx]
                rules.append(rules_for_key.get(name, "none"))

            provided_mean = provided.get("mean")
            if provided_mean is not None:
                pm = torch.as_tensor(provided_mean, dtype=torch.float32, device=mean.device).flatten()
                if pm.numel() != feature_dim:
                    logging.warning(
                        "Provided mean for %s has %d elements (expected %d); truncating/padding with defaults.",
                        key,
                        pm.numel(),
                        feature_dim,
                    )
                mean[: min(pm.numel(), feature_dim)] = pm[:feature_dim]
            else:
                logging.info("No mean provided for %s; using zeros.", key)

            provided_std = provided.get("std")
            if provided_std is not None:
                ps = torch.as_tensor(provided_std, dtype=torch.float32, device=std.device).flatten()
                if ps.numel() != feature_dim:
                    logging.warning(
                        "Provided std for %s has %d elements (expected %d); truncating/padding with defaults.",
                        key,
                        ps.numel(),
                        feature_dim,
                    )
                std[: min(ps.numel(), feature_dim)] = ps[:feature_dim]
            else:
                logging.info("No std provided for %s; using ones.", key)

            std = std.clamp_min(1e-6)

            self._stats[key] = {"mean": mean, "std": std}
            self._resolved_rules[key] = rules
            self._feature_names[key] = names

    def _apply_rule(self, tensor: torch.Tensor, mean: torch.Tensor, std: torch.Tensor, rule: str) -> torch.Tensor:
        if rule == "none":
            return tensor
        if rule == "log_normalize":
            tensor = tensor.log1p()
        tensor = (tensor - mean) / std
        if rule == "normalize_uniform":
            tensor = (tensor + math.sqrt(3)) / (2 * math.sqrt(3))
            tensor = torch.clamp(tensor, 1e-6, 1 - 1e-6)
            tensor = self._normal.icdf(tensor)
        return tensor

    def _invert_rule(self, tensor: torch.Tensor, mean: torch.Tensor, std: torch.Tensor, rule: str) -> torch.Tensor:
        if rule == "normalize_uniform":
            tensor = self._normal.cdf(tensor)
            tensor = tensor * 2 * math.sqrt(3) - math.sqrt(3)
        if rule != "none":
            tensor = tensor * std + mean
        if rule == "log_normalize":
            tensor = tensor.expm1()
        return tensor

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
            rules = self._resolved_rules.get(key, [])
            mean = stats["mean"]
            std = stats["std"]
            feature_dim = mean.shape[0]
            reshaped = tensor.float().view(-1, feature_dim)
            normed_features: List[torch.Tensor] = []
            for idx in range(feature_dim):
                normed_features.append(
                    self._apply_rule(reshaped[:, idx], mean[idx], std[idx], rules[idx] if idx < len(rules) else "normalize")
                )
            stacked = torch.stack(normed_features, dim=-1)
            normalized_tensor = stacked.view_as(tensor)

            mask_key = f"{key}_mask"
            if mask_key in data and isinstance(data[mask_key], torch.Tensor) and data[mask_key].dim() >= tensor.dim() - 1:
                mask = data[mask_key].unsqueeze(-1).to(normalized_tensor.device)
                normalized_tensor = normalized_tensor * mask
            normalized[key] = normalized_tensor
        return normalized

    def denormalize(self, data: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        if not self._stats:
            return data

        restored: Dict[str, torch.Tensor] = {}
        for key, tensor in data.items():
            if key not in self._stats:
                restored[key] = tensor
                continue

            stats = self._stats[key]
            rules = self._resolved_rules.get(key, [])
            mean = stats["mean"]
            std = stats["std"]
            feature_dim = mean.shape[0]
            reshaped = tensor.float().view(-1, feature_dim)
            restored_features: List[torch.Tensor] = []
            for idx in range(feature_dim):
                restored_features.append(
                    self._invert_rule(reshaped[:, idx], mean[idx], std[idx], rules[idx] if idx < len(rules) else "normalize")
                )
            restored_tensor = torch.stack(restored_features, dim=-1).view_as(tensor)
            restored[key] = restored_tensor
        return restored

    def state_dict(self) -> Dict[str, Any]:
        return {
            "stats": copy.deepcopy(self._stats),
            "rules": copy.deepcopy(self._resolved_rules),
            "feature_names": copy.deepcopy(self._feature_names),
        }

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        if not state_dict:
            return
        self._stats = copy.deepcopy(state_dict.get("stats", {}))
        self._resolved_rules = copy.deepcopy(state_dict.get("rules", {}))
        self._feature_names = copy.deepcopy(state_dict.get("feature_names", {}))


class NormalizationCallback(Callback):
    """Callback that handles fitting and applying a normalizer.

    The normalizer is fitted on the training data at the start of training and
    then attached to all datasets managed by the trainer.
    """

    def __init__(
        self,
        normalizer: Optional[EvenetLiteNormalizer] = None,
        normalization_rules: Optional[Dict[str, Dict[str, str]]] = None,
        normalization_stats: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.normalizer = normalizer or EvenetLiteNormalizer(normalization_rules)
        if normalization_rules is not None:
            self.normalizer.set_rules(normalization_rules)
        self.normalization_stats = normalization_stats

    def set_rules(self, normalization_rules: Optional[Dict[str, Dict[str, str]]]) -> None:
        self.normalizer.set_rules(normalization_rules)

    def set_stats(self, normalization_stats: Optional[Dict[str, Any]]) -> None:
        self.normalization_stats = normalization_stats

    def on_train_start(self, trainer: "Trainer") -> None:
        train_data = trainer.train_dataset.raw_features
        feature_names = trainer.feature_names
        if self.normalization_stats is not None:
            self.normalizer.apply_user_stats(train_data, feature_names, self.normalization_stats)
            trainer.attach_normalizer(self.normalizer)
            logging.info(
                "Loaded provided normalizer stats for features: %s",
                {k: list(v) for k, v in feature_names.items()},
            )
        else:
            self.normalizer.fit(train_data, feature_names)
            trainer.attach_normalizer(self.normalizer)
            logging.info("Fitted normalizer with features: %s", {k: list(v) for k, v in feature_names.items()})

    def state_dict(self) -> Dict[str, Any]:
        return {"normalizer": self.normalizer.state_dict()}

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        if "normalizer" in state_dict:
            self.normalizer.load_state_dict(state_dict["normalizer"])


class DebugCallback(Callback):
    """Verbose, rank-safe logging helper for rapid debugging.

    Logs sampler summaries at the start of each epoch, gradient norms after
    backward passes, and per-batch/epoch metrics. Only the global rank 0
    process emits logs when running under DDP.
    """

    def __init__(self, log_every_n_batches: int = 10) -> None:
        self.log_every_n_batches = max(1, log_every_n_batches)

    def _grad_norms(self, model: torch.nn.Module) -> Optional[Dict[str, float]]:
        norms = [p.grad.detach().data.norm(2) for p in model.parameters() if p.grad is not None]
        if not norms:
            return None
        stacked = torch.stack(norms)
        total = torch.sqrt(torch.sum(stacked**2)).item()
        return {"total": float(total), "max": float(stacked.max().item())}

    def on_epoch_start(self, trainer: "Trainer", epoch: int) -> None:
        if not trainer.is_rank_zero():
            return
        train_sampler = trainer.train_sampler if hasattr(trainer, "train_sampler") else None
        val_sampler = trainer.val_sampler if hasattr(trainer, "val_sampler") else None
        train_loader = trainer.train_loader if hasattr(trainer, "train_loader") else None
        val_loader = trainer.val_loader if hasattr(trainer, "val_loader") else None

        train_sampler_desc = trainer._describe_sampler(train_sampler) if train_sampler is not None else "None"
        val_sampler_desc = trainer._describe_sampler(val_sampler) if val_sampler is not None else "None"
        train_steps = len(train_loader) if train_loader is not None else 0
        val_steps = len(val_loader) if val_loader is not None else 0

        logging.info(
            "[Debug] Epoch %d start | train_steps=%d | train_sampler=%s | world_size=%d",
            epoch + 1,
            train_steps,
            train_sampler_desc,
            trainer.world_size,
        )
        if val_loader is not None:
            logging.info(
                "[Debug] Epoch %d validation | val_steps=%d | val_sampler=%s",
                epoch + 1,
                val_steps,
                val_sampler_desc,
            )

    def on_batch_end(
        self,
        trainer: "Trainer",
        epoch: int,
        batch_idx: int,
        batch: Dict[str, torch.Tensor],
        loss: float,
        metrics: Optional[Dict[str, float]] = None,
    ) -> None:
        if not trainer.is_rank_zero() or not trainer.model.training:
            return
        if batch_idx % self.log_every_n_batches != 0:
            return

        grad_norms = self._grad_norms(trainer._unwrap_model())
        metric_parts = [f"loss={loss:.4f}"]
        if metrics:
            metric_parts.extend([f"{k}={v:.4f}" for k, v in metrics.items() if v is not None])
        if grad_norms is not None:
            metric_parts.append(
                "grad_norm(total={total:.4f}, max={max:.4f})".format(
                    total=grad_norms["total"], max=grad_norms["max"]
                )
            )

        logging.info(
            "[Debug] Epoch %d | Batch %d -> %s", epoch + 1, batch_idx + 1, ", ".join(metric_parts)
        )

    def on_epoch_end(self, trainer: "Trainer", epoch: int, metrics: Dict[str, float]) -> None:
        if not trainer.is_rank_zero():
            return
        if not metrics:
            logging.info("[Debug] Epoch %d end | no metrics available", epoch + 1)
            return
        formatted = ", ".join([f"{k}={v:.4f}" for k, v in metrics.items()])
        logging.info("[Debug] Epoch %d complete | %s", epoch + 1, formatted)
