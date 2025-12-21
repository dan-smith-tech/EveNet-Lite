
import copy
import logging
import math
from typing import Any, Dict, Iterable, List, Optional

import torch
from torch.distributions import Normal


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
                rule = rules_for_key.get(name, "normalize")
                rules.append(rule)
                if rule == "none":
                    continue
                feature_values = flat[:, idx]
                mean[idx] = feature_values.mean()
                std[idx] = feature_values.std(unbiased=False).clamp_min(1e-6)

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
    ) -> None:
        self.normalizer = normalizer or EvenetLiteNormalizer(normalization_rules)
        if normalization_rules is not None:
            self.normalizer.set_rules(normalization_rules)

    def set_rules(self, normalization_rules: Optional[Dict[str, Dict[str, str]]]) -> None:
        self.normalizer.set_rules(normalization_rules)

    def on_train_start(self, trainer: "Trainer") -> None:
        train_data = trainer.train_dataset.raw_features
        feature_names = trainer.feature_names
        self.normalizer.fit(train_data, feature_names)
        trainer.attach_normalizer(self.normalizer)
        logging.info("Fitted normalizer with features: %s", {k: list(v) for k, v in feature_names.items()})

    def state_dict(self) -> Dict[str, Any]:
        return {"normalizer": self.normalizer.state_dict()}

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        if "normalizer" in state_dict:
            self.normalizer.load_state_dict(state_dict["normalizer"])
