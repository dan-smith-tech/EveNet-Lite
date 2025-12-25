

import logging
from pathlib import Path
import copy
from collections import Counter
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

import torch
import yaml
from evenet.control.global_config import DotDict

from .callbacks import Callback, EvenetLiteNormalizer, NormalizationCallback
from .data import EvenetTensorDataset
from .hf_utils import load_pretrained_weights
from .model import EveNetLite
from .optim import DEFAULT_HEAD_LR, DEFAULT_MODULE_GROUPS, DEFAULT_WEIGHT_DECAY
from .trainer import Trainer, TrainerConfig


class EvenetLiteClassifier:
    """High-level classifier API for Evenet-Lite."""

    DEFAULT_HF_REPO_ID="Avencast/EveNet"
    DEFAULT_HF_REPO_FILENAME="checkpoints.20M.a4.last.ckpt"

    DEFAULT_FEATURE_NAMES = {
        "x": [
            "energy",
            "pt",
            "eta",
            "phi",
            "isBTag",
            "isLepton",
            "charge",
        ],
        "globals": [
            "met",
            "met_phi",
            "nLepton",
            "nbJet",
            "nJet",
            "HT",
            "HT_lep",
            "M_all",
            "M_leps",
            "M_bjets",
        ],
    }

    DEFAULT_NORMALIZATION_RULES = {
        "x": {
            "energy": "log_normalize",
            "pt": "log_normalize",
            "eta": "normalize",
            "phi": "normalize_uniform",
            "isBTag": "none",
            "isLepton": "none",
            "charge": "none",
        },
        "globals": {
            "met": "log_normalize",
            "met_phi": "normalize",
            "nLepton": "none",
            "nbJet": "none",
            "nJet": "none",
            "HT": "log_normalize",
            "HT_lep": "log_normalize",
            "M_all": "log_normalize",
            "M_leps": "log_normalize",
            "M_bjets": "log_normalize",
        },
    }

    def __init__(
            self,
            class_labels: List[str],
            device: str = "auto",
            lr: List[float] | float = DEFAULT_HEAD_LR,
            weight_decay: List[float] | float = DEFAULT_WEIGHT_DECAY,
            model: torch.nn.Module = None,
            optimizer_fn: Optional[Callable[..., torch.optim.Optimizer]] = None,
            grad_clip: Optional[float] = None,
            scheduler_fn: Optional[Callable[..., Any]] = None,
            module_lists: Optional[List[List[str]]] = None,
            warmup_epochs: Optional[int] = 1,
            warmup_ratio: float = 0.1,
            warmup_start_factor: float = 0.1,
            min_lr: float = 0.0,
            global_input_dim: int = 10,
            sequential_input_dim: int = 7,
            use_wandb: bool = False,
            wandb: Optional[Dict[str, Any]] = None,
            log_level: int = logging.INFO,
            pretrained: bool = False,
            pretrained_source: str = "hf",
            pretrained_path: Optional[str] = None,
            pretrained_repo_id: Optional[str] = DEFAULT_HF_REPO_ID,
            pretrained_filename: Optional[str] = DEFAULT_HF_REPO_FILENAME,
            pretrained_cache_dir: Optional[str] = None,
            num_workers: int = 0,
            n_ensemble: int = 1,
            ensemble_mode: str = "independent",
    ) -> None:
        root_logger = logging.getLogger()
        log_format = "%(asctime)s | %(levelname)s | %(message)s"
        if not root_logger.handlers:
            logging.basicConfig(level=log_level, format=log_format)
        else:
            for handler in root_logger.handlers:
                handler.setFormatter(logging.Formatter(log_format))
                handler.setLevel(log_level)
        root_logger.setLevel(log_level)
        if model is None:
            default_config = Path(__file__).parent / 'config' / 'default_network_config.yaml'
            with open(str(default_config), 'r') as f:
                config = yaml.safe_load(f)
            logging.info(
                "Initializing default EveNet-Lite model from %s with global_input_dim=%d, sequential_input_dim=%d",
                default_config,
                global_input_dim,
                sequential_input_dim,
            )
            self.model = EveNetLite(
                config=DotDict(config),
                global_input_dim=global_input_dim,
                sequential_input_dim=sequential_input_dim,
                cls_label=class_labels,
                n_ensemble=n_ensemble,
                ensemble_mode=ensemble_mode,
            )
        else:
            self.model = model
        if pretrained:
            self._load_pretrained_weights(
                source=pretrained_source,
                local_path=pretrained_path,
                repo_id=pretrained_repo_id,
                filename=pretrained_filename,
                cache_dir=pretrained_cache_dir,
            )
        self.class_labels = class_labels
        self.config = TrainerConfig(
            device=device,
            lr=lr,
            weight_decay=weight_decay,
            module_lists=module_lists if module_lists is not None else [list(group) for group in DEFAULT_MODULE_GROUPS],
            grad_clip=grad_clip,
            optimizer_fn=optimizer_fn,
            scheduler_fn=scheduler_fn,
            warmup_epochs=warmup_epochs,
            warmup_ratio=warmup_ratio,
            warmup_start_factor=warmup_start_factor,
            min_lr=min_lr,
            use_wandb=use_wandb,
            wandb=wandb,
            num_workers=num_workers,
        )
        self.trainer: Optional[Trainer] = None
        self.normalizer: Optional[EvenetLiteNormalizer] = None
        self.feature_names: Optional[Dict[str, Iterable[str]]] = None
        self.debug: bool = False

    def fit(
            self,
            train_data: Tuple[Dict[str, torch.Tensor], torch.Tensor, Optional[torch.Tensor]],
            val_data: Optional[Tuple[Dict[str, torch.Tensor], torch.Tensor, Optional[torch.Tensor]]] = None,
            train_params: Optional[torch.Tensor] = None,
            val_params: Optional[torch.Tensor] = None,
            feature_names: Optional[Dict[str, Iterable[str]]] = None,
            normalization_rules: Optional[Dict[str, Dict[str, str]]] = None,
            normalization_stats: Optional[Dict[str, Any]] = None,
            callbacks: Optional[List[Callback]] = None,
            epochs: int = 10,
            batch_size: int = 256,
            sampler: Optional[str] = None,
            epoch_size: Optional[int] = None,
            checkpoint_path: Optional[str] = None,
            resume_from: Optional[str] = None,
            checkpoint_every: int = 1,
            save_top_k: int = 0,
            monitor_metric: str = "val_loss",
            minimize_metric: bool = True,
            early_stop_metric: str = "val_loss",
            early_stop_patience: int = 0,
            early_stop_minimize: bool = True,
            eval_data: Optional[Tuple[Dict[str, torch.Tensor], torch.Tensor, Optional[torch.Tensor]]] = None,
            eval_params: Optional[torch.Tensor] = None,
            eval_output_path: Optional[str] = None,
            eval_batch_size: Optional[int] = None,
            sic_min_bkg_events: int = 100,
            debug: bool = False,
    ) -> None:
        if feature_names is None:
            feature_names = copy.deepcopy(self.DEFAULT_FEATURE_NAMES)
        else:
            feature_names = copy.deepcopy(feature_names)

        param_dim = None
        for tensor in (train_params, val_params, eval_params):
            if tensor is not None:
                param_dim = tensor.shape[-1]
                break
        if param_dim is not None and "params" not in feature_names:
            feature_names["params"] = [f"param_{i}" for i in range(param_dim)]

        self.feature_names = feature_names
        if normalization_rules is None:
            normalization_rules = copy.deepcopy(self.DEFAULT_NORMALIZATION_RULES)
        callback_list = callbacks or []
        if not any(isinstance(cb, NormalizationCallback) for cb in callback_list):
            callback_list = [
                NormalizationCallback(
                    normalization_rules=normalization_rules, normalization_stats=normalization_stats
                )
            ] + callback_list
        elif normalization_rules is not None:
            for cb in callback_list:
                if isinstance(cb, NormalizationCallback):
                    cb.set_rules(normalization_rules)
                    cb.set_stats(normalization_stats)
                    break

        self.config.checkpoint_path = checkpoint_path
        self.config.resume_from = resume_from
        self.config.checkpoint_every = checkpoint_every
        self.config.save_top_k = save_top_k
        self.config.monitor_metric = monitor_metric
        self.config.minimize_metric = minimize_metric
        self.config.early_stop_metric = early_stop_metric
        self.config.early_stop_patience = early_stop_patience
        self.config.early_stop_minimize = early_stop_minimize
        self.config.eval_output_path = eval_output_path
        self.config.eval_batch_size = eval_batch_size
        self.config.sic_min_bkg_events = sic_min_bkg_events

        def _attach_params(
            data: Optional[Tuple[Dict[str, torch.Tensor], torch.Tensor, Optional[torch.Tensor]]],
            params: Optional[torch.Tensor],
        ) -> Optional[Tuple[Dict[str, torch.Tensor], torch.Tensor, Optional[torch.Tensor]]]:
            if data is None or params is None:
                return data
            features, labels, weights = data
            merged_features = dict(features)
            merged_features["params"] = params
            return merged_features, labels, weights

        train_data = _attach_params(train_data, train_params)
        val_data = _attach_params(val_data, val_params)
        eval_data = _attach_params(eval_data, eval_params)

        self.debug = debug
        self.trainer = Trainer(
            self.model, feature_names, self.config, callback_list, class_labels=self.class_labels, debug=debug
        )
        self.trainer.train(train_data, val_data, eval_data, epochs, batch_size, sampler, epoch_size)
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
            self.trainer = Trainer(
                self.model,
                feature_names,
                self.config,
                callbacks,
                class_labels=self.class_labels,
                debug=self.debug,
            )
        self.trainer.restore_checkpoint(path, map_location=map_location)
        norm_callback = next((cb for cb in self.trainer.callbacks if isinstance(cb, NormalizationCallback)), None)
        if norm_callback is not None:
            self.normalizer = norm_callback.normalizer

    def save_checkpoint(self, path: str) -> None:
        if self.trainer is None:
            raise RuntimeError("Model must be fitted before saving checkpoints")
        self.trainer.save_checkpoint(path)

    # ------------------------------------------------------------------
    # Pretrained weight loading
    # ------------------------------------------------------------------
    def _load_pretrained_weights(
        self,
        source: str = "hf",
        local_path: Optional[str] = None,
        repo_id: Optional[str] = None,
        filename: Optional[str] = None,
        cache_dir: Optional[str] = None,
    ) -> None:
        """Soft-load pretrained weights with shape safety and concise reporting."""

        logger = logging.getLogger(__name__)
        logger.info(
            "Loading pretrained weights using source=%s (repo_id=%s, filename=%s, local_path=%s)",
            source,
            repo_id,
            filename,
            local_path,
        )

        checkpoint: Optional[Dict[str, Any]] = None
        if source == "hf":
            if repo_id is None or filename is None:
                logger.warning("Pretrained source set to HF but repo_id/filename not provided; skipping load.")
                return
            checkpoint = load_pretrained_weights(repo_id=repo_id, filename=filename, cache_dir=cache_dir)
            if checkpoint is None:
                logger.warning("Failed to load pretrained weights from Hugging Face for repo %s", repo_id)
                return
        elif source == "local":
            if local_path is None:
                logger.warning("Local pretrained weights requested but no path provided; skipping load.")
                return
            resolved = Path(local_path).expanduser()
            if not resolved.exists():
                logger.warning("Local pretrained weights not found at %s; skipping load.", resolved)
                return
            checkpoint = torch.load(resolved, map_location="cpu")
        else:
            logger.warning("Unknown pretrained source '%s'; skipping load.", source)
            return

        state = checkpoint.get("state_dict") if isinstance(checkpoint, dict) else None
        if state is None and isinstance(checkpoint, dict):
            state = checkpoint.get("model")
        if state is None:
            state = checkpoint

        if not isinstance(state, dict):
            logger.warning("Pretrained checkpoint did not contain a state dict; skipping load.")
            return

        self._soft_load_state_dict(state)

    def _soft_load_state_dict(self, state: Dict[str, torch.Tensor]) -> None:
        """Partially load a state dict when shapes match and report a summary."""

        logger = logging.getLogger(__name__)
        ckpt_state = {k.replace("model.", "").replace("module.", ""): v for k, v in state.items()}
        if hasattr(self.model, "expand_state_dict"):
            ckpt_state = self.model.expand_state_dict(ckpt_state)
        model_state = self.model.state_dict()

        loaded_keys: List[str] = []
        shape_mismatch_keys: List[str] = []
        missing_keys: List[str] = []
        unexpected_keys: List[str] = []

        filtered_state: Dict[str, torch.Tensor] = {}

        for key, model_value in model_state.items():
            if key in ckpt_state:
                ckpt_value = ckpt_state[key]
                if ckpt_value.shape == model_value.shape:
                    filtered_state[key] = ckpt_value
                    loaded_keys.append(key)
                else:
                    shape_mismatch_keys.append(
                        f"{key} (ckpt: {list(ckpt_value.shape)} vs model: {list(model_value.shape)})"
                    )
            else:
                missing_keys.append(key)

        for key in ckpt_state:
            if key not in model_state:
                unexpected_keys.append(key)

        self.model.load_state_dict(filtered_state, strict=False)

        logger.info("======== Soft Load Summary ========")
        total_layers = len(model_state)
        loaded_layers = len(loaded_keys)
        load_ratio = loaded_layers / total_layers if total_layers > 0 else 0.0
        fully_loaded = loaded_layers == total_layers and not (shape_mismatch_keys or missing_keys or unexpected_keys)

        logger.info(
            "Loaded %d / %d layers (%.1f%%)",
            loaded_layers,
            total_layers,
            load_ratio * 100,
        )
        logger.info("All components loaded: %s", "YES" if fully_loaded else "NO")

        if model_state:
            component_copies = {}
            if hasattr(self.model, "component_copies"):
                component_copies = self.model.component_copies()

            def _component_prefix(key: str) -> str:
                tokens = key.split(".")
                if tokens and tokens[0] == "models" and len(tokens) >= 3:
                    tokens = tokens[2:]  # drop "models.<idx>"
                if tokens and tokens[0] == "backbone":
                    tokens = tokens[1:]
                if tokens and tokens[0] in {"Classification", "GlobalEmbedding", "PET", "ObjectEncoder"}:
                    if len(tokens) > 1 and tokens[1].isdigit():
                        return tokens[0]
                    return tokens[0]
                return tokens[0] if tokens else key

            total_groups = Counter([_component_prefix(k) for k in model_state])
            loaded_groups = Counter([_component_prefix(k) for k in loaded_keys])
            missing_groups = Counter([_component_prefix(k) for k in missing_keys])
            mismatch_groups = Counter([_component_prefix(k) for k in shape_mismatch_keys])

            logger.info("--- Breakdown ---")
            for prefix, total in total_groups.items():
                loaded_count = loaded_groups.get(prefix, 0)
                missing_count = missing_groups.get(prefix, 0)
                mismatch_count = mismatch_groups.get(prefix, 0)
                not_loaded = total - loaded_count
                copies = component_copies.get(prefix, 1)
                logger.info(
                    "• %s [copies=%d]: %d/%d loaded (%d not loaded: %d missing, %d mismatched)",
                    prefix.ljust(15),
                    copies,
                    loaded_count,
                    total,
                    not_loaded,
                    missing_count,
                    mismatch_count,
                )

        if shape_mismatch_keys:
            logger.warning("Shape mismatches for %d layers", len(shape_mismatch_keys))
            for msg in shape_mismatch_keys[:5]:
                logger.warning("  - %s", msg)
            if len(shape_mismatch_keys) > 5:
                logger.warning("  - ... and %d more", len(shape_mismatch_keys) - 5)

        if missing_keys:
            missing_groups = Counter([k.split(".")[0] for k in missing_keys])
            logger.warning("Missing layers (randomly initialized): %d", len(missing_keys))
            for prefix, count in missing_groups.items():
                logger.warning("• %s: %d layers missing", prefix.ljust(15), count)

        if unexpected_keys:
            logger.warning("Unexpected layers in checkpoint (ignored): %d", len(unexpected_keys))

        logger.info("===================================")
