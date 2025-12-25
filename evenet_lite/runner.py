import logging
import os
from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch

from .callbacks import Callback
from .classifier import EvenetLiteClassifier


def _detect_ddp() -> Tuple[bool, int, int]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    return world_size > 1, world_size, local_rank


def _configure_logging(log_level: int) -> None:
    root_logger = logging.getLogger()
    log_format = "%(asctime)s | %(levelname)s | %(message)s"
    if not root_logger.handlers:
        logging.basicConfig(level=log_level, format=log_format)
    else:
        for handler in root_logger.handlers:
            handler.setFormatter(logging.Formatter(log_format))
            handler.setLevel(log_level)
    root_logger.setLevel(log_level)


def run_evenet_lite_training(
    train_features: Dict[str, torch.Tensor],
    train_labels: torch.Tensor,
    train_weights: Optional[torch.Tensor] = None,
    *,
    class_labels: List[str],
    val_features: Optional[Dict[str, torch.Tensor]] = None,
    val_labels: Optional[torch.Tensor] = None,
    val_weights: Optional[torch.Tensor] = None,
    feature_names: Optional[Dict[str, Iterable[str]]] = None,
    normalization_rules: Optional[Dict[str, Dict[str, str]]] = None,
    normalization_stats: Optional[Dict[str, Any]] = None,
    callbacks: Optional[List[Callback]] = None,
    sampler: Optional[str] = None,
    epoch_size: Optional[int] = None,
    epochs: int = 10,
    batch_size: int = 256,
    checkpoint_path: Optional[str] = None,
    resume_from: Optional[str] = None,
    checkpoint_every: int = 1,
    save_top_k: int = 0,
    monitor_metric: str = "val_loss",
    minimize_metric: bool = True,
    early_stop_metric: str = "val_loss",
    early_stop_minimize: bool = True,
    early_stop_patience: int = 0,
    eval_features: Optional[Dict[str, torch.Tensor]] = None,
    eval_labels: Optional[torch.Tensor] = None,
    eval_weights: Optional[torch.Tensor] = None,
    eval_output_path: Optional[str] = None,
    eval_batch_size: Optional[int] = None,
    sic_min_bkg_events: int = 100,
    debug: bool = False,
    log_level: int = logging.INFO,
    loss_gamma: float = 0.0,
    **classifier_kwargs: Any,
) -> EvenetLiteClassifier:
    """Convenience entrypoint for running Evenet-Lite training on prepared tensors.

    The runner assembles the expected training tuples, detects DDP from
    ``torchrun`` environment variables, instantiates :class:`EvenetLiteClassifier`,
    and invokes :meth:`EvenetLiteClassifier.fit`. When no distributed variables
    are present, it defaults to single-GPU (or CPU) execution.

    Args:
        train_features: Mapping of feature group name to tensor with shape
            matching the model contract (e.g., ``{"objects": Tensor[N, M, F]}``).
        train_labels: Class indices for each training example.
        train_weights: Optional per-example weights aligned with ``train_labels``.
        class_labels: Ordered list of class names passed to
            :class:`EvenetLiteClassifier`.
        val_features: Optional validation features following the same structure
            as ``train_features``.
        val_labels: Optional validation labels aligned with ``val_features``.
        val_weights: Optional per-example validation weights.
        feature_names: Optional mapping of feature group name to the list of
            feature strings, used by normalization callbacks.
        normalization_rules: Optional normalization configuration passed through
            to :meth:`EvenetLiteClassifier.fit`.
        normalization_stats: Optional precomputed normalization statistics that
            bypass fitting. Missing entries default to mean ``0`` and std ``1``.
        callbacks: Additional callbacks to append to training (debugging,
            metrics logging, etc.).
        sampler: Sampler strategy name (e.g., ``"weighted"``) forwarded to
            :class:`Trainer`.
        epoch_size: Optional number of samples per epoch when using a sampler.
        epochs: Number of epochs to train.
        batch_size: Mini-batch size.
        checkpoint_path: Directory for checkpoints.
        resume_from: Optional path to a checkpoint to resume training from.
        checkpoint_every: Checkpoint frequency in epochs.
        save_top_k: Maximum number of checkpoints to retain when monitoring a
            metric.
        monitor_metric: Metric name used for checkpoint ranking.
        minimize_metric: Whether ``monitor_metric`` should be minimized.
        loss_gamma: Focal-loss gamma parameter (``0`` reduces to standard cross-entropy).
        debug: Whether to enable verbose ``DebugCallback`` logging.
        log_level: Logging level applied before runner diagnostics and forwarded
            to the classifier when unspecified.
        **classifier_kwargs: Any additional arguments forwarded to
            :class:`EvenetLiteClassifier`.
    """

    _configure_logging(log_level)

    is_ddp, world_size, local_rank = _detect_ddp()
    if is_ddp:
        logging.info("Detected DDP environment (WORLD_SIZE=%d, LOCAL_RANK=%d)", world_size, local_rank)
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
    else:
        logging.info("No DDP environment variables detected; running in single-process mode")

    if "log_level" not in classifier_kwargs:
        classifier_kwargs["log_level"] = log_level

    classifier = EvenetLiteClassifier(class_labels=class_labels, loss_gamma=loss_gamma, **classifier_kwargs)

    train_payload = (train_features, train_labels, train_weights)
    val_payload = None
    if val_features is not None and val_labels is not None:
        val_payload = (val_features, val_labels, val_weights)

    classifier.fit(
        train_data=train_payload,
        val_data=val_payload,
        eval_data=(eval_features, eval_labels, eval_weights) if eval_features is not None and eval_labels is not None else None,
        feature_names=feature_names,
        normalization_rules=normalization_rules,
        normalization_stats=normalization_stats,
        callbacks=callbacks,
        epochs=epochs,
        batch_size=batch_size,
        sampler=sampler,
        epoch_size=epoch_size,
        checkpoint_path=checkpoint_path,
        resume_from=resume_from,
        checkpoint_every=checkpoint_every,
        save_top_k=save_top_k,
        monitor_metric=monitor_metric,
        minimize_metric=minimize_metric,
        early_stop_metric=early_stop_metric,
        early_stop_minimize=early_stop_minimize,
        early_stop_patience=early_stop_patience,
        eval_output_path=eval_output_path,
        eval_batch_size=eval_batch_size,
        sic_min_bkg_events=sic_min_bkg_events,
        debug=debug,
    )
    return classifier
