from __future__ import annotations

import logging
from typing import Any, Callable, Iterable, List, Tuple

import torch

DEFAULT_HEAD_LR = 1e-3
DEFAULT_WEIGHT_DECAY = 1e-2
BODY_LR_SCALE = 0.1
DEFAULT_BODY_MODULES: List[str] = ["GlobalEmbedding", "PET", "ObjectEncoder"]
DEFAULT_HEAD_MODULES: List[str] = ["Classification"]


def _collect_parameters(model: torch.nn.Module, module_names: Iterable[str]) -> List[torch.nn.Parameter]:
    params: List[torch.nn.Parameter] = []
    seen: set[int] = set()
    for name in module_names:
        module = getattr(model, name, None)
        if module is None:
            logging.warning("Requested module '%s' not found on model; skipping.", name)
            continue
        for param in module.parameters():
            if id(param) not in seen:
                params.append(param)
                seen.add(id(param))
    return params


def resolve_module_groups(config: Any) -> Tuple[List[str], List[str]]:
    body_modules = config.body_modules or list(DEFAULT_BODY_MODULES)
    head_modules = config.head_modules or list(DEFAULT_HEAD_MODULES)
    return body_modules, head_modules


def _scale_for_distributed(value: float, world_size: int) -> float:
    if world_size <= 1:
        return value
    return value * (world_size ** 0.5)


def resolve_group_lr(config: Any, tag: str, world_size: int = 1) -> float:
    head_lr = config.head_lr if config.head_lr is not None else config.lr or DEFAULT_HEAD_LR
    body_lr = config.body_lr if config.body_lr is not None else head_lr * BODY_LR_SCALE
    head_lr = _scale_for_distributed(head_lr, world_size)
    body_lr = _scale_for_distributed(body_lr, world_size)
    if tag == "body":
        return body_lr
    if tag == "head":
        return head_lr
    return head_lr


def resolve_group_weight_decay(config: Any, tag: str, world_size: int = 1) -> float:
    default_wd = config.weight_decay if config.weight_decay is not None else DEFAULT_WEIGHT_DECAY
    head_wd = config.head_weight_decay if config.head_weight_decay is not None else default_wd
    body_wd = config.body_weight_decay if config.body_weight_decay is not None else default_wd
    head_wd = _scale_for_distributed(head_wd, world_size)
    body_wd = _scale_for_distributed(body_wd, world_size)
    return body_wd if tag == "body" else head_wd


def default_optimizer_builder(
    config: Any, world_size: int
) -> Callable[[Iterable[torch.nn.Parameter], str], torch.optim.Optimizer]:
    def _builder(params: Iterable[torch.nn.Parameter], tag: str = "head") -> torch.optim.Optimizer:
        lr = resolve_group_lr(config, tag, world_size=world_size)
        weight_decay = resolve_group_weight_decay(config, tag, world_size=world_size)
        return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)

    return _builder


def _compute_warmup_epochs(config: Any, epochs: int) -> int:
    if config.warmup_epochs is not None:
        return max(0, min(epochs, int(config.warmup_epochs)))
    return max(0, min(epochs, int(epochs * config.warmup_ratio)))


def default_scheduler_builder(config: Any, epochs: int) -> Callable[[torch.optim.Optimizer, str], Any]:
    warmup_epochs = _compute_warmup_epochs(config, epochs)
    cosine_epochs = max(1, epochs - warmup_epochs)

    def _builder(optimizer: torch.optim.Optimizer, tag: str = "head") -> Any:
        cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=cosine_epochs, eta_min=config.min_lr
        )
        if warmup_epochs <= 0:
            return cosine

        warmup = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=config.warmup_start_factor,
            total_iters=warmup_epochs,
        )
        return torch.optim.lr_scheduler.SequentialLR(
            optimizer, schedulers=[warmup, cosine], milestones=[warmup_epochs]
        )

    return _builder


def build_optimizers_and_schedulers(
    model: torch.nn.Module, config: Any, epochs: int, world_size: int = 1
) -> Tuple[List[torch.optim.Optimizer], List[Any]]:
    body_modules, head_modules = resolve_module_groups(config)
    group_configs = [("body", body_modules), ("head", head_modules)]

    optimizer_builder = config.optimizer_fn or default_optimizer_builder(config, world_size)
    scheduler_builder = config.scheduler_fn or default_scheduler_builder(config, epochs)

    optimizers: List[torch.optim.Optimizer] = []
    schedulers: List[Any] = []

    for tag, modules in group_configs:
        params = _collect_parameters(model, modules)
        if not params:
            logging.warning("No parameters collected for optimizer group '%s'", tag)
            continue

        optimizer: torch.optim.Optimizer
        for args in [(params, tag), (params,)]:
            try:
                optimizer = optimizer_builder(*args)  # type: ignore[misc]
                break
            except TypeError:
                continue
        optimizers.append(optimizer)

        scheduler: Any = None
        for args in [
            (optimizer, tag),
            (optimizer, epochs, tag),
            (optimizer, epochs),
            (optimizer,),
        ]:
            try:
                scheduler = scheduler_builder(*args)  # type: ignore[misc]
                break
            except TypeError:
                continue
        schedulers.append(scheduler)

    return optimizers, schedulers
