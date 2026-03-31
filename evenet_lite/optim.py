import logging
import math
from dataclasses import dataclass
from typing import Any, Callable, Iterable, List, Sequence, Tuple
import torch

DEFAULT_HEAD_LR = 1e-3
DEFAULT_WEIGHT_DECAY = 1e-2
DEFAULT_LR_GROUPS: List[float] = [1e-3, 3e-4, 1e-4]
DEFAULT_WEIGHT_DECAY_GROUPS: List[float] = [DEFAULT_WEIGHT_DECAY] * len(DEFAULT_LR_GROUPS)
DEFAULT_MODULE_GROUPS: List[List[str]] = [
    ["Classification"],
    ["ObjectEncoder"],
    ["PET", "GlobalEmbedding"],
]


def unwrap_model(model: torch.nn.Module):
    return model.module if hasattr(model, "module") else model

def _get_by_path(root: torch.nn.Module, path: str) -> torch.nn.Module | None:
    cur: Any = root
    for part in path.split("."):
        if not hasattr(cur, part):
            return None
        cur = getattr(cur, part)
    return cur if isinstance(cur, torch.nn.Module) else None

def set_peft_trainable(model: torch.nn.Module, train_layernorm: bool = True):
    root = unwrap_model(model)
    if getattr(root, "ensemble_mode", None) == "independent":
        for model_element in root.models:
            m = unwrap_model(model_element)

            # 1) freeze all
            for p in m.backbone.parameters():
                p.requires_grad_(False)

            # 2) unfreeze adapters + head
            for name, p in m.backbone.named_parameters():
                if ("adapters" in name) or ("GlobalEmbedding" in name) or ("ObjectEncoder" in name):
                    p.requires_grad_(True)

            # 3) optional: unfreeze LayerNorm
            if train_layernorm:
                for mod in m.backbone.modules():
                    if isinstance(mod, torch.nn.LayerNorm):
                        for p in mod.parameters():
                            p.requires_grad_(True)

            # 4) explicitly keep MoE params frozen (gate router and expert FFNs
            #    are pre-trained weights that must not be updated during adapter
            #    fine-tuning, even if LayerNorms were unselectively unfrozen above)
            for name, p in m.backbone.named_parameters():
                if any(seg in name for seg in ("mlp.gate", "mlp.routed_experts", "mlp.shared_experts")):
                    p.requires_grad_(False)

    else:
        m = root

        # 1) freeze all
        for p in m.backbone.parameters():
            p.requires_grad_(False)

        # 2) unfreeze adapters + head
        for name, p in m.backbone.named_parameters():
            if ("adapters" in name) or ("GlobalEmbedding" in name) or ("ObjectEncoder" in name):
                p.requires_grad_(True)

        # 3) optional: unfreeze LayerNorm
        if train_layernorm:
            for mod in m.backbone.modules():
                if isinstance(mod, torch.nn.LayerNorm):
                    for p in mod.parameters():
                        p.requires_grad_(True)

        # 4) explicitly keep MoE params frozen (gate router and expert FFNs
        #    are pre-trained weights that must not be updated during adapter
        #    fine-tuning, even if LayerNorms were unselectively unfrozen above)
        for name, p in m.backbone.named_parameters():
            if any(seg in name for seg in ("mlp.gate", "mlp.routed_experts", "mlp.shared_experts")):
                p.requires_grad_(False)


def print_trainable(model):
    m = unwrap_model(model)
    total = 0
    for n, p in m.named_parameters():
        if p.requires_grad:
            total += p.numel()
    print(f"Trainable params: {total:,}")

def _collect_parameters(model: torch.nn.Module, module_names: Iterable[str]) -> List[torch.nn.Parameter]:
    params: List[torch.nn.Parameter] = []
    seen: set[int] = set()
    for name in module_names:
        base_model = unwrap_model(model)
        modules_to_collect: List[torch.nn.Module] = []
        direct_module = _get_by_path(base_model, name) or getattr(base_model, name, None) # support nested paths
        # direct_module = getattr(base_model, name, None)
        if direct_module is not None:
            modules_to_collect.append(direct_module)
        elif hasattr(base_model, "backbone"):
            backbone_module = _get_by_path(base_model, "backbone") or getattr(getattr(base_model, "backbone"), name, None)
            # backbone_module = getattr(getattr(base_model, "backbone"), name, None)
            if backbone_module is not None:
                modules_to_collect.append(backbone_module)
        if hasattr(base_model, "models"):
            for member in getattr(base_model, "models"):
                submodule = getattr(member, name, None)
                if submodule is None and hasattr(member, "backbone"):
                    submodule = getattr(member.backbone, name, None)
                if submodule is not None:
                    modules_to_collect.append(submodule)
        if not modules_to_collect:
            logging.warning("Requested module '%s' not found on model; skipping.", name)
            continue
        for module in modules_to_collect:
            for param in module.parameters():
                if (not param.requires_grad):
                    continue
                if id(param) not in seen:
                    params.append(param)
                    seen.add(id(param))
    return params


def _scale_for_distributed(value: float, world_size: int) -> float:
    if world_size <= 1:
        return value
    return value * (world_size ** 0.5)


def _as_list(value: Any) -> List[Any]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


def _normalize_hparam(
        values: Sequence[Any],
        target_len: int,
        name: str,
) -> List[Any]:
    if len(values) == target_len:
        return list(values)
    if len(values) == 1:
        return list(values) * target_len
    raise ValueError(f"{name} length {len(values)} does not match module list length {target_len}")


@dataclass
class OptimizerGroupConfig:
    tag: str
    modules: List[str]
    lr: float
    weight_decay: float


def resolve_optimizer_groups(config: Any, world_size: int = 1) -> List[OptimizerGroupConfig]:
    module_groups: List[List[str]] = config.module_lists or list(DEFAULT_MODULE_GROUPS)
    lrs = _as_list(config.lr) or list(DEFAULT_LR_GROUPS)
    weight_decays = _as_list(config.weight_decay) or list(DEFAULT_WEIGHT_DECAY_GROUPS)

    module_groups = [list(group) for group in module_groups]
    lrs = _normalize_hparam(lrs, len(module_groups), "lr")
    weight_decays = _normalize_hparam(weight_decays, len(module_groups), "weight_decay")

    lrs = [_scale_for_distributed(lr, world_size) for lr in lrs]
    weight_decays = [_scale_for_distributed(wd, world_size) for wd in weight_decays]

    return [
        OptimizerGroupConfig(
            tag=f"group_{idx}",
            modules=modules,
            lr=lr,
            weight_decay=wd,
        )
        for idx, (modules, lr, wd) in enumerate(zip(module_groups, lrs, weight_decays))
    ]


def default_optimizer_builder(
        config: Any, world_size: int,
) -> Callable[[Iterable[torch.nn.Parameter], float, float, str], torch.optim.Optimizer]:
    def _builder(
            params: Iterable[torch.nn.Parameter],
            lr: float,
            weight_decay: float,
            tag: str = "group",
    ) -> torch.optim.Optimizer:
        return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)

    return _builder


def _compute_warmup_epochs(config: Any, epochs: int) -> int:
    if config.warmup_epochs is not None:
        return max(0, min(epochs, int(config.warmup_epochs)))
    return max(0, min(epochs, int(epochs * config.warmup_ratio)))


def default_scheduler_builder(
        config: Any, epochs: int, steps_per_epoch: int | None = None,
) -> Callable[[torch.optim.Optimizer, str], Any]:
    warmup_epochs = _compute_warmup_epochs(config, epochs)
    cosine_epochs = max(1, epochs - warmup_epochs)

    warmup_iters = warmup_epochs * steps_per_epoch if steps_per_epoch else warmup_epochs
    cosine_iters = max(1, cosine_epochs * steps_per_epoch) if steps_per_epoch else max(1, cosine_epochs)

    def _builder(optimizer: torch.optim.Optimizer, tag: str = "group") -> Any:
        base_lrs = [group["lr"] for group in optimizer.param_groups]
        min_lr = config.min_lr

        def _make_lambda(base_lr: float) -> Callable[[int], float]:
            min_lr_ratio = min_lr / base_lr if base_lr > 0 else 0.0
            min_lr_ratio = min(min_lr_ratio, 1.0)

            def lr_lambda(step: int) -> float:
                if warmup_iters + cosine_iters <= 0:
                    return 1.0

                step_clamped = min(step, warmup_iters + cosine_iters)
                if warmup_iters > 0 and step_clamped < warmup_iters:
                    return config.warmup_start_factor + (
                            (1 - config.warmup_start_factor)
                            * (step_clamped / max(1, warmup_iters))
                    )

                progress = (step_clamped - warmup_iters) / max(1, cosine_iters)
                cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
                return min_lr_ratio + (1 - min_lr_ratio) * cosine_decay

            return lr_lambda

        lambda_fns = [_make_lambda(base_lr) for base_lr in base_lrs]
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_fns)

    return _builder


def build_optimizers_and_schedulers(
        model: torch.nn.Module,
        config: Any,
        epochs: int,
        world_size: int = 1,
        steps_per_epoch: int | None = None,
) -> Tuple[List[torch.optim.Optimizer], List[Any], List[str]]:
    group_configs = resolve_optimizer_groups(config, world_size=world_size)
    optimizer_builder = config.optimizer_fn or default_optimizer_builder(config, world_size)
    scheduler_builder = config.scheduler_fn or default_scheduler_builder(
        config, epochs, steps_per_epoch
    )

    optimizers: List[torch.optim.Optimizer] = []
    schedulers: List[Any] = []
    tags: List[str] = []

    for group in group_configs:
        params = _collect_parameters(model, group.modules)
        if not params:
            logging.warning("No parameters collected for optimizer group '%s'", group.tag)
            continue

        optimizer: torch.optim.Optimizer
        for args in [
            (params, group.lr, group.weight_decay, group.tag),
            (params, group.lr, group.weight_decay),
            (params, group.tag),
            (params,),
        ]:
            try:
                optimizer = optimizer_builder(*args)  # type: ignore[misc]
                break
            except TypeError:
                continue
        optimizers.append(optimizer)

        scheduler: Any = None
        for args in [
            (optimizer, group.tag),
            (optimizer, epochs, group.tag),
            (optimizer, epochs),
            (optimizer,),
        ]:
            try:
                scheduler = scheduler_builder(*args)  # type: ignore[misc]
                break
            except TypeError:
                continue
        schedulers.append(scheduler)
        tags.append(group.tag)

    return optimizers, schedulers, tags
