"""Trainer metric reduction sanity test."""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional

import torch
from torch.utils.data import DataLoader, DistributedSampler

from evenet_lite.callbacks import NormalizationCallback
from evenet_lite.trainer import Trainer, TrainerConfig

from .common import DDPEnv, cleanup_process_group, configure_logging, init_process_group, spawn, TinyClassifier, build_synthetic_dataset


def _worker(rank: int, world_size: int) -> None:
    logger = configure_logging(__name__, rank)
    env = DDPEnv(rank=rank, local_rank=rank, world_size=world_size)
    init_process_group(env)
    torch.manual_seed(1234 + rank)

    feature_names: Dict[str, Iterable[str]] = {"globals": ["f0", "f1"]}
    callbacks = [NormalizationCallback(normalization_rules={"globals": {"f0": "none", "f1": "none"}})]
    config = TrainerConfig(
        device="cpu",
        compute_physics_metrics=False,
        use_wandb=False,
        num_workers=0,
        grad_clip=None,
    )
    trainer = Trainer(TinyClassifier(), feature_names, config, callbacks=callbacks, class_labels=["a", "b"])
    dataset = build_synthetic_dataset(num_samples=8)
    sampler = DistributedSampler(dataset, shuffle=False) if world_size > 1 else None
    loader = DataLoader(dataset, batch_size=2, sampler=sampler, shuffle=sampler is None, num_workers=0)
    logger.info("Starting metric reduction training step", extra={"rank": rank})

    for cb in trainer.callbacks:
        cb.on_train_start(trainer)

    trainer.model = trainer._maybe_wrap_ddp()
    trainer._setup_optimizers_and_schedulers(epochs=1, steps_per_epoch=max(1, len(loader)))
    metrics = trainer._run_epoch(trainer.model, loader, epoch=0, training=True)
    logger.info("Local metrics computed: %s", metrics, extra={"rank": rank})

    gathered: List[Optional[Dict[str, float]]] = [None for _ in range(world_size)]
    torch.distributed.all_gather_object(gathered, metrics)
    if rank == 0:
        consistent = all(gathered[0] == other for other in gathered[1:])
        logger.info(
            "Metric reduction consistency across ranks: %s | gathered=%s",
            "ok" if consistent else "failed",
            gathered,
            extra={"rank": rank},
        )

    cleanup_process_group()


def run(world_size: int) -> None:
    """Launch the metric reduction test across ``world_size`` ranks."""

    spawn(world_size, _worker)
