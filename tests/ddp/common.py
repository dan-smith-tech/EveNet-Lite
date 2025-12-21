"""Shared utilities for DDP sanity tests."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Callable

import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def configure_logging(name: str, rank: int) -> logging.Logger:
    """Create a logger dedicated to the current test file.

    Each test module should call this once per spawned worker to ensure logs
    include the rank and the module name to help differentiate output coming
    from multiple processes.
    """

    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        fmt="[%(asctime)s][%(name)s][rank=%(rank)s][%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    handler.setFormatter(formatter)

    class RankFilter(logging.Filter):
        def filter(self, record: logging.LogRecord) -> bool:  # type: ignore[override]
            record.rank = rank
            return True

    handler.addFilter(RankFilter())
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    return logger


@dataclass
class DDPEnv:
    """Environment variables required for torch.distributed."""

    rank: int
    world_size: int
    local_rank: int
    master_addr: str = "127.0.0.1"
    master_port: str = "29500"

    def apply(self) -> None:
        os.environ.setdefault("MASTER_ADDR", self.master_addr)
        os.environ.setdefault("MASTER_PORT", self.master_port)
        os.environ["RANK"] = str(self.rank)
        os.environ["WORLD_RANK"] = str(self.rank)
        os.environ["LOCAL_RANK"] = str(self.local_rank)
        os.environ["WORLD_SIZE"] = str(self.world_size)


def init_process_group(env: DDPEnv) -> None:
    env.apply()
    if not dist.is_initialized():
        dist.init_process_group(backend="gloo", rank=env.rank, world_size=env.world_size)


def cleanup_process_group() -> None:
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


def spawn(world_size: int, target: Callable, *args) -> None:
    mp.spawn(target, args=(world_size, *args), nprocs=world_size, join=True)


def all_equal_across_ranks(tensor: torch.Tensor) -> bool:
    gathered = [torch.zeros_like(tensor) for _ in range(dist.get_world_size())]
    dist.all_gather(gathered, tensor)
    return all(torch.allclose(gathered[0], other) for other in gathered[1:])


def build_synthetic_dataset(num_samples: int = 8):
    from evenet_lite.data import EvenetTensorDataset

    features = {"globals": torch.randn(num_samples, 2)}
    labels = torch.tensor([0, 1] * (num_samples // 2))
    return EvenetTensorDataset(features, labels)


class TinyClassifier(torch.nn.Module):
    """Minimal classifier that operates on the `globals` feature group."""

    num_classes = 2

    def __init__(self) -> None:
        super().__init__()
        self.layer = torch.nn.Linear(2, self.num_classes)

    def forward(self, globals: torch.Tensor, **_: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self.layer(globals)
