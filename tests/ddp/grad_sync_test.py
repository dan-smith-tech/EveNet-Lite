"""DDP gradient synchronization sanity test."""

from __future__ import annotations

import torch
from torch.nn.parallel import DistributedDataParallel as DDP

from .common import DDPEnv, all_equal_across_ranks, cleanup_process_group, configure_logging, init_process_group, spawn


def _worker(rank: int, world_size: int) -> None:
    logger = configure_logging(__name__, rank)
    env = DDPEnv(rank=rank, local_rank=rank, world_size=world_size)
    init_process_group(env)
    torch.manual_seed(42)

    model = torch.nn.Linear(4, 2)
    device = torch.device("cpu")
    model.to(device)
    ddp_model = DDP(model)

    data = torch.randn(4, 4, device=device)
    targets = torch.tensor([0, 1, 0, 1], device=device)
    logger.info("Running forward pass", extra={"rank": rank})
    outputs = ddp_model(data)
    loss = torch.nn.functional.cross_entropy(outputs, targets)
    logger.info("Backward pass with loss=%f", loss.item(), extra={"rank": rank})
    loss.backward()

    grads_synced = all(all_equal_across_ranks(p.grad) for p in ddp_model.parameters())
    logger.info("Gradient synchronization status: %s", "ok" if grads_synced else "failed", extra={"rank": rank})

    cleanup_process_group()


def run(world_size: int) -> None:
    """Launch the gradient synchronization test across ``world_size`` ranks."""

    spawn(world_size, _worker)
