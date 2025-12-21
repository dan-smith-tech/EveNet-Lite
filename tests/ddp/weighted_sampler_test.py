"""DistributedWeightedSampler bias sanity test."""



from typing import List

import torch
import torch.distributed as dist

from evenet_lite.data import DistributedWeightedSampler

from .common import DDPEnv, cleanup_process_group, configure_logging, init_process_group, spawn


def _worker(rank: int, world_size: int) -> None:
    logger = configure_logging(__name__, rank)
    env = DDPEnv(rank=rank, local_rank=rank, world_size=world_size)
    init_process_group(env)
    torch.manual_seed(7 + rank)

    weights = torch.tensor([0.05, 0.05, 0.9, 0.9, 0.9, 0.9])
    sampler = DistributedWeightedSampler(weights, epoch_size=60)
    sampler.set_epoch(0)
    sampled_indices = list(iter(sampler))
    logger.info("Sampled indices for rank: %s", sampled_indices, extra={"rank": rank})

    gathered: List[List[int]] = [None for _ in range(world_size)]  # type: ignore[list-item]
    dist.all_gather_object(gathered, sampled_indices)
    if rank == 0:
        combined = [idx for replica in gathered for idx in replica]
        counts = torch.bincount(torch.tensor(combined), minlength=len(weights)).float()
        high_weight_mean = counts[2:].mean()
        low_weight_mean = counts[:2].mean()
        bias_ok = high_weight_mean > (low_weight_mean * 2)
        logger.info(
            "DistributedWeightedSampler bias: %s | expected heavier sampling for indices >=2 | counts=%s",
            "ok" if bias_ok else "failed",
            counts.tolist(),
            extra={"rank": rank},
        )
        assert bias_ok, (
            "DistributedWeightedSampler should draw high-weighted samples at least twice as often; "
            f"observed counts={counts.tolist()}"
        )

    cleanup_process_group()


def run(world_size: int) -> None:
    """Launch the weighted sampler bias test across ``world_size`` ranks."""

    spawn(world_size, _worker)
