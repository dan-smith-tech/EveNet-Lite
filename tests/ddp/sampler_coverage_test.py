"""DistributedSampler coverage sanity test."""



from typing import List

from torch.utils.data import DistributedSampler
import torch.distributed as dist

from .common import DDPEnv, cleanup_process_group, configure_logging, init_process_group, spawn, build_synthetic_dataset


def _worker(rank: int, world_size: int) -> None:
    logger = configure_logging(__name__, rank)
    env = DDPEnv(rank=rank, local_rank=rank, world_size=world_size)
    init_process_group(env)

    dataset = build_synthetic_dataset(num_samples=8)
    sampler = DistributedSampler(dataset, shuffle=False)
    indices = list(iter(sampler))
    logger.info("Sampler indices for rank: %s", indices, extra={"rank": rank})

    gathered: List[List[int]] = [None for _ in range(world_size)]  # type: ignore[list-item]
    dist.all_gather_object(gathered, indices)
    if rank == 0:
        flattened = [idx for replica in gathered for idx in replica]
        expected = list(range(len(dataset)))
        coverage_complete = sorted(flattened) == expected
        replicas_distinct = len(flattened) == len(set(flattened))
        logger.info(
            "DistributedSampler coverage: %s | expected global indices=%s | per-rank indices=%s",
            "ok" if coverage_complete and replicas_distinct else "failed",
            expected,
            gathered,
            extra={"rank": rank},
        )
        assert coverage_complete and replicas_distinct, (
            "DistributedSampler should cover each dataset element exactly once across ranks; "
            f"expected {expected}, got {gathered}"
        )

    cleanup_process_group()


def run(world_size: int) -> None:
    """Launch the sampler coverage test across ``world_size`` ranks."""

    spawn(world_size, _worker)
