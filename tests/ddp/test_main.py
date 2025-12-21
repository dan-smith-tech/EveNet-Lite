"""DDP and sampler sanity tests for EveNet-Lite.

These lightweight tests are intended for environments such as NERSC where
Distributed Data Parallel (DDP) jobs are launched via Slurm. Each test uses a
small synthetic dataset and runs only a handful of steps so it can be executed
as part of a launch sanity check before large jobs are submitted.

Example Slurm environment variables (from the NERSC tutorial)::

    export RANK=$SLURM_PROCID
    export WORLD_RANK=$SLURM_PROCID
    export LOCAL_RANK=$SLURM_LOCALID
    export WORLD_SIZE=$SLURM_NTASKS
    export MASTER_PORT=29500

Usage (CPU example)::

    python -m tests.ddp.test_main --test grad --world_size 2
"""



import argparse
from typing import Callable, Dict

from . import grad_sync_test, metric_reduction_test, sampler_coverage_test, weighted_sampler_test


def main() -> None:
    parser = argparse.ArgumentParser(description="DDP and sampler sanity tests")
    parser.add_argument(
        "--test",
        choices=["grad", "metrics", "sampler", "weighted_sampler"],
        required=True,
        help="Which test to run",
    )
    parser.add_argument("--world_size", type=int, default=2, help="Number of processes to launch")
    args = parser.parse_args()

    tests: Dict[str, Callable[[int], None]] = {
        "grad": grad_sync_test.run,
        "metrics": metric_reduction_test.run,
        "sampler": sampler_coverage_test.run,
        "weighted_sampler": weighted_sampler_test.run,
    }

    tests[args.test](args.world_size)


if __name__ == "__main__":
    main()
