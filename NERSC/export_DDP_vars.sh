#!/bin/bash
# Set the environment variables expected by torch.distributed.launch / torchrun
# when running under SLURM. Source this script *within* the srun context.

export RANK=${SLURM_PROCID}
export WORLD_RANK=${SLURM_PROCID}
export LOCAL_RANK=${SLURM_LOCALID}
export WORLD_SIZE=${SLURM_NTASKS}
export MASTER_PORT=${MASTER_PORT:-29500}
