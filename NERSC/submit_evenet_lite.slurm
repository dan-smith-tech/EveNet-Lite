#!/bin/bash -l
#SBATCH --job-name=evenet-lite-ddp
#SBATCH --time=00:10:00
#SBATCH -C gpu
#SBATCH --account=nstaff
#SBATCH -q debug
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=32
#SBATCH --image=avencast/event:1.3

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)

# Update these paths (or glob patterns) to point at your data inside the container or shared filesystem.
# Example: TRAIN_SIG="/global/cfs/cdirs/.../data/NMSSM_*300*/evenet/train/*.pt"
TRAIN_SIG="/path/to/train_signal.pt"
TRAIN_BKG="/path/to/train_background.pt"
VAL_SIG="/path/to/valid_signal.pt"
VAL_BKG="/path/to/valid_background.pt"
CHECKPOINT_DIR="/path/to/checkpoints"

cmd="python ${SCRIPT_DIR}/train_multi_gpu.py \\
  --train-sig \"${TRAIN_SIG}\" \\
  --train-bkg \"${TRAIN_BKG}\" \\
  --val-sig \"${VAL_SIG}\" \\
  --val-bkg \"${VAL_BKG}\" \\
  --checkpoint-path ${CHECKPOINT_DIR} \\
  --sampler weighted \\
  --epochs 3 \\
  --batch-size 512 \\
  --checkpoint-every 1 \\
  --save-top-k 2 \\
  --debug"

set -x
srun -l shifter \
  bash -c "source ${SCRIPT_DIR}/export_DDP_vars.sh && ${cmd}"
