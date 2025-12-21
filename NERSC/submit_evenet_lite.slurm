#!/bin/bash -l
#SBATCH --job-name=evenet-lite-ddp
#SBATCH --time=00:10:00
#SBATCH -C gpu
#SBATCH --account=m2616_g
#SBATCH -q debug
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=32
#SBATCH --image=registry.nersc.gov/m2616/avencast/evenet:1.3

set -euo pipefail

export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)

# Update these paths (or glob patterns) to point at your data inside the container or shared filesystem.
# Example: TRAIN_SIG="/global/cfs/cdirs/.../data/NMSSM_*300*/evenet/train/*.pt"
TRAIN_SIG="../data/46915_NMSSM_XToYHTo2B2Tau_MX-600_MY-300_TuneCP5_13TeV-madgraph-pythia8_RunIISummer20UL16NanoAODv9-106X_mcRun2_asymptotic_v17-v1_NANOAODSIM/evenet/train/skim_32F37BC4-BDDF-BF4F-91ED-3CF4F049586E.pt"
TRAIN_BKG="../data/67993_TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8_RunIISummer20UL16NanoAODv9-106X_mcRun2_asymptotic_v17-v1_NANOAODSIM/evenet/train/skim_0AFFFA0B-438F-0B4E-A74E-344E41B69B2D.pt"
VAL_SIG="../data/46915_NMSSM_XToYHTo2B2Tau_MX-600_MY-300_TuneCP5_13TeV-madgraph-pythia8_RunIISummer20UL16NanoAODv9-106X_mcRun2_asymptotic_v17-v1_NANOAODSIM/evenet/valid/skim_32F37BC4-BDDF-BF4F-91ED-3CF4F049586E.pt"
VAL_BKG="../data/67993_TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8_RunIISummer20UL16NanoAODv9-106X_mcRun2_asymptotic_v17-v1_NANOAODSIM/evenet/valid/skim_0AFFFA0B-438F-0B4E-A74E-344E41B69B2D.pt"
CHECKPOINT_DIR="/path/to/checkpoints"

cmd="python train_multi_gpu.py \\
  --train-sig \"${TRAIN_SIG}\" \\
  --train-bkg \"${TRAIN_BKG}\" \\
  --val-sig \"${VAL_SIG}\" \\
  --val-bkg \"${VAL_BKG}\" \\
  --sampler weighted \\
  --epochs 3 \\
  --batch-size 512 \\
  --debug"

set -x
srun -l shifter \
  bash -c "source export_DDP_vars.sh && ${cmd}"
