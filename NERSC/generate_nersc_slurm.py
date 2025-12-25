#!/usr/bin/env python3
"""Generate a NERSC-friendly Slurm script for multi-mass Evenet-Lite runs.

This script inspects a data directory for signal mass points (``MX``/``MY``) and
builds a Slurm array job that trains one model per mass point using
``train_multi_gpu.py``. Background datasets are configurable via CLI and are
passed to all mass points.
"""
import argparse
import re
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

# SIGNAL_PATTERN = re.compile(r"^(?P<dsid>\d+)_NMSSM_.*_MX-(?P<mx>\d+)_MY-(?P<my>\d+)_.*")
SIGNAL_PATTERN = re.compile(r"MX-(?P<mx>\d+)_MY-(?P<my>\d+)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("/global/cfs/cdirs/m5019/tihsu"),
        help="Directory containing signal/background subdirectories",
    )
    parser.add_argument(
        "--only-masses",
        nargs="+",
        default=None,
        help="Only run these mass labels, e.g. MX300_MY50 MX500_MY100",
    )
    parser.add_argument(
        "--background",
        dest="backgrounds",
        action="append",
        default=[
            "ggHtautau", "VBFHtautau", "DYBJets_pt100to200", "DYBJets_pt200toInf", "tt1l"
        ],
        help="Background directory name (relative to data root). Provide multiple times to include more.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("submit_evenet_lite_multi.slurm"),
        help="Path to write the generated Slurm script",
    )
    parser.add_argument("--job-name", default="evenet-lite-grid", help="Slurm job name")
    parser.add_argument("--time", default="04:00:00", help="Walltime for each array element")
    parser.add_argument("--account", default="m2616_g", help="NERSC account")
    parser.add_argument("--queue", default="regular", help="Slurm queue/partition")
    parser.add_argument("--nodes", type=int, default=1, help="Nodes per mass-point job")
    parser.add_argument(
        "--gpus-per-node",
        type=int,
        default=4,
        help="GPUs per node to request (Perlmutter GPU nodes have 4)",
    )
    parser.add_argument(
        "--ntasks-per-node",
        type=int,
        help="MPI tasks per node. Defaults to --gpus-per-node if omitted",
    )
    parser.add_argument("--cpus-per-task", type=int, default=32, help="CPUs per task")
    parser.add_argument(
        "--image",
        default="registry.nersc.gov/m2616/avencast/evenet:1.3",
        help="Container image",
    )
    parser.add_argument(
        "--checkpoint-root",
        type=Path,
        default=Path("./checkpoints/"),
        help="Base directory where checkpoints will be stored per mass point",
    )
    parser.add_argument("--epochs", type=int, default=10, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=2048, help="Training batch size")
    parser.add_argument(
        "--sampler",
        choices=["weighted", "none"],
        default="weighted",
        help="Sampler strategy passed to train_multi_gpu.py",
    )
    parser.add_argument(
        "--pretrained-path",
        default="/global/cfs/cdirs/m5019/avencast/Checkpoints/checkpoints.20M.ablation.4.newcls/epoch=29_train=2.4809_val=2.5040.ckpt",
        help="Checkpoint used for --pretrained runs",
    )
    parser.add_argument(
        "--extra-args",
        default="",
        help="Additional CLI arguments forwarded to train_multi_gpu.py",
    )
    parser.add_argument(
        "--sequential",
        action="store_true",
        help="Generate a script that loops over mass points sequentially (no Slurm array)",
    )
    return parser.parse_args()


def find_signal_datasets(data_root: Path) -> List[Tuple[str, str, str]]:
    """Return a list of (label, train_glob, valid_glob) tuples for signals."""
    signals: List[Tuple[str, str, str]] = []
    for child in sorted(data_root.iterdir()):
        if not child.is_dir():
            continue
        match = SIGNAL_PATTERN.match(child.name)
        if not match:
            continue
        label = f"MX{match.group('mx')}_MY{match.group('my')}"
        train_glob = str(child / "evenet" / "train" / "*.pt")
        valid_glob = str(child / "evenet" / "valid" / "*.pt")
        signals.append((label, train_glob, valid_glob))
    if not signals:
        raise SystemExit(f"No signal datasets found in {data_root}")
    return signals


def build_background_glob(data_root: Path, backgrounds: Sequence[str], split: str) -> str:
    brace = "{" + ",".join(backgrounds) + "}"
    return str(data_root / brace / "evenet" / split / "*.pt")


def format_bash_array(values: Iterable[str]) -> str:
    escaped = [v.replace("\"", "\\\"") for v in values]
    body = "\n  \"" + "\"\n  \"".join(escaped) + "\"\n"
    return f"( {body})"


def write_slurm_script(args: argparse.Namespace, signals: List[Tuple[str, str, str]]) -> None:
    ntasks_per_node = args.ntasks_per_node or args.gpus_per_node
    background_train = build_background_glob(args.data_root, args.backgrounds, "train")
    background_valid = build_background_glob(args.data_root, args.backgrounds, "valid")

    labels, train_globs, valid_globs = zip(*signals)

    script_dir = Path(__file__).resolve().parent  # .../EveNet-Lite/NERSC
    repo_root = script_dir.parent  # .../EveNet-Lite

    array_directive = "" if args.sequential else f"#SBATCH --array=0-{len(signals) - 1}"
    extra_args = args.extra_args.strip()
    command_parts = [
        "python train_multi_gpu.py",
        "--train-sig \\\"${TRAIN_SIG}\\\"",
        "--train-bkg \\\"${BACKGROUND_TRAIN_PATTERN}\\\"",
        "--val-sig \\\"${VAL_SIG}\\\"",
        "--val-bkg \\\"${BACKGROUND_VALID_PATTERN}\\\"",
        "--eval-sig \\\"${VAL_SIG}\\\"",
        "--eval-bkg \\\"${BACKGROUND_VALID_PATTERN}\\\"",
        "--eval-output \\\"${EVAL_OUTPUT}\\\"",
        "--checkpoint-path \\\"${OUTPUT_DIR}\\\"",
        f"--sampler {args.sampler}",
        f"--epochs {args.epochs}",
        f"--batch-size {args.batch_size}",
        "--pretrained",
        f"--pretrained-path \\\"{args.pretrained_path}\\\"",
        "--pretrained-source local",
        "--wandb-name \\\"${MASS_POINT}\\\"",
        "--lr 1e-4 5e-5 1e-5",

    ]
    if extra_args:
        command_parts.append(extra_args)

    command_template = " \\\n  ".join(command_parts)

    script = f"""#!/bin/bash -l
#SBATCH --job-name={args.job_name}
#SBATCH --time={args.time}
#SBATCH -C gpu
#SBATCH --account={args.account}
#SBATCH -q {args.queue}
#SBATCH --nodes={args.nodes}
#SBATCH --ntasks-per-node={ntasks_per_node}
#SBATCH --gpus-per-node={args.gpus_per_node}
#SBATCH --cpus-per-task={args.cpus_per_task}
#SBATCH --image={args.image}
{array_directive}

set -euo pipefail

export PYTHONPATH="{repo_root}:${{PYTHONPATH:-}}"
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
DATA_ROOT="{args.data_root}"
CHECKPOINT_ROOT="{args.checkpoint_root}"

SIGNAL_LABELS={format_bash_array(labels)}
SIGNAL_TRAIN_PATTERNS={format_bash_array(train_globs)}
SIGNAL_VALID_PATTERNS={format_bash_array(valid_globs)}

BACKGROUND_TRAIN_PATTERN="{background_train}"
BACKGROUND_VALID_PATTERN="{background_valid}"
"""

    if args.sequential:
        script += f"""
set -x
for idx in "${{!SIGNAL_LABELS[@]}}"; do
  MASS_POINT="${{SIGNAL_LABELS[$idx]}}"
  TRAIN_SIG="${{SIGNAL_TRAIN_PATTERNS[$idx]}}"
  VAL_SIG="${{SIGNAL_VALID_PATTERNS[$idx]}}"

  OUTPUT_DIR="${{CHECKPOINT_ROOT}}/${{MASS_POINT}}"
  EVAL_OUTPUT="${{OUTPUT_DIR}}/eval"
  mkdir -p "${{OUTPUT_DIR}}" "${{EVAL_OUTPUT}}"

  cmd="{command_template}"

  srun --nodes={args.nodes} --ntasks-per-node={ntasks_per_node} --gpus-per-node={args.gpus_per_node}  -l shifter \\
    bash -c "source export_DDP_vars.sh && ${{cmd}}"
done
"""
    else:
        script += f"""

MASS_POINT=${{SIGNAL_LABELS[$SLURM_ARRAY_TASK_ID]}}
TRAIN_SIG=${{SIGNAL_TRAIN_PATTERNS[$SLURM_ARRAY_TASK_ID]}}
VAL_SIG=${{SIGNAL_VALID_PATTERNS[$SLURM_ARRAY_TASK_ID]}}

OUTPUT_DIR="${{CHECKPOINT_ROOT}}/${{MASS_POINT}}"
EVAL_OUTPUT="${{OUTPUT_DIR}}/eval"
mkdir -p "${{OUTPUT_DIR}}" "${{EVAL_OUTPUT}}"

cmd="{command_template}"

set -x
srun -l shifter \\
  bash -c "source export_DDP_vars.sh && ${{cmd}}"
"""

    args.output.write_text(script)
    print(f"Wrote Slurm script for {len(signals)} mass points to {args.output}")


def main() -> None:
    args = parse_args()
    signals = find_signal_datasets(args.data_root)
    if args.only_masses is not None:
        keep = set(args.only_masses)
        signals = [s for s in signals if s[0] in keep]
        if not signals:
            raise SystemExit(f"No matching mass points in --only-masses {args.only_masses}")

    write_slurm_script(args, signals)


if __name__ == "__main__":
    main()
