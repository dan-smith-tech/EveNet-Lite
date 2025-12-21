import argparse
import logging
from glob import glob
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch
import yaml

from evenet_lite import run_evenet_lite_training
from evenet_lite.optim import DEFAULT_HEAD_LR, DEFAULT_WEIGHT_DECAY


FeatureBundle = Tuple[Dict[str, torch.Tensor], torch.Tensor]


logger = logging.getLogger(__name__)


def _resolve_paths(path_str: str, description: str) -> List[Path]:
    """Resolve one or more concrete files from a path or glob pattern."""

    candidate = Path(path_str)
    if candidate.is_file():
        return [candidate]

    matches = sorted(Path(p) for p in glob(path_str))
    if not matches:
        raise FileNotFoundError(f"No files matched {description} pattern: {path_str}")

    logger.info("%s pattern matched %d files: %s", description, len(matches), ", ".join(str(m) for m in matches))
    return matches


def _concat_tensors(data_iter: Iterable[torch.Tensor]) -> torch.Tensor:
    return torch.cat(list(data_iter), dim=0)


def _load_split(sig_paths: List[Path], bkg_paths: List[Path]) -> FeatureBundle:
    sig_parts = [torch.load(p) for p in sig_paths]
    bkg_parts = [torch.load(p) for p in bkg_paths]

    features = {
        ("globals" if key == "global" else key): _concat_tensors(
            [*(part[key] for part in sig_parts), *(part[key] for part in bkg_parts)]
        )
        for key in ["x", "x_mask", "global"]
    }

    labels = torch.cat(
        [
            torch.ones(sum(len(part["x"]) for part in sig_parts), device=features["x"].device),
            torch.zeros(sum(len(part["x"]) for part in bkg_parts), device=features["x"].device),
        ],
        dim=0,
    )
    return features, labels


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="DDP-friendly Evenet-Lite runner for NERSC")
    parser.add_argument("--train-sig", type=str, required=True, help="Path or glob pattern to signal training tensor (.pt)")
    parser.add_argument("--train-bkg", type=str, required=True, help="Path or glob pattern to background training tensor (.pt)")
    parser.add_argument("--val-sig", type=str, help="Optional path or glob pattern to signal validation tensor (.pt)")
    parser.add_argument("--val-bkg", type=str, help="Optional path or glob pattern to background validation tensor (.pt)")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--epoch-size", type=int, help="Number of samples per epoch when using a sampler")
    parser.add_argument("--batch-size", type=int, default=512, help="Mini-batch size")
    parser.add_argument("--sampler", choices=["weighted", "none"], default="weighted", help="Sampler strategy")
    parser.add_argument("--checkpoint-path", type=Path, default=Path("checkpoints"), help="Directory for checkpoints")
    parser.add_argument("--resume-from", type=Path, help="Optional checkpoint to resume training")
    parser.add_argument("--checkpoint-every", type=int, default=1, help="Checkpoint frequency in epochs")
    parser.add_argument("--save-top-k", type=int, default=2, help="Number of best checkpoints to keep")
    parser.add_argument("--monitor-metric", type=str, default="val_loss", help="Metric name used to rank checkpoints")
    parser.add_argument(
        "--minimize-metric",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whether the monitor metric should be minimized",
    )
    parser.add_argument("--debug", action="store_true", help="Enable verbose DebugCallback logs")
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Root logger level",
    )

    parser.add_argument(
        "--class-labels",
        nargs="+",
        default=["background", "signal"],
        help="Ordered class labels passed to the classifier",
    )
    parser.add_argument(
        "--feature-names",
        type=Path,
        help="Optional YAML/JSON file providing feature names for each input group",
    )
    parser.add_argument(
        "--normalization-rules",
        type=Path,
        help="Optional YAML/JSON file with normalization rules to pass to the trainer",
    )

    parser.add_argument("--device", type=str, default="auto", help="Device placement for training")
    parser.add_argument("--lr", type=float, default=DEFAULT_HEAD_LR, help="Global learning rate")
    parser.add_argument("--body-lr", type=float, default=1e-4, help="Learning rate for body modules")
    parser.add_argument("--head-lr", type=float, default=1e-3, help="Learning rate for head modules")
    parser.add_argument("--weight-decay", type=float, default=DEFAULT_WEIGHT_DECAY, help="Weight decay for optimizer")
    parser.add_argument("--body-weight-decay", type=float, help="Weight decay for body modules")
    parser.add_argument("--head-weight-decay", type=float, help="Weight decay for head modules")
    parser.add_argument("--grad-clip", type=float, default=1.0, help="Gradient clipping value")
    parser.add_argument("--body-modules", nargs="+", help="Override default body module names")
    parser.add_argument("--head-modules", nargs="+", help="Override default head module names")
    parser.add_argument("--warmup-epochs", type=int, default=1, help="Number of warmup epochs")
    parser.add_argument("--warmup-ratio", type=float, default=0.1, help="Warmup ratio for scheduler")
    parser.add_argument(
        "--warmup-start-factor",
        type=float,
        default=0.1,
        help="Initial LR multiplier applied at the start of warmup",
    )
    parser.add_argument("--min-lr", type=float, default=0.0, help="Minimum learning rate for scheduler")
    parser.add_argument("--global-input-dim", type=int, default=10, help="Number of global features")
    parser.add_argument("--sequential-input-dim", type=int, default=7, help="Number of sequential features per object")
    parser.add_argument(
        "--use-wandb",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable Weights & Biases logging",
    )
    parser.add_argument("--wandb-project", type=str, default="EvenetLite", help="Weights & Biases project name")
    parser.add_argument("--wandb-name", type=str, default="test", help="Weights & Biases run name")
    parser.add_argument(
        "--pretrained",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Load pretrained weights before training",
    )
    parser.add_argument(
        "--pretrained-source",
        choices=["hf", "local"],
        default="hf",
        help="Source for pretrained weights",
    )
    parser.add_argument("--pretrained-path", type=str, help="Local checkpoint path when using --pretrained --pretrained-source local")
    parser.add_argument(
        "--pretrained-repo-id",
        type=str,
        default="Avencast/EveNet",
        help="Hugging Face repo id for pretrained weights",
    )
    parser.add_argument(
        "--pretrained-filename",
        type=str,
        default="checkpoints.20M.a4.last.ckpt",
        help="File name for the pretrained checkpoint",
    )
    parser.add_argument(
        "--pretrained-cache-dir",
        type=str,
        help="Optional cache directory for pretrained weights",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    log_level = getattr(logging, args.log_level.upper())

    feature_names: Optional[Dict[str, Iterable[str]]] = None
    normalization_rules: Optional[Dict[str, Dict[str, str]]] = None

    def _load_yaml_or_json(path: Path) -> Dict[str, Any]:
        with open(path) as handle:
            return yaml.safe_load(handle)

    if args.feature_names:
        feature_names = _load_yaml_or_json(args.feature_names)

    if args.normalization_rules:
        normalization_rules = _load_yaml_or_json(args.normalization_rules)

    (train_features, train_labels) = _load_split(
        _resolve_paths(args.train_sig, "train signal"),
        _resolve_paths(args.train_bkg, "train background"),
    )

    val_features = None
    val_labels = None
    if args.val_sig and args.val_bkg:
        val_features, val_labels = _load_split(
            _resolve_paths(args.val_sig, "validation signal"),
            _resolve_paths(args.val_bkg, "validation background"),
        )


    classifier_config = dict(
        device=args.device,
        lr=args.lr,
        body_lr=args.body_lr,
        head_lr=args.head_lr,
        weight_decay=args.weight_decay,
        body_weight_decay=args.body_weight_decay,
        head_weight_decay=args.head_weight_decay,
        body_modules=args.body_modules,
        head_modules=args.head_modules,
        grad_clip=args.grad_clip,
        warmup_epochs=args.warmup_epochs,
        warmup_ratio=args.warmup_ratio,
        warmup_start_factor=args.warmup_start_factor,
        min_lr=args.min_lr,
        global_input_dim=args.global_input_dim,
        sequential_input_dim=args.sequential_input_dim,
        pretrained=args.pretrained,
        pretrained_source=args.pretrained_source,
        pretrained_path=args.pretrained_path,
        pretrained_repo_id=args.pretrained_repo_id,
        pretrained_filename=args.pretrained_filename,
        pretrained_cache_dir=args.pretrained_cache_dir,
        use_wandb=args.use_wandb,
        wandb={
            "project": args.wandb_project,
            "name": args.wandb_name,
        },
    )

    run_evenet_lite_training(
        train_features=train_features,
        train_labels=train_labels,
        val_features=val_features,
        val_labels=val_labels,
        class_labels=args.class_labels,
        feature_names=feature_names,
        normalization_rules=normalization_rules,
        sampler=None if args.sampler == "none" else args.sampler,
        epoch_size=args.epoch_size,
        epochs=args.epochs,
        batch_size=args.batch_size,
        checkpoint_path=str(args.checkpoint_path),
        resume_from=str(args.resume_from) if args.resume_from else None,
        checkpoint_every=args.checkpoint_every,
        save_top_k=args.save_top_k,
        monitor_metric=args.monitor_metric,
        minimize_metric=args.minimize_metric,
        debug=args.debug,
        log_level=log_level,
        **classifier_config
    )


if __name__ == "__main__":
    main()
