import argparse
import logging
import re
from glob import glob
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch
import yaml

from evenet_lite import run_evenet_lite_training
from evenet_lite.optim import (
    DEFAULT_LR_GROUPS,
    DEFAULT_MODULE_GROUPS,
    DEFAULT_WEIGHT_DECAY,
)


FeatureBundle = Tuple[Dict[str, torch.Tensor], torch.Tensor]


logger = logging.getLogger(__name__)


def _resolve_paths(path_str: str, description: str) -> List[Path]:
    """Resolve one or more concrete files from a path or glob pattern."""

    candidate = Path(path_str)
    if candidate.is_file():
        return [candidate]

    def _expand_braces(pattern: str) -> List[str]:
        match = re.search(r"\{([^{}]+)\}", pattern)
        if not match:
            return [pattern]

        options = match.group(1).split(",")
        prefix = pattern[: match.start()]
        suffix = pattern[match.end():]
        expanded: List[str] = []
        for option in options:
            expanded.extend(_expand_braces(prefix + option + suffix))
        return expanded

    expanded_patterns = _expand_braces(path_str)
    matches = sorted({Path(p) for pat in expanded_patterns for p in glob(pat)})
    if not matches:
        raise FileNotFoundError(
            f"No files matched {description} pattern: {path_str} (expanded to {expanded_patterns})"
        )

    return matches


def _concat_tensors(data_iter: Iterable[torch.Tensor]) -> torch.Tensor:
    return torch.cat(list(data_iter), dim=0)


BKG_META = {
    "tt1l": {
        "xsec": 365.35,
        "nEvent": 144_722_000,
    },
    "DYBJets_pt100to200": {
        "xsec": 3.222,
        "nEvent": 8_848_155,
    },
    "DYBJets_pt200toInf": {
        "xsec": 0.6181,
        "nEvent": 887_122,
    },
    "ggHtautau": {
        "xsec": 3.08,
        "nEvent": 6_439_000,
    },
    "VBFHtautau": {
        "xsec": 0.237,
        "nEvent": 1_500_000,
    },
}

def _match_bkg_sample(path: Path) -> str:
    path_str = str(path)
    for name in BKG_META:
        if name in path_str:
            return name
    raise ValueError(f"Cannot match background sample for path: {path}")

def _make_sample_weights(path: Path, n_events: int) -> torch.Tensor:
    sample = _match_bkg_sample(path)
    meta = BKG_META[sample]
    # w = meta["xsec"] / meta["nEvent"]
    w = meta["xsec"] / meta["nEvent"] * 1000 * 36
    return torch.full((n_events,), w, dtype=torch.float32)

def _load_split(sig_paths: List[Path], bkg_paths: List[Path]):
    sig_parts = [torch.load(p, weights_only=False, map_location="cpu") for p in sig_paths]
    bkg_parts = [torch.load(p, weights_only=False, map_location="cpu") for p in bkg_paths]

    available_keys = {key for part in [*sig_parts, *bkg_parts] for key in part.keys()}
    requested_keys = ["x", "x_mask", "global", "params"]

    features = {}
    for key in requested_keys:
        if key not in available_keys or any(key not in part for part in [*sig_parts, *bkg_parts]):
            continue
        alias = "globals" if key == "global" else key
        features[alias] = _concat_tensors(
            [*(part[key] for part in sig_parts),
             *(part[key] for part in bkg_parts)]
        )

    # labels
    n_sig = sum(len(part["x"]) for part in sig_parts)
    n_bkg = sum(len(part["x"]) for part in bkg_parts)

    labels = torch.cat(
        [
            torch.ones(n_sig),
            torch.zeros(n_bkg),
        ],
        dim=0,
    )

    # ---- weights (physics-correct) ----
    sig_weights = torch.ones(n_sig, dtype=torch.float32)

    bkg_weights = []
    for path, part in zip(bkg_paths, bkg_parts):
        n = len(part["x"])
        bkg_weights.append(_make_sample_weights(path, n))

    weights = torch.cat([sig_weights, *bkg_weights], dim=0)

    return features, labels, weights


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="DDP-friendly Evenet-Lite runner for NERSC")
    parser.add_argument("--train-sig", type=str, required=True, help="Path or glob pattern to signal training tensor (.pt)")
    parser.add_argument("--train-bkg", type=str, required=True, help="Path or glob pattern to background training tensor (.pt)")
    parser.add_argument(
        "--train-fraction",
        type=float,
        default=0.7,
        help="Fraction of the training split to use for fitting (rest used for validation)",
    )
    parser.add_argument(
        "--split-seed",
        type=int,
        default=42,
        help="Random seed used when splitting the training set into train/validation",
    )
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
    parser.add_argument("--early-stop-metric", type=str, default="val_loss", help="Metric used for early stopping")
    parser.add_argument("--early-stop-patience", type=int, default=0, help="Epochs to wait before early stopping")
    parser.add_argument(
        "--early-stop-minimize",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whether the early stop metric should be minimized",
    )
    parser.add_argument("--eval-sig", type=str, help="Optional path or glob pattern to signal evaluation tensor (.pt)")
    parser.add_argument("--eval-bkg", type=str, help="Optional path or glob pattern to background evaluation tensor (.pt)")
    parser.add_argument("--eval-output", type=Path, help="Path to save evaluation predictions")
    parser.add_argument("--eval-batch-size", type=int, help="Optional batch size override for evaluation")
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
        "--normalization-stats",
        type=Path,
        help="Optional YAML/JSON file with precomputed normalization stats to pass to the trainer",
    )
    parser.add_argument(
        "--normalization-rules",
        type=Path,
        help="Optional YAML/JSON file with normalization rules to pass to the trainer",
    )

    parser.add_argument("--device", type=str, default="auto", help="Device placement for training")
    parser.add_argument(
        "--lr",
        type=float,
        nargs="+",
        default=list(DEFAULT_LR_GROUPS),
        help="Learning rates for each optimizer group",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        nargs="+",
        default=[DEFAULT_WEIGHT_DECAY] * len(DEFAULT_LR_GROUPS),
        help="Weight decay values for each optimizer group",
    )
    parser.add_argument(
        "--module-lists",
        action="append",
        nargs="+",
        help="Module names assigned to an optimizer group; repeat for multiple groups",
    )
    parser.add_argument("--grad-clip", type=float, default=1.0, help="Gradient clipping value")
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
    parser.add_argument("--sic-min-bkg-events", type=int, default=10, help="Minimum background events for SIC")
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

    parser.add_argument(
        "--num-workers", type=int, default=0, help="Number of workers for data loading"
    )
    parser.add_argument("--n-ensemble", type=int, default=1, help="Number of ensemble members in the EveNet-Lite model")
    parser.add_argument(
        "--ensemble-mode",
        type=str,
        choices=["independent", "shared_backbone"],
        default="independent",
        help="Whether ensemble heads share a backbone or are fully independent",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    log_level = getattr(logging, args.log_level.upper())

    feature_names: Optional[Dict[str, Iterable[str]]] = None
    normalization_rules: Optional[Dict[str, Dict[str, str]]] = None
    normalization_stats: Optional[Dict[str, Any]] = None

    def _load_yaml_or_json(path: Path) -> Dict[str, Any]:
        with open(path) as handle:
            return yaml.safe_load(handle)

    if args.feature_names:
        feature_names = _load_yaml_or_json(args.feature_names)

    if args.normalization_rules:
        normalization_rules = _load_yaml_or_json(args.normalization_rules)
    if args.normalization_stats:
        normalization_stats = _load_yaml_or_json(args.normalization_stats)

    (train_features, train_labels, train_weights) = _load_split(
        _resolve_paths(args.train_sig, "train signal"),
        _resolve_paths(args.train_bkg, "train background"),
    )

    if not 0 < args.train_fraction < 1:
        raise ValueError("--train-fraction must be between 0 and 1 (exclusive)")

    total_train = train_labels.shape[0]
    split_seed = torch.Generator().manual_seed(args.split_seed) if args.split_seed is not None else None
    perm = torch.randperm(total_train, generator=split_seed)
    train_size = int(args.train_fraction * total_train)
    train_idx = perm[:train_size]
    val_idx = perm[train_size:]

    def _slice_features(features: Dict[str, torch.Tensor], indices: torch.Tensor) -> Dict[str, torch.Tensor]:
        return {name: tensor[indices] for name, tensor in features.items()}

    val_features = _slice_features(train_features, val_idx)
    val_labels = train_labels[val_idx]
    val_weights = train_weights[val_idx] if train_weights is not None else None

    train_features = _slice_features(train_features, train_idx)
    train_labels = train_labels[train_idx]
    train_weights = train_weights[train_idx] if train_weights is not None else None

    eval_features = None
    eval_labels = None
    eval_weights = None
    eval_sig_pattern = args.eval_sig or args.val_sig
    eval_bkg_pattern = args.eval_bkg or args.val_bkg
    if eval_sig_pattern and eval_bkg_pattern:
        eval_features, eval_labels, eval_weights = _load_split(
            _resolve_paths(eval_sig_pattern, "evaluation signal"),
            _resolve_paths(eval_bkg_pattern, "evaluation background"),
        )


    classifier_config = dict(
        device=args.device,
        lr=args.lr,
        weight_decay=args.weight_decay,
        module_lists=args.module_lists if args.module_lists else list(DEFAULT_MODULE_GROUPS),
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
        num_workers=args.num_workers,
        n_ensemble=args.n_ensemble,
        ensemble_mode=args.ensemble_mode,
    )

    run_evenet_lite_training(
        train_features=train_features,
        train_labels=train_labels,
        train_weights=train_weights,
        val_features=val_features,
        val_labels=val_labels,
        val_weights=val_weights,
        class_labels=args.class_labels,
        feature_names=feature_names,
        normalization_rules=normalization_rules,
        normalization_stats=normalization_stats,
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
        early_stop_metric=args.early_stop_metric,
        early_stop_minimize=args.early_stop_minimize,
        early_stop_patience=args.early_stop_patience,
        eval_features=eval_features,
        eval_labels=eval_labels,
        eval_weights=eval_weights,
        eval_output_path=str(args.eval_output) if args.eval_output else None,
        eval_batch_size=args.eval_batch_size,
        sic_min_bkg_events=args.sic_min_bkg_events,
        debug=args.debug,
        log_level=log_level,
        **classifier_config
    )


if __name__ == "__main__":
    main()
