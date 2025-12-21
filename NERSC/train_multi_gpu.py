import argparse
import logging
from pathlib import Path
from typing import Dict, Tuple

import torch

from evenet_lite import run_evenet_lite_training


FeatureBundle = Tuple[Dict[str, torch.Tensor], torch.Tensor]


def _load_split(sig_path: Path, bkg_path: Path) -> FeatureBundle:
    sig_data = torch.load(sig_path)
    bkg_data = torch.load(bkg_path)

    features = {
        ("globals" if key == "global" else key): torch.cat([sig_data[key], bkg_data[key]], dim=0)
        for key in ["x", "x_mask", "global"]
    }
    labels = torch.cat(
        [
            torch.ones(len(sig_data["x"]), device=features["x"].device),
            torch.zeros(len(bkg_data["x"]), device=features["x"].device),
        ],
        dim=0,
    )
    return features, labels


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="DDP-friendly Evenet-Lite runner for NERSC")
    parser.add_argument("--train-sig", type=Path, required=True, help="Path to signal training tensor (.pt)")
    parser.add_argument("--train-bkg", type=Path, required=True, help="Path to background training tensor (.pt)")
    parser.add_argument("--val-sig", type=Path, help="Optional path to signal validation tensor (.pt)")
    parser.add_argument("--val-bkg", type=Path, help="Optional path to background validation tensor (.pt)")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=512, help="Mini-batch size")
    parser.add_argument("--sampler", choices=["weighted", "none"], default="weighted", help="Sampler strategy")
    parser.add_argument("--checkpoint-path", type=Path, default=Path("checkpoints"), help="Directory for checkpoints")
    parser.add_argument("--resume-from", type=Path, help="Optional checkpoint to resume training")
    parser.add_argument("--checkpoint-every", type=int, default=1, help="Checkpoint frequency in epochs")
    parser.add_argument("--save-top-k", type=int, default=2, help="Number of best checkpoints to keep")
    parser.add_argument("--debug", action="store_true", help="Enable verbose DebugCallback logs")
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Root logger level",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    log_level = getattr(logging, args.log_level.upper())

    (train_features, train_labels) = _load_split(args.train_sig, args.train_bkg)

    val_features = None
    val_labels = None
    if args.val_sig and args.val_bkg:
        val_features, val_labels = _load_split(args.val_sig, args.val_bkg)

    run_evenet_lite_training(
        train_features=train_features,
        train_labels=train_labels,
        val_features=val_features,
        val_labels=val_labels,
        class_labels=["background", "signal"],
        sampler=None if args.sampler == "none" else args.sampler,
        epochs=args.epochs,
        batch_size=args.batch_size,
        checkpoint_path=str(args.checkpoint_path),
        resume_from=str(args.resume_from) if args.resume_from else None,
        checkpoint_every=args.checkpoint_every,
        save_top_k=args.save_top_k,
        debug=args.debug,
        log_level=log_level,
    )


if __name__ == "__main__":
    main()
