import math
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from scipy.special import expit, softmax
from sklearn.metrics import roc_auc_score


def _flatten_ensemble(
        logits: torch.Tensor, targets: torch.Tensor, weights: Optional[torch.Tensor]
) -> tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """Flatten an ensemble logits tensor to align with repeated targets/weights.

    Shapes
    ------
    logits: [E, B, C] -> [E * B, C]
    targets: [B] -> [E * B]
    weights: [B] -> [E * B]
    """

    if logits.dim() == 3:
        ensemble, batch, channels = logits.shape
        logits = logits.view(ensemble * batch, channels)
        targets = targets.repeat(ensemble)
        if weights is not None:
            weights = weights.repeat(ensemble)
    return logits, targets, weights


def _mean_ensemble_logits(logits: torch.Tensor) -> torch.Tensor:
    """Average ensemble logits along the ensemble dimension."""
    return logits.mean(dim=0) if logits.dim() == 3 else logits


def compute_loss(logits: torch.Tensor, targets: torch.Tensor, weights: Optional[torch.Tensor]) -> torch.Tensor:
    logits, targets, weights = _flatten_ensemble(logits, targets, weights)
    per_sample = F.cross_entropy(logits, targets, reduction="none")
    if weights is not None:
        weights = weights.to(per_sample.device)
        return sum(per_sample * weights) / sum(weights)
    return per_sample.mean()


def compute_accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    logits = _mean_ensemble_logits(logits)
    preds = logits.argmax(dim=1)
    correct = (preds == targets).sum().item()
    return correct / max(1, targets.numel())


def weighted_roc_curve(
        y_true: np.ndarray,
        y_score: np.ndarray,
        sample_weight: np.ndarray,
        bin_edges: Optional[np.ndarray] = None,
        n_points: int = 1000,
        safe_eps: float = 1e-6,
) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    # Sort by score descending
    sorted_idx = np.argsort(-y_score)
    y_true = y_true[sorted_idx]
    sample_weight = sample_weight[sorted_idx]
    is_sig = y_true == 1
    is_bkg = y_true == 0

    total_sig = np.sum(sample_weight[is_sig])
    total_bkg = np.sum(sample_weight[is_bkg])

    # Cumulative signal/background sums
    cum_sig = np.cumsum(sample_weight * is_sig)
    cum_bkg = np.cumsum(sample_weight * is_bkg)

    tpr_raw = np.concatenate(([0.0], cum_sig / (total_sig + safe_eps)))
    fpr_raw = np.concatenate(([0.0], cum_bkg / (total_bkg + safe_eps)))

    # Prepare cumulative background weight and weight^2
    w_bkg = sample_weight * is_bkg
    w_bkg2 = (sample_weight ** 2) * is_bkg
    cum_w_bkg = np.cumsum(w_bkg)
    cum_w2_bkg = np.cumsum(w_bkg2)

    # Insert zero at start to align with tpr/fpr
    cum_w_bkg = np.concatenate(([0.0], cum_w_bkg))
    cum_w2_bkg = np.concatenate(([0.0], cum_w2_bkg))

    # Compute effective n_bkg and σ_fpr vectorized
    n_eff = (cum_w_bkg ** 2) / (cum_w2_bkg + safe_eps)
    fpr_clipped = np.clip(fpr_raw, 0.0, 1.0)

    # Poisson-style uncertainty: σ = sqrt(sum w^2) / total_bkg
    sigma_fpr_raw = np.sqrt(cum_w2_bkg) / (total_bkg + safe_eps)

    # Interpolate everything to fixed TPR grid
    tpr_uniform = np.linspace(0, 1, n_points)
    fpr_interp = np.interp(tpr_uniform, fpr_raw, fpr_clipped)
    sigma_fpr_interp = np.interp(tpr_uniform, tpr_raw, sigma_fpr_raw)

    auc = np.trapz(tpr_raw, fpr_raw)
    return auc, fpr_interp, tpr_uniform, sigma_fpr_interp


def convert_to_SIC(sig_eff: float, bkg_rej: float, bkg_rej_unc: Optional[float] = None) -> Tuple[
    Optional[float], Optional[float]]:
    """Convert background rejection to SIC and propagate uncertainty."""

    if bkg_rej <= 0:
        return None, None

    sic = sig_eff * math.sqrt(bkg_rej)

    if bkg_rej_unc is None:
        return sic, None

    sic_unc = sig_eff * (0.5 / math.sqrt(bkg_rej)) * bkg_rej_unc
    return sic, sic_unc


def compute_sic_from_scores(
        y_true: np.ndarray,
        scores: np.ndarray,
        weights: np.ndarray,
        edges: np.ndarray,
        min_bkg_events: int = 10,
) -> Dict[str, np.ndarray]:
    """Compute SIC curve and related quantities on weighted scores.

    The calculation follows::

        SIC = ε_s / sqrt(ε_b) = ε_s * sqrt(1 / ε_b)

    where ``ε_s`` and ``ε_b`` are the weighted signal and background efficiencies.
    Background rejection is defined as ``1 / ε_b`` and its uncertainty is derived
    from the cumulative sum of squared background weights.
    """

    if scores.size == 0:
        empty = np.array([], dtype=float)
        return {
            "sig_eff": empty,
            "bkg_eff": empty,
            "bkg_eff_unc": empty,
            "bkg_rej": empty,
            "bkg_rej_unc": empty,
            "sic": empty,
            "sic_unc": empty,
            "max_sic": 0.0,
            "max_sic_unc": 0.0,
            "best_idx": 0,
        }

    eps = 1e-12

    # Sort by score descending
    order = np.argsort(-scores)
    scores_sorted = scores[order]
    y_sorted = y_true[order]
    w_sorted = weights[order]

    sig_mask = y_sorted == 1
    bkg_mask = y_sorted == 0

    # Cumulative weighted sums
    w_sig = w_sorted * sig_mask
    w_bkg = w_sorted * bkg_mask
    w_bkg2 = (w_sorted ** 2) * bkg_mask

    cum_sig = np.cumsum(w_sig)
    cum_bkg = np.cumsum(w_bkg)
    cum_bkg2 = np.cumsum(w_bkg2)

    total_sig = cum_sig[-1]
    total_bkg = cum_bkg[-1]

    if total_sig <= 0 or total_bkg <= 0:
        empty = np.zeros_like(edges, dtype=float)
        return {
            "sig_eff": empty,
            "bkg_eff": empty,
            "bkg_eff_unc": empty,
            "bkg_rej": empty,
            "bkg_rej_unc": empty,
            "sic": empty,
            "sic_unc": empty,
            "max_sic": 0.0,
            "max_sic_unc": 0.0,
            "best_idx": 0,
        }

    # Indices corresponding to score cuts
    idxs = np.searchsorted(-scores_sorted, -edges, side="left")
    idxs = np.clip(idxs, 0, len(cum_sig) - 1)

    # Efficiencies at cuts
    sig_eff = cum_sig[idxs] / (total_sig + eps)
    bkg_eff = cum_bkg[idxs] / (total_bkg + eps)
    bkg_eff_unc = np.sqrt(cum_bkg2[idxs]) / (total_bkg + eps)

    bkg_yield = cum_bkg[idxs]

    # Valid region
    valid = (bkg_eff > 0) & (bkg_yield >= min_bkg_events)

    # Full curves (without the minimum-background cut) for plotting
    bkg_rej_full = np.full_like(sig_eff, np.nan, dtype=float)
    sic_full = np.full_like(sig_eff, np.nan, dtype=float)
    positive_bkg = bkg_eff > 0
    bkg_rej_full[positive_bkg] = 1.0 / bkg_eff[positive_bkg]
    sic_full[positive_bkg] = sig_eff[positive_bkg] * np.sqrt(bkg_rej_full[positive_bkg])

    sic = np.full_like(sig_eff, np.nan, dtype=float)
    sic_unc = np.full_like(sig_eff, np.nan, dtype=float)
    bkg_rej = np.full_like(sig_eff, np.nan, dtype=float)
    bkg_rej_unc = np.full_like(sig_eff, np.nan, dtype=float)

    bkg_rej[valid] = 1.0 / bkg_eff[valid]
    bkg_rej_unc[valid] = bkg_eff_unc[valid] * (bkg_rej[valid] ** 2)
    sic[valid] = sig_eff[valid] * np.sqrt(bkg_rej[valid])
    sic_unc[valid] = sig_eff[valid] * 0.5 / np.sqrt(bkg_rej[valid]) * bkg_rej_unc[valid]

    if np.any(valid):
        best_idx = int(np.nanargmax(sic))
        max_sic = float(sic[best_idx])
        max_sic_unc = float(sic_unc[best_idx])
    else:
        best_idx = 0
        max_sic = 0.0
        max_sic_unc = 0.0

    valid = bkg_yield >= min_bkg_events

    if not np.any(valid):
        min_bkg_idx = None  # or raise / handle gracefully
    else:
        min_bkg_idx = np.where(valid)[0][-1]

    return {
        "sig_eff": sig_eff,
        "bkg_eff": bkg_eff,
        "bkg_eff_unc": bkg_eff_unc,
        "bkg_rej": bkg_rej,
        "bkg_rej_unc": bkg_rej_unc,
        "sic": sic,
        "sic_unc": sic_unc,
        "sic_full": sic_full,
        "bkg_rej_full": bkg_rej_full,
        "valid_mask": valid,
        "min_bkg_idx": min_bkg_idx,
        "max_sic": max_sic,
        "max_sic_unc": max_sic_unc,
        "best_idx": best_idx,
    }


def find_score_at_min_bkg(
        scores: np.ndarray,
        targets: np.ndarray,
        weights: Optional[np.ndarray],
        min_bkg_events: float,
) -> Optional[float]:
    """Return score threshold where remaining bkg yield == min_bkg_events."""
    bkg_mask = targets == 0

    bkg_scores = scores[bkg_mask]
    bkg_weights = weights[bkg_mask] if weights is not None else np.ones_like(bkg_scores)

    if bkg_scores.size == 0:
        return None

    # Sort by score descending (tightest cut first)
    order = np.argsort(-bkg_scores)
    bkg_scores = bkg_scores[order]
    bkg_weights = bkg_weights[order]

    # Cumulative remaining background
    cum_bkg = np.cumsum(bkg_weights)

    # Find first point where we exceed min_bkg_events
    idx = np.searchsorted(cum_bkg, min_bkg_events)

    if idx >= len(bkg_scores):
        return None

    return bkg_scores[idx]


def plot_sic_diagnostics(
        targets: np.ndarray,
        scores: np.ndarray,
        weights: np.ndarray,
        sic_result: Dict[str, np.ndarray],
        min_bkg_events: int = 0,
        *,
        figsize: Tuple[int, int] = (12, 12),
        dpi: int = 300,
):
    """Create a four-panel diagnostic plot for SIC-related curves."""

    import matplotlib.pyplot as plt

    fig, axs = plt.subplots(2, 2, figsize=figsize, dpi=dpi, constrained_layout=True)

    sig_eff = sic_result["sig_eff"]
    bkg_eff = sic_result["bkg_eff"]
    bkg_eff_unc = sic_result["bkg_eff_unc"]
    sic = sic_result["sic"]
    sic_unc = sic_result["sic_unc"]
    bkg_rej = sic_result["bkg_rej"]
    bkg_rej_unc = sic_result["bkg_rej_unc"]
    sic_full = sic_result.get("sic_full", sic)
    bkg_rej_full = sic_result.get("bkg_rej_full", bkg_rej)
    valid_mask = sic_result.get("valid_mask", np.isfinite(sic))
    min_bkg_idx = sic_result.get("min_bkg_idx")

    min_bkg_line_x = None
    if min_bkg_idx is not None:
        min_bkg_line_x = float(sig_eff[min_bkg_idx])

    # ROC curve
    axs[0, 0].plot(sig_eff, bkg_eff, lw=2, color="#1f77b4")
    axs[0, 0].set_ylabel("Background efficiency (FPR)")
    axs[0, 0].set_xlabel("Signal efficiency (TPR)")
    axs[0, 0].set_title("ROC curve")
    axs[0, 0].set_xlim(0.0, 1.0)
    # SIC vs signal efficiency
    axs[0, 1].plot(sig_eff, sic_full, lw=2, color="#d62728", alpha=0.5)
    axs[0, 1].plot(sig_eff, sic, lw=2, color="#d62728")
    axs[0, 1].fill_between(
        sig_eff,
        sic - sic_unc,
        sic + sic_unc,
        where=valid_mask,
        color="#d62728",
        alpha=0.2,
    )
    axs[0, 1].set_xlabel("Signal efficiency")
    axs[0, 1].set_ylabel("SIC")
    axs[0, 1].set_title("SIC vs signal efficiency")
    axs[0, 1].set_xlim(0.0, 1.0)
    # Background rejection
    axs[1, 0].plot(sig_eff, bkg_rej_full, lw=2, color="#2ca02c", alpha=0.5)
    axs[1, 0].plot(sig_eff, bkg_rej, lw=2, color="#2ca02c")
    axs[1, 0].fill_between(
        sig_eff,
        bkg_rej - bkg_rej_unc,
        bkg_rej + bkg_rej_unc,
        where=valid_mask,
        color="#2ca02c",
        alpha=0.2,
    )
    axs[1, 0].set_xlabel("Signal efficiency")
    axs[1, 0].set_ylabel("Background rejection (1 / ε_b)")
    axs[1, 0].set_yscale("log")
    axs[1, 0].set_title("Background rejection")
    axs[1, 0].set_xlim(0.0, 1.0)
    # Score distributions (signal vs background)
    sig_mask = targets == 1
    bkg_mask = targets == 0
    axs[1, 1].hist(
        scores[bkg_mask],
        bins=50,
        weights=weights[bkg_mask] if weights is not None else None,
        histtype="step",
        density=True,
        label="Background",
        linewidth=2,
        color="#1f77b4",
    )
    axs[1, 1].hist(
        scores[sig_mask],
        bins=50,
        weights=weights[sig_mask] if weights is not None else None,
        histtype="step",
        density=True,
        label="Signal",
        linewidth=2,
        color="#d62728",
    )
    axs[1, 1].set_xlabel("Classifier score")
    axs[1, 1].set_ylabel("Normalized events")
    axs[1, 1].set_yscale("log")
    axs[1, 1].legend()
    axs[1, 1].set_title("Score distribution")
    axs[1, 1].set_xlim(0.0, 1.0)

    if min_bkg_line_x is not None:
        score_cut = find_score_at_min_bkg(
            scores=scores,
            targets=targets,
            weights=weights,
            min_bkg_events=min_bkg_events,
        )

        if score_cut is not None:
            axs[1, 1].axvline(
                score_cut,
                color="gray", linestyle="--", alpha=0.75, lw=2,
                label=f"bkg = {min_bkg_events:g}",
            )

        for ax in [axs[0,0], axs[0,1], axs[1,0]]:
            ax.axvline(min_bkg_line_x, color="gray", linestyle="--", alpha=0.75, lw=2, label=f"bkg={min_bkg_events:5d}")
        axs[0, 0].legend(loc="lower right")
        axs[0, 1].legend(loc="upper right")
        axs[1, 0].legend(loc="upper right")


    for ax in axs.flat:
        ax.grid(True, alpha=0.3)

    fig.suptitle("SIC diagnostics", fontsize=14)

    return fig


def calculate_physics_metrics(
        logits: np.ndarray,
        targets: np.ndarray,
        weights: np.ndarray,
        training: bool,
        bins: int = 1000,
        min_bkg_events: int = 100,
        log_plots: bool = False,
        wandb_run: Optional[object] = None,
        log_step: Optional[int] = None,
        f_name: Optional[str] = None,
) -> Dict[str, np.ndarray]:
    """Calculates AUC and Max SIC with statistical uncertainty."""

    if logits.ndim == 3:
        logits = logits.mean(axis=0)

    # Convert logits → scores
    if logits.ndim == 1 or logits.shape[1] == 1:
        scores = logits
    else:
        scores = softmax(logits, axis=1)[:, 1]

    edges = np.linspace(0, 1, bins + 1)

    sic_result = compute_sic_from_scores(
        targets, scores, weights, edges, min_bkg_events=min_bkg_events
    )

    try:
        # auc_val = roc_auc_score(targets, scores, sample_weight=weights)
        auc_val, _, _, _ = weighted_roc_curve(targets, scores, sample_weight=weights)
    except ValueError:
        auc_val = 0.5

    metrics = {
        "auc": float(auc_val),
        "max_sic": float(sic_result["max_sic"]),
        "max_sic_unc": float(sic_result["max_sic_unc"]),
        "sic": sic_result["sic"],
        "sic_unc": sic_result["sic_unc"],
        "edges": edges,
    }

    if log_plots:
        fig = plot_sic_diagnostics(
            targets=targets,
            scores=scores,
            weights=weights,
            sic_result=sic_result,
            min_bkg_events=min_bkg_events,
        )

        if f_name is not None:
            fig.savefig(f_name, dpi=300, bbox_inches="tight")
        if wandb_run is not None:
            try:
                import wandb

                if training:
                    log_name = "Physics/train-SIC"
                else:
                    log_name = "Physics/valid-SIC"

                log_kwargs = {"step": log_step} if log_step is not None else {}
                wandb_run.log({log_name: wandb.Image(fig)}, **log_kwargs)
            finally:
                pass
        import matplotlib.pyplot as plt
        plt.close(fig)

    return metrics


def summarize_metrics(accumulator: Dict[str, float], counts: Dict[str, int]) -> Dict[str, float]:
    return {name: accumulator[name] / max(1, counts[name]) for name in accumulator}
