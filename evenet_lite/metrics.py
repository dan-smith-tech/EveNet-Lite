import math
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from scipy.special import expit, softmax
from sklearn.metrics import roc_auc_score

def compute_loss(logits: torch.Tensor, targets: torch.Tensor, weights: Optional[torch.Tensor]) -> torch.Tensor:
    per_sample = F.cross_entropy(logits, targets, reduction="none")
    if weights is not None:
        weights = weights.to(per_sample.device)
        per_sample = per_sample * weights
    return per_sample.mean()


def compute_accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
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
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
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

    return fpr_interp, tpr_uniform, sigma_fpr_interp


def convert_to_SIC(sig_eff: float, bkg_rej: float, bkg_rej_unc: Optional[float] = None) -> Tuple[Optional[float], Optional[float]]:
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

    # Raw background count at cuts (unweighted)
    bkg_raw = np.searchsorted(np.sort(scores[y_true == 0]), edges, side="left")
    bkg_raw = bkg_mask.sum() - bkg_raw

    # Valid region
    valid = (bkg_eff > 0) & (bkg_raw >= min_bkg_events)

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

    return {
        "sig_eff": sig_eff,
        "bkg_eff": bkg_eff,
        "bkg_eff_unc": bkg_eff_unc,
        "bkg_rej": bkg_rej,
        "bkg_rej_unc": bkg_rej_unc,
        "sic": sic,
        "sic_unc": sic_unc,
        "max_sic": max_sic,
        "max_sic_unc": max_sic_unc,
        "best_idx": best_idx,
    }


def plot_sic_diagnostics(
    targets: np.ndarray,
    scores: np.ndarray,
    weights: np.ndarray,
    sic_result: Dict[str, np.ndarray],
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

    valid_mask = np.isfinite(sic)

    # ROC curve
    axs[0, 0].plot(sig_eff, bkg_eff, lw=2, color="#1f77b4")
    axs[0, 0].set_ylabel("Background efficiency (FPR)")
    axs[0, 0].set_xlabel("Signal efficiency (TPR)")
    axs[0, 0].set_title("ROC curve")

    # SIC vs signal efficiency
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

    # Background rejection
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

    for ax in axs.flat:
        ax.grid(True, alpha=0.3)

    fig.suptitle("SIC diagnostics", fontsize=14)

    return fig


def calculate_physics_metrics(
    logits: np.ndarray,
    targets: np.ndarray,
    weights: np.ndarray,
    bins: int = 1000,
    min_bkg_events: int = 100,
    log_plots: bool = False,
    wandb_run: Optional[object] = None,
) -> Dict[str, np.ndarray]:
    """Calculates AUC and Max SIC with statistical uncertainty."""

    # Convert logits → scores
    if logits.ndim == 1 or logits.shape[1] == 1:
        scores = expit(logits.reshape(-1))
    else:
        scores = softmax(logits, axis=1)[:, 1]

    edges = np.linspace(0, 1, bins + 1)

    sic_result = compute_sic_from_scores(
        targets, scores, weights, edges, min_bkg_events=min_bkg_events
    )

    try:
        auc_val = roc_auc_score(targets, scores, sample_weight=weights)
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

    if log_plots and wandb_run is not None:
        fig = plot_sic_diagnostics(
            targets=targets,
            scores=scores,
            weights=weights,
            sic_result=sic_result,
        )
        try:
            import wandb

            wandb_run.log({"Physics/SIC": wandb.Image(fig)})
        finally:
            import matplotlib.pyplot as plt

            plt.close(fig)

    return metrics


def summarize_metrics(accumulator: Dict[str, float], counts: Dict[str, int]) -> Dict[str, float]:
    return {name: accumulator[name] / max(1, counts[name]) for name in accumulator}
