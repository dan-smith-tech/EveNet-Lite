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
) -> Tuple[float, float, np.ndarray, np.ndarray]:
    # Sort by score descending
    order = np.argsort(-scores)
    scores_sorted = scores[order]
    y_sorted = y_true[order]
    w_sorted = weights[order]

    sig_mask = (y_sorted == 1)
    bkg_mask = (y_sorted == 0)

    # Cumulative weighted sums
    w_sig = w_sorted * sig_mask
    w_bkg = w_sorted * bkg_mask
    w_bkg2 = (w_sorted ** 2) * bkg_mask

    cum_sig = np.cumsum(w_sig)
    cum_bkg = np.cumsum(w_bkg)
    cum_bkg2 = np.cumsum(w_bkg2)

    total_sig = cum_sig[-1]
    total_bkg = cum_bkg[-1]

    # Indices corresponding to score cuts
    idxs = np.searchsorted(-scores_sorted, -edges, side="left")

    # Efficiencies at cuts
    sig_eff = cum_sig[idxs] / (total_sig + 1e-12)
    bkg_eff = cum_bkg[idxs] / (total_bkg + 1e-12)

    # Raw background count at cuts (unweighted)
    bkg_raw = np.searchsorted(
        np.sort(scores[y_true == 0]), edges, side="left"
    )
    bkg_raw = bkg_mask.sum() - bkg_raw

    # Valid region
    valid = (bkg_eff > 0) & (bkg_raw >= min_bkg_events)

    sic = np.zeros_like(sig_eff)
    sic_unc = np.zeros_like(sig_eff)

    # Background rejection and uncertainty
    bkg_rej = np.zeros_like(bkg_eff)
    bkg_rej[valid] = 1.0 / bkg_eff[valid]

    bkg_rej_unc = np.zeros_like(bkg_eff)
    bkg_rej_unc[valid] = (
        np.sqrt(cum_bkg2[idxs][valid]) / (total_bkg + 1e-12)
    ) * bkg_rej[valid] ** 2

    sic[valid] = sig_eff[valid] * np.sqrt(bkg_rej[valid])
    sic_unc[valid] = sig_eff[valid] * 0.5 / np.sqrt(bkg_rej[valid]) * bkg_rej_unc[valid]

    idx_max = np.argmax(sic)

    eff_s = np.asarray(tpr)
    eff_b = np.asarray(fpr)
    eff_b_unc = np.asarray(fpr_unc)

    bkg_rej = np.zeros_like(eff_b)
    bkg_rej_unc = np.zeros_like(eff_b)

    mask = eff_b > 0
    bkg_rej[mask] = 1.0 / eff_b[mask]
    bkg_rej_unc[mask] = eff_b_unc[mask] * (bkg_rej[mask] ** 2)

    sic_vals = np.asarray(sic_vals)
    sic_unc_vals = np.asarray(sic_unc_vals)

    import matplotlib.pyplot as plt

    fig, axs = plt.subplots(2, 2, figsize=(12, 10))

    # ─────────────────────────────
    # ROC curve
    axs[0, 0].plot(eff_s, eff_b, lw=2)
    axs[0, 0].set_ylabel("Background efficiency (FPR)")
    axs[0, 0].set_xlabel("Signal efficiency (TPR)")
    axs[0, 0].grid(True)
    axs[0, 0].set_title("ROC curve")

    # ─────────────────────────────
    # SIC vs signal efficiency
    axs[0, 1].plot(eff_s, sic_vals, lw=2)
    axs[0, 1].fill_between(
        eff_s,
        sic_vals - sic_unc_vals,
        sic_vals + sic_unc_vals,
        alpha=0.3,
    )
    axs[0, 1].set_xlabel("Signal efficiency")
    axs[0, 1].set_ylabel("SIC")
    axs[0, 1].grid(True)
    axs[0, 1].set_title("SIC vs signal efficiency")

    # ─────────────────────────────
    # Background rejection
    axs[1, 0].plot(eff_s, bkg_rej, lw=2)
    axs[1, 0].fill_between(
        eff_s,
        bkg_rej - bkg_rej_unc,
        bkg_rej + bkg_rej_unc,
        alpha=0.3,
    )
    axs[1, 0].set_xlabel("Signal efficiency")
    axs[1, 0].set_ylabel("Background rejection (1 / ε_b)")
    axs[1, 0].set_yscale("log")
    axs[1, 0].grid(True)
    axs[1, 0].set_title("Background rejection")

    # ─────────────────────────────
    # Score distributions (signal vs background)
    sig_mask = (y_true == 1)
    bkg_mask = (y_true == 0)

    axs[1, 1].hist(
        scores[bkg_mask],
        bins=50,
        weights=weights[bkg_mask] if weights is not None else None,
        histtype="step",
        density=True,
        label="Background",
        linewidth=2,
    )

    axs[1, 1].hist(
        scores[sig_mask],
        bins=50,
        weights=weights[sig_mask] if weights is not None else None,
        histtype="step",
        density=True,
        label="Signal",
        linewidth=2,
    )

    axs[1, 1].set_xlabel("Classifier score")
    axs[1, 1].set_ylabel("Normalized events")
    axs[1, 1].set_yscale("log")  # optional but usually helpful
    axs[1, 1].grid(True)
    axs[1, 1].legend()
    axs[1, 1].set_title("Score distribution")

    plt.tight_layout()
    plt.show()

    return sic[idx_max], sic_unc[idx_max], sic, sic_unc


def calculate_physics_metrics(
    logits: np.ndarray,
    targets: np.ndarray,
    weights: np.ndarray,
    bins: int = 1000,
    min_bkg_events: int = 100,
) -> Dict[str, np.ndarray]:
    """Calculates AUC and Max SIC with statistical uncertainty."""

    # Convert logits → scores
    if logits.ndim == 1 or logits.shape[1] == 1:
        scores = expit(logits.reshape(-1))
    else:
        scores = softmax(logits, axis=1)[:, 1]

    edges = np.linspace(0, 1, bins + 1)

    max_sic, max_sic_unc, sic, sic_unc = compute_sic_from_scores(
        targets, scores, weights, edges, min_bkg_events=min_bkg_events
    )

    try:
        auc_val = roc_auc_score(targets, scores, sample_weight=weights)
    except ValueError:
        auc_val = 0.5

    return {
        "auc": float(auc_val),
        "max_sic": float(max_sic),
        "max_sic_unc": float(max_sic_unc),
        "sic": sic,
        "sic_unc": sic_unc,
        "edges": edges,
    }


def summarize_metrics(accumulator: Dict[str, float], counts: Dict[str, int]) -> Dict[str, float]:
    return {name: accumulator[name] / max(1, counts[name]) for name in accumulator}
