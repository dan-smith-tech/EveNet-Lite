import math
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

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
    fpr, tpr, fpr_unc = weighted_roc_curve(y_true, scores, weights, edges)

    bkg_scores_all = scores[y_true == 0]
    bkg_scores_sorted = np.sort(bkg_scores_all)
    n_total_bkg = len(bkg_scores_sorted)
    idxs = np.searchsorted(bkg_scores_sorted, edges, side="left")
    n_bkg_pass_raw_arr = n_total_bkg - idxs

    sic_vals = []
    sic_unc_vals = []

    for eff_s, eff_b, eff_b_unc, n_raw in zip(tpr, fpr, fpr_unc, n_bkg_pass_raw_arr):
        if eff_b <= 0 or n_raw < min_bkg_events:
            sic_vals.append(0.0)
            sic_unc_vals.append(0.0)
            continue

        bkg_rej = 1.0 / eff_b
        bkg_rej_unc = eff_b_unc * (bkg_rej ** 2)

        sic, sic_unc = convert_to_SIC(sig_eff=eff_s, bkg_rej=bkg_rej, bkg_rej_unc=bkg_rej_unc)

        sic_vals.append(0.0 if sic is None else sic)
        sic_unc_vals.append(0.0 if sic_unc is None else sic_unc)

    sic_vals = np.array(sic_vals)
    sic_unc_vals = np.array(sic_unc_vals)

    idx = np.argmax(sic_vals)

    return sic_vals[idx], sic_unc_vals[idx], sic_vals, sic_unc_vals


def calculate_physics_metrics(
    probs: np.ndarray,
    targets: np.ndarray,
    weights: np.ndarray,
    bins: int = 1000,
) -> Dict[str, np.ndarray]:
    """Calculates AUC and Max SIC with statistical uncertainty."""

    if probs.ndim > 1:
        scores = probs[:, 1]
    else:
        scores = probs

    edges = np.linspace(0, 1, bins + 1)

    max_sic, max_sic_unc, sic, sic_unc = compute_sic_from_scores(targets, scores, weights, edges)

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
