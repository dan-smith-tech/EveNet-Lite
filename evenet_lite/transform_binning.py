import numpy as np
import logging

logger = logging.getLogger("AutoBin")


def trafo60_binning(
        scores, labels, weights,
        Zb, Zs,
        edges_low=None,
        edges_high=None,
        min_mc_yield=10.0,
        mc_stat_bound=0.5,
        include_signal=False,
        *,
        logger: logging.Logger | None = None,
        log_level=logging.INFO,
        log_all_steps: bool = True,  # True = log every j step; False = throttle
        log_every: int = 20,  # if not log_all_steps, log every N j-steps
):
    if logger is None:
        logger = logging.getLogger("AutoBin")
    logger.setLevel(log_level)

    # --- default fine binning ---
    if edges_low is None:
        edges_low = np.linspace(0.0, 0.999, 1001)
    if edges_high is None:
        edges_high = np.linspace(0.999, 1.000, 1001)

    bin_edges = np.concatenate([edges_low[:-1], edges_high])
    nbins = len(bin_edges) - 1

    # --- build histograms ---
    is_bkg = labels == 0
    is_sig = labels == 1

    bkg_hist, _ = np.histogram(scores[is_bkg], bins=bin_edges, weights=weights[is_bkg])
    sig_hist, _ = np.histogram(scores[is_sig], bins=bin_edges, weights=weights[is_sig])
    bkg_w2_hist, _ = np.histogram(scores[is_bkg], bins=bin_edges, weights=weights[is_bkg] ** 2)

    # Totals
    N_b = float(bkg_hist.sum())
    N_s = float(sig_hist.sum())

    logger.debug(
        "Trafo60 start | nbins=%d | N_b=%.6g N_s=%.6g | min_mc_yield=%.6g mc_stat_bound=%.6g include_signal=%s | Zb=%.6g Zs=%.6g",
        nbins, N_b, N_s, float(min_mc_yield), float(mc_stat_bound), include_signal, float(Zb), float(Zs)
    )

    # --- Trafo-60 (right → left) ---
    rebin_edges = [nbins]
    i = nbins - 1

    # running sums of finalized bins (useful to see what you already “packed”)
    packed_b = 0.0
    packed_s = 0.0
    packed_err2_b = 0.0

    step_counter = 0

    while i >= 0:
        sum_b = 0.0
        sum_s = 0.0
        err2_b = 0.0
        best_dist = np.inf
        best_j = None
        passed = False

        j = i
        logger.debug("---- new target (right edge) i=%d | score_high=%.6f", i, float(bin_edges[i + 1]))

        while j >= 0:
            sum_b += float(bkg_hist[j])
            sum_s += float(sig_hist[j])
            err2_b += float(bkg_w2_hist[j])

            bkg_unc = np.sqrt(err2_b)
            rel_mc_stat = (bkg_unc / sum_b) if sum_b > 0 else np.inf

            if (sum_b + sum_s) <= 0:
                j -= 1
                continue

            # Trafo-D core
            denom = 0.0
            if sum_b > 0 and N_b > 0:
                denom += sum_b / (N_b / float(Zb))
            if sum_s > 0 and N_s > 0:
                denom += sum_s / (N_s / float(Zs))

            if denom <= 0.0:
                logger.warning(
                    "break: denom<=0 at i=%d j=%d | sum_b=%.6g sum_s=%.6g (N_b=%.6g N_s=%.6g)",
                    i, j, sum_b, sum_s, N_b, N_s
                )
                break

            err2Rel = 1.0 / denom
            dist = abs(err2Rel - 1.0)

            pass_core = (np.sqrt(err2Rel) < 1.0)
            pass_mc = (rel_mc_stat < mc_stat_bound)
            pass_yield = ((sum_b + sum_s) >= min_mc_yield) if include_signal else (sum_b >= min_mc_yield)
            ok = pass_core and pass_mc and pass_yield

            # logging (throttled unless log_all_steps)
            step_counter += 1
            do_log = log_all_steps or (step_counter % log_every == 0) or ok
            if do_log:
                logger.debug(
                    "i=%d j=%d | edge=[%.6f, %.6f] | S=%.6g B=%.6g Bunc=%.6g relMC=%.4f | "
                    "err2Rel=%.4f sqrt=%.4f dist=%.4f | pass(core=%s mc=%s yield=%s) ok=%s | "
                    "packed(S=%.6g B=%.6g Bunc=%.6g)",
                    i, j,
                    float(bin_edges[j]), float(bin_edges[i + 1]),
                    sum_s, sum_b, bkg_unc, rel_mc_stat,
                    err2Rel, np.sqrt(err2Rel), dist,
                    pass_core, pass_mc, pass_yield, ok,
                    packed_s, packed_b, np.sqrt(packed_err2_b)
                )

            if ok:
                passed = True
                if dist < best_dist:
                    best_dist = dist
                    best_j = j
                else:
                    logger.debug(
                        "stop scan (distance worsened) at i=%d j=%d | best_j=%s best_dist=%.4f current_dist=%.4f",
                        i, j, str(best_j), best_dist, dist
                    )
                    break

            j -= 1

        if not passed:
            logger.debug(
                "FAILED to find a valid bin for i=%d. Stopping.",
                i
            )
            break

        # finalize this merged bin: [best_j, i]
        # compute finalized sums precisely from hists (robust)
        fin_b = float(bkg_hist[best_j:i + 1].sum())
        fin_s = float(sig_hist[best_j:i + 1].sum())
        fin_err2 = float(bkg_w2_hist[best_j:i + 1].sum())

        packed_b += fin_b
        packed_s += fin_s
        packed_err2_b += fin_err2

        logger.debug(
            "FINAL bin: idx [%d..%d] | edge=[%.6f, %.6f] | S=%.6g B=%.6g Bunc=%.6g | "
            "packed: S=%.6g/%.6g B=%.6g/%.6g (fracB=%.3f fracS=%.3f)",
            best_j, i,
            float(bin_edges[best_j]), float(bin_edges[i + 1]),
            fin_s, fin_b, np.sqrt(fin_err2),
            packed_s, N_s, packed_b, N_b,
            (packed_b / N_b) if N_b > 0 else np.nan,
            (packed_s / N_s) if N_s > 0 else np.nan,
        )

        rebin_edges.append(best_j)
        i = best_j - 1

    # --- convert bin indices → score edges ---
    rebin_edges = sorted(set(rebin_edges))
    final_edges = [bin_edges[idx] for idx in rebin_edges if idx < len(bin_edges)]
    if len(final_edges) == 0 or final_edges[-1] < bin_edges[-1]:
        final_edges.append(bin_edges[-1])

    final_edges = np.array(final_edges, dtype=float)

    # summary
    logger.debug(
        "Trafo60 done | n_final_bins=%d | packed totals: S=%.6g/%.6g B=%.6g/%.6g",
        len(final_edges) - 1,
        packed_s, N_s,
        packed_b, N_b
    )

    return final_edges


def calculate_binned_significance(N_sig, N_bkg, method="asimov"):
    """
    Calculate the significance for each bin.

    Parameters:
    - signal_counts: np.array, array of signal event counts per bin.
    - background_counts: np.array, array of background event counts per bin.
    - method: str, method to calculate significance ('simple' or 'asimov').

    Returns:
    - np.array of significances for each bin.
    """
    significances = np.zeros_like(N_sig)

    if method == "simple":
        with np.errstate(divide='ignore', invalid='ignore'):
            significances = np.where(
                N_bkg > 0,
                N_sig / np.sqrt(N_bkg),
                0
            )

    elif method == "asimov":
        with np.errstate(divide='ignore', invalid='ignore'):
            significances = np.where(
                N_bkg > 0,
                np.sqrt(2 * ((N_sig + N_bkg) * np.log(1 + N_sig / N_bkg) - N_sig)),
                0
            )

    else:
        raise ValueError("Invalid method specified. Choose 'simple' or 'asimov'.")

    return significances


def binned_sig(
        test_data, test_label, test_weights,
        Zb=5, Zs=10, min_bkg_per_bin=3, min_mc_stats=1.0, method="asimov",
        reweight_factor=1, include_signal=True, edges_low=None, edges_high=None,
        logger: logging.Logger | None = None,
):
    """
    Calculate the binned significance based on Transformation D binning.

    Parameters:
    - test_data: np.array, data values to bin (e.g., classifier scores).
    - test_label: np.array, binary labels (0 for background, 1 for signal).
    - test_weights: np.array, weights for each event.
    - Zb, Zs: float, weight factors for background and signal in TrafoD binning.
    - min_bkg_per_bin: int, minimum number of background events per bin.
    - min_mc_stats: float, minimum statistical uncertainty for MC stats.
    - method: str, significance calculation method ('simple' or 'asimov').

    Returns:
    - bin_edges: list of bin edges used.
    - significances: np.array of significances for each bin.
    """
    # Step 1: Perform TrafoD binning to get bin edges
    bin_edges = trafo60_binning(
        test_data, test_label, test_weights, Zb, Zs,
        edges_low=edges_low,
        edges_high=edges_high,
        min_mc_yield=min_bkg_per_bin,
        mc_stat_bound=min_mc_stats,
        include_signal=include_signal,
        logger=logger,
    )

    # Step 2: Initialize lists to store signal and background counts per bin
    is_bkg = test_label == 0
    is_signal = test_label == 1
    bkg_hist, _ = np.histogram(test_data[is_bkg], bins=bin_edges, weights=test_weights[is_bkg])
    sig_hist, _ = np.histogram(test_data[is_signal], bins=bin_edges, weights=test_weights[is_signal])

    # Step 4: Calculate significance for each bin
    significances = calculate_binned_significance(sig_hist, bkg_hist, method=method)

    def log_binning_summary(
            logger,
            bin_edges,
            sig_hist,
            bkg_hist,
            significances,
            precision=3,
            level="info",
    ):
        if logger is None:
            logger = logging.getLogger("AutoBin")

        log = getattr(logger, level)

        header = (
            f"{'bin':>3} | {'low':>8} {'high':>8} | "
            f"{'signal':>10} {'bkg':>10} | {'Z':>8}"
        )
        sep = "-" * len(header)

        log(header)
        log(sep)

        for i in range(len(sig_hist)):
            log(
                f"{i:3d} | "
                f"{bin_edges[i]:8.{precision}f} {bin_edges[i + 1]:8.{precision}f} | "
                f"{sig_hist[i]:10.{precision}f} {bkg_hist[i]:10.{precision}f} | "
                f"{significances[i]:8.{precision}f}"
            )

    log_binning_summary(
        logger,
        bin_edges,
        sig_hist,
        bkg_hist,
        significances,
        precision=4,
        level="info",  # or "debug"
    )

    return bin_edges, sum(significances)
