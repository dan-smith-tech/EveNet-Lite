import numpy as np
import torch
import matplotlib.pyplot as plt


# ==========================================
# 3. Plotting Helpers (accept torch or numpy; convert internally)
# ==========================================

def plot_score_overlay(y_eval, y_pred, w_eval, p_eval, bins=None, fname=None, uniform_bin_plot=False):
    # Convert tensors to numpy for matplotlib
    if isinstance(y_eval, torch.Tensor): y_eval = y_eval.detach().cpu().numpy()
    if isinstance(y_pred, torch.Tensor): y_pred = y_pred.detach().cpu().numpy()
    if isinstance(w_eval, torch.Tensor): w_eval = w_eval.detach().cpu().numpy()

    mask_signal = (y_eval == 1)
    mask_bkg = (y_eval == 0)

    bkg_processes = np.unique(p_eval[mask_bkg])
    bkg_data, bkg_weights, bkg_labels = [], [], []

    for proc in bkg_processes:
        mask_proc = (p_eval == proc) & mask_bkg
        if np.sum(mask_proc) > 0:
            bkg_data.append(y_pred[mask_proc])
            bkg_weights.append(w_eval[mask_proc])
            bkg_labels.append(f"{proc}[{np.sum(w_eval[mask_proc]):.0f}]")

    plt.figure(figsize=(10, 7))

    if bins is None:
        bins = np.linspace(0, 1, 40)

    # =========================
    # NEW: uniform bin plotting
    # =========================
    if uniform_bin_plot:
        x = np.arange(len(bins) - 1)

        bottom = np.zeros_like(x, dtype=float)
        for data, weights, label in zip(bkg_data, bkg_weights, bkg_labels):
            counts, _ = np.histogram(data, bins=bins, weights=weights)
            plt.bar(
                x,
                counts,
                bottom=bottom,
                width=1.0,
                alpha=0.7,
                edgecolor="white",
                linewidth=0.3,
                label=label,
            )
            bottom += counts

        if np.sum(mask_signal) > 0:
            sig_counts, _ = np.histogram(
                y_pred[mask_signal],
                bins=bins,
                weights=w_eval[mask_signal],
            )
            n = len(sig_counts)
            # Edges in "bin-index space" so bin i is [-0.5+i, +0.5+i] and centered at i
            x_edges = np.arange(n + 1) - 0.5
            # Repeat last value so the final horizontal segment is drawn
            y_step = np.r_[sig_counts, sig_counts[-1]]
            plt.step(
                x_edges,
                y_step,
                where="post",
                linewidth=2.5,
                color="red",
                label="Signal",
            )
            plt.xlim(-0.5, n - 0.5)  # optional, but makes the intent explicit

        plt.yscale("log")
        plt.xlabel("Bin index")
        plt.ylim(plt.ylim()[0], plt.ylim()[1] * 50)

    # =========================
    # ORIGINAL behavior
    # =========================
    else:
        if bkg_data:
            plt.hist(
                bkg_data, bins=bins, weights=bkg_weights, stacked=True,
                label=bkg_labels, alpha=0.7, edgecolor="white", linewidth=0.3,
                density=True, log=True
            )

        if np.sum(mask_signal) > 0:
            plt.hist(
                y_pred[mask_signal], bins=bins, weights=w_eval[mask_signal],
                histtype="step", linewidth=2.5, color="red", label="Signal",
                density=True, log=True
            )

        plt.xlabel("Score ($y_{pred}$)")

    plt.ylabel("Weighted Events")
    plt.title("Score Distribution (EveNet)")
    plt.legend(loc="upper center", bbox_to_anchor=(0.5, 0.98), ncol=3)
    plt.grid(axis="y", linestyle="--", alpha=0.3)

    if fname:
        plt.savefig(fname)
        plt.close()
