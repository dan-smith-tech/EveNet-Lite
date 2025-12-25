import sys
import os
import argparse
import re
import json
import yaml
import numpy as np
import logging
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple, Any
from pathlib import Path
import time

# --- PyTorch & EveNet Imports ---
import torch
import torch.distributed as dist

try:
    from evenet_lite import run_evenet_lite_training
    from evenet_lite.callbacks import ParameterRandomizationCallback
    HAS_EVENET = True

except ImportError as e:
    print(e)
    HAS_EVENET = False
    print("Error: evenet_lite not installed. Run 'pip install -e .'")

import matplotlib.pyplot as plt

try:
    import mplhep as hep

    plt.style.use(hep.style.CMS)
except ImportError:
    pass

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

# --- Physics Metric Fallback ---
try:
    from evenet_lite.metrics import calculate_physics_metrics
except ImportError:
    from sklearn.metrics import roc_auc_score


    def calculate_physics_metrics(probs, targets, weights):
        return {
            'max_sic_unc': 0.0, 'max_sic': 0.0,
            'auc': roc_auc_score(targets, probs, sample_weight=weights)
        }


# ==========================================
# 1. Configuration & Data Structures
# ==========================================

def concat_ds(d1: dict, d2: dict, keys):
    """
    Concatenate two dataset dicts.
    - torch.Tensor keys: torch.cat
    - non-tensor keys (e.g. proc): numpy concatenate as object array
    """
    out = {}
    for k in keys:
        a, b = d1[k], d2[k]
        if isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor):
            out[k] = torch.cat([a, b], dim=0)
        else:
            out[k] = np.concatenate(
                [np.asarray(a, dtype=object), np.asarray(b, dtype=object)],
                axis=0
            )
    return out


def idx_select(x, idx):
    """
    Index helper that supports:
    - torch tensor indexed by torch idx/mask
    - numpy/object array indexed by numpy idx/mask (convert torch->numpy when needed)
    """
    if isinstance(x, torch.Tensor):
        return x[idx]

    x_np = np.asarray(x, dtype=object)
    if isinstance(idx, torch.Tensor):
        if idx.dtype == torch.bool:
            idx = idx.cpu().numpy().astype(bool)
        else:
            idx = idx.cpu().numpy()
    return x_np[idx]


def slice_data(data: dict, idx):
    """Slice every key in a dict using idx (torch idx/mask)."""
    return {k: idx_select(v, idx) for k, v in data.items()}


def filter_dict(data: dict, mask):
    """Alias for slice_data, but semantically used for boolean masks."""
    return slice_data(data, mask)

@dataclass
class DatasetInfo:
    name: str
    category: str
    path: Path
    is_signal: bool
    xsec: float = 1.0
    nevents: float = 1.0
    mx: float = 0.0
    my: float = 0.0


class ConfigLoader:
    def __init__(self, yaml_path: str, base_data_dir: str):
        self.yaml_path = Path(yaml_path)
        self.base_dir = Path(base_data_dir)
        self.signal_config = {}
        self.bkg_config = {}
        self._load_yaml()

    def _load_yaml(self):
        if not self.yaml_path.exists():
            logger.error(f"YAML config not found: {self.yaml_path}")
            sys.exit(1)
        with open(self.yaml_path) as f:
            raw = yaml.safe_load(f)
            self.signal_config = raw.get('signal', {})
            self.bkg_config = raw.get('background', {})

    def parse_mass(self, folder_name: str) -> Tuple[float, float]:
        match = re.search(r"MX-(\d+)_MY-(\d+)", folder_name)
        if match:
            return float(match.group(1)), float(match.group(2))
        return 0.0, 0.0

    def discover_datasets(self) -> List[DatasetInfo]:
        found_datasets = []
        # Note: Assuming structure base_dir/process_name/...
        existing_folders = [f for f in self.base_dir.iterdir() if f.is_dir()]

        # 1. Discover Backgrounds
        for name, cfg in self.bkg_config.items():
            matched = [f for f in existing_folders if name in f.name]
            if not matched:
                logger.warning(f"Background '{name}' not found in {self.base_dir}")
                continue

            target = matched[0]
            found_datasets.append(DatasetInfo(
                name=name,
                path=target,
                is_signal=False,
                xsec=cfg.get('xsec', 1.0),
                nevents=cfg.get('nEvents', 1.0),
                category=cfg.get("name", "background")
            ))

        # 2. Discover Signals
        for folder in existing_folders:
            if "MX-" in folder.name:
                mx, my = self.parse_mass(folder.name)
                found_datasets.append(DatasetInfo(
                    name=folder.name,
                    path=folder,
                    is_signal=True,
                    xsec=1.0,
                    nevents=1.0,
                    mx=mx,
                    my=my,
                    category="signal"
                ))

        logger.info(
            f"Discovered {len(found_datasets)} datasets ({len([d for d in found_datasets if d.is_signal])} Signal).")
        return found_datasets


# ==========================================
# 2. Data Management (EveNet Specific)
# ==========================================

class EveNetDatasetManager:
    def __init__(self, config_loader: ConfigLoader, parameterize: bool = False):
        self.cfg = config_loader
        self.parameterize = parameterize

    def load_data(
            self,
            datasets: List[DatasetInfo],
            split: str = "train",
            target_masses: Optional[np.ndarray] = None,
            lumi:float = 1.0
    ) -> Dict[str, Any]:
        """
        Loads .pt files for EveNet.
        Expects keys: 'x', 'globals', 'mask' (and optional 'weights').
        Returns aggregated tensors on CPU.
        """
        data_store = {
            "x": [], "globals": [], "x_mask": [],
            "y": [], "w": [], "m": [], "proc": []
        }

        # Convert target_masses once (if provided) to a CPU torch tensor [K, 2]
        target_masses_t = None
        if target_masses is not None:
            target_masses_t = torch.as_tensor(target_masses, dtype=torch.float32, device="cpu")
            if target_masses_t.ndim != 2 or target_masses_t.shape[1] != 2:
                raise ValueError(f"target_masses must have shape [K, 2], got {tuple(target_masses_t.shape)}")

        for ds in datasets:
            search_path = ds.path / "evenet" / split
            files = list(search_path.glob("*.pt"))
            if not files:
                continue

            for fp in files:
                try:
                    data = torch.load(fp, map_location="cpu")  # force CPU to avoid device mismatch
                    if "x" not in data:
                        continue

                    x_data = data["x"]
                    if not torch.is_tensor(x_data):
                        x_data = torch.as_tensor(x_data)

                    x_data = x_data.to(dtype=torch.float32, device="cpu")  # [N, M, F]
                    N = x_data.shape[0]

                    # --- globals/mask ---
                    g = data.get("global")
                    msk = data.get("x_mask")
                    if g is None or msk is None:
                        continue  # not a valid evenet sample

                    g = torch.as_tensor(g, device="cpu").to(torch.float32)
                    msk = torch.as_tensor(msk, device="cpu").to(torch.float32)

                    # --- weights ---
                    raw_w = data.get("weights", None)
                    if raw_w is None:
                        raw_w = torch.ones(N, dtype=torch.float32, device="cpu")
                    else:
                        raw_w = torch.as_tensor(raw_w, device="cpu").to(torch.float32)
                        if raw_w.ndim != 1:
                            raw_w = raw_w.view(-1)
                        if raw_w.shape[0] != N:
                            raise ValueError(f"weights length {raw_w.shape[0]} != N {N} for {fp}")

                    # avoid python floats leaking dtype/device
                    xsec = float(ds.xsec)
                    nevents = float(ds.nevents) if float(ds.nevents) != 0.0 else 1.0
                    phys_w = raw_w * (xsec * lumi / nevents)
                    # if split == "train":
                    #     phys_w = phys_w.abs()

                    # --- labels ---
                    y = torch.ones(N, dtype=torch.float32, device="cpu") if ds.is_signal else torch.zeros(N, dtype=torch.float32, device="cpu")

                    # --- mass injection: make [N, 2] float32 ---
                    if ds.is_signal:
                        mx = torch.full((N,), float(ds.mx), dtype=torch.float32, device="cpu")
                        my = torch.full((N,), float(ds.my), dtype=torch.float32, device="cpu")
                        mass_arr = torch.stack([mx, my], dim=1)  # [N, 2]
                    else:
                        if self.parameterize and split == "train":
                            if target_masses_t is None:
                                raise ValueError("Target masses required for Bkg parameterization")
                            rand_idx = torch.randint(0, target_masses_t.shape[0], (N,), device="cpu")
                            mass_arr = target_masses_t[rand_idx]  # [N, 2]
                        else:
                            mass_arr = torch.zeros((N, 2), dtype=torch.float32, device="cpu")

                    # --- store ---
                    data_store["x"].append(x_data)
                    data_store["globals"].append(g)
                    data_store["x_mask"].append(msk)
                    data_store["y"].append(y)
                    data_store["w"].append(phys_w)
                    data_store["m"].append(mass_arr)
                    data_store["proc"].append([ds.category] * N)

                except Exception as e:
                    logger.warning(f"Corrupt/bad file {fp}: {e}")

        if not data_store["x"]:
            logger.error(f"No EveNet data loaded for split {split}!")
            return {}

        # Concatenate tensors (torch-only)
        final_data: Dict[str, Any] = {}
        for k in ["x", "globals", "x_mask", "y", "w", "m"]:
            final_data[k] = torch.cat(data_store[k], dim=0)

        # proc stays strings (numpy array or list is fine)
        final_data["proc"] = np.concatenate([np.asarray(p, dtype=object) for p in data_store["proc"]])

        # Ensure globals shape (common fix: [N] -> [N, 1])
        if final_data["globals"].ndim == 1:
            final_data["globals"] = final_data["globals"].unsqueeze(1)

        return final_data

    def reweight_signals(self, data: dict, logger=None) -> dict:
        """
        Reweight signal points so each unique (mx,my) has equal total weight.
        Expects:
          - data['w']: torch.Tensor [N]
          - data['m']: torch.Tensor [N,2]
        """
        if "m" not in data or "w" not in data:
            return data

        w = data["w"]
        m = data["m"]

        if not (isinstance(w, torch.Tensor) and isinstance(m, torch.Tensor)):
            raise TypeError("Expected data['w'] and data['m'] to be torch.Tensor")

        if m.numel() == 0:
            return data

        # Unique mass points [K,2]
        unique_masses = torch.unique(m, dim=0)
        K = unique_masses.shape[0]
        if K == 0:
            return data

        target_w = w.sum() / K
        if logger is not None:
            logger.info(f"Reweighting {K} mass points to target weight {target_w.item():.2e}")

        # Loop over unique mass points (K is typically small)
        for i in range(K):
            mx = unique_masses[i, 0]
            my = unique_masses[i, 1]
            mask = (m[:, 0] == mx) & (m[:, 1] == my)
            current_sum = w[mask].sum()
            if current_sum > 0:
                w[mask] = w[mask] * (target_w / current_sum)

        data["w"] = w
        return data


# ==========================================
# 3. Plotting Helpers (accept torch or numpy; convert internally)
# ==========================================

def plot_score_overlay(y_eval, y_pred, w_eval, p_eval, fname=None):
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
            bkg_labels.append(proc)

    plt.figure(figsize=(10, 7))
    bins = np.linspace(0, 1, 40)

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

# ==========================================
# 4. Execution Flow
# ==========================================

def prepare_evenet_features(data_dict, parameterize=False):
    """
    Organize dict for EveNetLite runner.
    If parameterize=True:
      - provide 'params' for callbacks
      - concatenate params into globals so the network can see them
    """
    feats = {
        "x": data_dict["x"],
        "globals": data_dict["globals"],
        "x_mask": data_dict["x_mask"],
    }

    if parameterize:
        feats["params"] = data_dict["m"]
        feats["globals"] = torch.cat([data_dict["globals"], data_dict["m"]], dim=1)

    return feats


def run_pipeline(args):
    if not HAS_EVENET:
        return

    mode_str = "parametrized" if args.parameterize else "individual"
    mass_target = "All" if args.parameterize else f"MX-{args.mX}_MY-{args.mY}"
    model_str = "evenet-pretrain" if args.pretrain else "evenet-scratch"
    out_dir = Path(args.out_dir) / model_str / mode_str / mass_target
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = out_dir / "checkpoints"
    ckpt_dir.mkdir(exist_ok=True)

    # ---- config & discovery ----
    cfg = ConfigLoader(args.yaml_path, args.base_dir)
    all_datasets = cfg.discover_datasets()

    if args.parameterize:
        if args.param_mx_step is not None and args.param_my_step is not None:
            # Filter signal datasets based on step size
            mx_set = set([d.mx for d in all_datasets if d.is_signal])
            my_set = set([d.my for d in all_datasets if d.is_signal])
            mx_sorted = sorted(mx_set)
            my_sorted = sorted(my_set)
            mx_filtered = mx_sorted[::args.param_mx_step]
            my_filtered = my_sorted[::args.param_my_step]
            sig_datasets_eval = [d for d in all_datasets if d.is_signal]
            sig_datasets_train = [d for d in all_datasets if d.is_signal and d.mx in mx_filtered and d.my in my_filtered]
        else:
            sig_datasets_train = [d for d in all_datasets if d.is_signal]
            sig_datasets_eval = sig_datasets_train
    else:
        sig_datasets_train = [d for d in all_datasets if d.is_signal and d.mx == args.mX and d.my == args.mY]
        sig_datasets_eval = sig_datasets_train
        if not sig_datasets_train or not sig_datasets_eval:
            logger.error(f"Signal MX={args.mX}, MY={args.mY} not found!")
            raise SystemExit(1)

    bkg_datasets = [d for d in all_datasets if not d.is_signal]

    # Target masses for background injection (torch tensor [K,2])
    target_masses = torch.tensor(
        [[float(d.mx), float(d.my)] for d in all_datasets if d.is_signal],
        dtype=torch.float32
    )

    # ---- load training data ----
    dm = EveNetDatasetManager(cfg, parameterize=args.parameterize)

    logger.info(">>> Loading Signal (Train)...")
    d_sig_tr = dm.load_data(sig_datasets_train, "train", lumi=args.lumi)
    d_sig_tr = dm.reweight_signals(d_sig_tr, logger=logger)

    logger.info(">>> Loading Background (Train)...")
    # Pass numpy if your loader expects numpy; otherwise pass torch and convert inside loader
    d_bkg_tr = dm.load_data(bkg_datasets, "train", target_masses=target_masses.cpu().numpy(), lumi=args.lumi)

    # ---- global balance: scale background to match total signal weight ----
    sig_sum = d_sig_tr["w"].sum()
    bkg_sum = d_bkg_tr["w"].sum()

    if bkg_sum > 0:
        num_bkg = d_bkg_tr["w"].shape[0]

        d_bkg_tr["w"] = d_bkg_tr["w"] * (num_bkg / bkg_sum)
        d_sig_tr["w"] = d_sig_tr["w"] * (num_bkg / sig_sum)

    # ---- merge for training ----
    keys_to_merge = ["x", "globals", "x_mask", "y", "w", "m", "proc"]
    train_data = concat_ds(d_bkg_tr, d_sig_tr, keys_to_merge)

    # ---- shuffle & split (torch indices for tensors, converted for proc) ----
    N_full = train_data["y"].shape[0]
    indices = torch.randperm(N_full)
    split_idx = int(N_full * 0.8)
    train_idx = indices[:split_idx]
    val_idx = indices[split_idx:]

    d_train = slice_data(train_data, train_idx)
    d_val = slice_data(train_data, val_idx)

    # No negative loss in training data
    d_train["w"] = d_train["w"].abs()
    # d_train = filter_dict(d_train, d_train["w"] > 0)
    # ---- features ----
    train_features = prepare_evenet_features(d_train, args.parameterize)
    val_features = prepare_evenet_features(d_val, args.parameterize)

    global_dim = train_features["globals"].shape[1]

    # ---- callbacks ----
    callbacks = []
    if args.parameterize:
        m_vals = d_train["m"]
        min_vals = m_vals.min(dim=0).values.tolist()
        max_vals = m_vals.max(dim=0).values.tolist()
        logger.info(f"Adding ParameterRandomizationCallback: Min={min_vals}, Max={max_vals}")
        callbacks.append(ParameterRandomizationCallback(min_values=min_vals, max_values=max_vals, pool_from_signal=True))

    # ---- feature names ----
    feature_names = {
        "x": [f"feat_{i}" for i in range(train_features["x"].shape[2])],
        "globals": [f"glob_{i}" for i in range(global_dim)],
    }

    normalize_pt = "norm/normalization_pretrain.pt"
    normalize_dict = torch.load(normalize_pt, map_location="cpu")
    normalization_stats = {
        "x": {
            "mean": normalize_dict["input_mean"]["Source"],  # len == num object features
            "std": normalize_dict["input_std"]["Source"],
        },
        "globals": {
            "mean": normalize_dict["input_mean"]["Conditions"],  # len == num global features
            "std": normalize_dict["input_std"]["Conditions"],
        },
    }

    learning_rate = args.learning_rate if hasattr(args, 'learning_rate') else 1e-3
    learning_rates = [learning_rate] if not args.pretrain else [0.1 * learning_rate, 0.3 * learning_rate, learning_rate]
    module_lists = ["Classification", "ObjectEncoder", "PET", "GlobalEmbedding"] if not args.pretrain else [["PET"], ["ObjectEncoder",  "GlobalEmbedding"], ["Classification"]]
    weight_decay = [1e-4 for x in learning_rates]
    # ---- train ----
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size > 1 and torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    logger.info(f">>> Starting EveNet Training on GPU: {local_rank} [of World: {world_size}]")

    classifier = run_evenet_lite_training(
        train_features=train_features,
        train_labels=d_train["y"],
        train_weights=d_train["w"],

        val_features=val_features,
        val_labels=d_val["y"],
        val_weights=d_val["w"],

        class_labels=["background", "signal"],
        global_input_dim=global_dim,
        # feature_names=feature_names,
        callbacks=callbacks,

        epochs=args.epochs,
        batch_size=args.batch_size,
        # sampler="weighted",

        checkpoint_path=str(ckpt_dir),
        save_top_k=1,
        monitor_metric="val_loss",
        sic_min_bkg_events = 10,
        normalization_stats = normalization_stats,
        use_wandb=True,
        wandb = {
            'project': 'EveNet-GridSearch',
            'name': f"{model_str}-{mode_str}-{mass_target}",
            'entity': "ytchou97-university-of-washington",
            'save_dir':"/pscratch/sd/t/tihsu/tmp/wandb"
        },
        pretrained=args.pretrain,
        pretrained_path="/global/cfs/cdirs/m5019/avencast/Checkpoints/checkpoints.20M.ablation.4.newcls/last.ckpt",

        module_lists = module_lists,
        lr = learning_rates,
        weight_decay=weight_decay,
        early_stop_patience=3,
        n_ensemble=args.ensemble,
        loss_gamma=args.gamma
    )

    # ---- evaluation data ----
    def is_rank_zero():
        return (not dist.is_available()) or (not dist.is_initialized()) or dist.get_rank() == 0
    logger.info(">>> Loading Test Data...")
    d_sig_te = dm.load_data(sig_datasets_eval, "valid", lumi=args.lumi)
    d_sig_te = dm.reweight_signals(d_sig_te, logger=logger)
    d_bkg_te = dm.load_data(bkg_datasets, "valid", lumi=args.lumi)

    # Unique mass points as torch [K,2]
    unique_masses = torch.unique(d_sig_te["m"], dim=0)

    for i in range(unique_masses.shape[0]):
        mx = unique_masses[i, 0]
        my = unique_masses[i, 1]

        # ---- signal subset by mass (torch mask) ----
        mask_s = (d_sig_te["m"][:, 0] == mx) & (d_sig_te["m"][:, 1] == my)
        sub_sig = filter_dict(d_sig_te, mask_s)

        # ---- background copy (avoid in-place edits of the original) ----
        sub_bkg = {}
        for k, v in d_bkg_te.items():
            if isinstance(v, torch.Tensor):
                sub_bkg[k] = v.clone()
            else:
                sub_bkg[k] = np.asarray(v, dtype=object).copy()

        # Inject parameters into background for parametrized inference
        if args.parameterize:
            N_b = sub_bkg["y"].shape[0]
            sub_bkg["m"] = torch.stack(
                [
                    torch.full((N_b,), mx.item(), dtype=torch.float32),
                    torch.full((N_b,), my.item(), dtype=torch.float32),
                ],
                dim=1
            )

        # ---- merge eval set ----
        eval_data = concat_ds(sub_bkg, sub_sig, keys_to_merge)

        # ---- features ----
        eval_features = prepare_evenet_features(eval_data, args.parameterize)

        # ---- predict ----
        print(sub_sig["y"][:10], sub_bkg["y"][:10])
        logits = classifier.predict(eval_features, batch_size=args.batch_size * 8)

        if not is_rank_zero():
            continue

        probs = torch.softmax(logits, dim=1)
        y_pred = probs[:, 1].detach().cpu().numpy()

        # ---- metrics inputs ----
        y_eval = eval_data["y"].detach().cpu().numpy()
        w_eval = eval_data["w"].detach().cpu().numpy()
        p_eval = eval_data["proc"]  # numpy/object

        metrics = calculate_physics_metrics(
            y_pred, y_eval, w_eval, training=False,
            min_bkg_events=10,
            log_plots=True,
            f_name=out_dir / f"sic_plots_MX-{int(round(mx.item()))}_MY-{int(round(my.item()))}.png"
        )

        key = f"MX-{int(round(mx.item()))}_MY-{int(round(my.item()))}"
        logger.info(f"Mass {key}: AUC={metrics['auc']:.4f}, Max SIC={metrics['max_sic']:.4f}")

        # ---- plots ----
        plot_score_overlay(
            y_eval=y_eval,
            y_pred=y_pred,
            w_eval=w_eval,
            p_eval=p_eval,
            fname=out_dir / f"score_{key}.png",
        )

        # ---- save metrics ----
        results = {
            "auc": float(metrics["auc"]),
            "max_sic": float(metrics["max_sic"]),
            "max_sic_unc": float(metrics["max_sic_unc"]),
        }
        with open(out_dir / f"metrics_{key}.json", "w") as f:
            json.dump(results, f, indent=4)

    logger.info(f"Done. Results saved to {out_dir}")

# ==========================================
# 5. Entry Point
# ==========================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="EveNet Grid Search Trainer")

    # Data Selection
    parser.add_argument("--base_dir", type=str, default="/pscratch/sd/t/tihsu/database/GridStudy_v2")
    parser.add_argument("--yaml_path", type=str, default="sample.yaml")
    parser.add_argument("--mX", type=float, default=None)
    parser.add_argument("--mY", type=float, default=None)

    # Model Config
    parser.add_argument("--parameterize", action="store_true", help="Include Mass as input")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=512)

    # IO
    parser.add_argument("--out_dir", type=str, default="results")
    parser.add_argument("--pretrain", action="store_true", help="Use pretrained model weights")
    parser.add_argument("--learning_rate", type=float, default=1e-3, help="Learning rate for training")
    parser.add_argument("--param-mx-step", type=int, default=None)
    parser.add_argument("--param-my-step", type=int, default=None)
    parser.add_argument("--lumi", type=float, default=36000)

    parser.add_argument("--ensemble", type=int, default=1, help="Number of ensemble models to train")
    parser.add_argument("--gamma", type=float, default=1.0, help="gamma for focal loss" )
    args = parser.parse_args()

    if not args.parameterize and (args.mX is None or args.mY is None):
        parser.error("Specify -mX and -mY, or use --parameterize for mass parameterization.")

    run_pipeline(args)