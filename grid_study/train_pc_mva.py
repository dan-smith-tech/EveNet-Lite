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
import random

seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)   # if using CUDA

from evenet.network.metrics.assignment import shared_epoch_end

try:
    from evenet_lite import run_evenet_lite_training, EvenetLiteClassifier
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
from shared_metrics import plot_score_overlay


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

            cutflow_json_first = target / "../cutflow.json"
            if cutflow_json_first.exists():
                with open(cutflow_json_first, "r") as f:
                    cutflow = json.load(f)
                nevents = cutflow[name].get("total", None)
                if nevents is None:
                    raise ValueError(f"Missing 'all' in {cutflow_json_first}")
                nevents = float(nevents)
                if nevents == 0.0:
                    nevents = 1.0
                print("Using cutflow from", cutflow_json_first)
            else:
                cutflow_json = target / "cutflow.json"
                if cutflow_json.exists():
                    with open(cutflow_json, "r") as f:
                        cutflow = json.load(f)
                    nevents = cutflow.get("all", None)
                    if nevents is None:
                        raise ValueError(f"Missing 'all' in {cutflow_json}")
                    nevents = float(nevents)
                    print("Using cutflow from", cutflow_json)
                    if nevents == 0.0:
                        nevents = 1.0
                else:
                    nevents = cfg.get('nEvent', 1.0)

            found_datasets.append(DatasetInfo(
                name=name,
                path=target,
                is_signal=False,
                xsec=cfg.get('xsec', 1.0),
                nevents=nevents,
                category=cfg.get("name", "background")
            ))

        # 2. Discover Signals
        for folder in existing_folders:
            if "MX-" in folder.name:
                mx, my = self.parse_mass(folder.name)
                cutflow_json = folder / "cutflow.json"
                cutflow_json_first = folder / "../cutflow.json"
                if cutflow_json_first.exists():
                    with open(cutflow_json_first, "r") as f:
                        cutflow = json.load(f)
                    nevents = cutflow[folder.name].get("total", None)
                    if nevents is None:
                        raise ValueError(f"Missing 'all' in {cutflow_json_first}")
                    nevents = float(nevents)
                    if nevents == 0.0:
                        nevents = 1.0
                    print("Using cutflow from", cutflow_json_first)
                else:
                    cutflow_json = folder / "cutflow.json"
                    if cutflow_json.exists():
                        with open(cutflow_json, "r") as f:
                            cutflow = json.load(f)
                        nevents = cutflow.get("all", None)
                        if nevents is None:
                            raise ValueError(f"Missing 'all' in {cutflow_json}")
                        nevents = float(nevents)
                        if nevents == 0.0:
                            nevents = 1.0
                    else:
                        nevents = 184000.0
                found_datasets.append(DatasetInfo(
                    name=folder.name,
                    path=folder,
                    is_signal=True,
                    xsec=0.01, # 10 fb
                    nevents=nevents,  # TODO: make configurable, now hardcoded
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
            lumi: float = 1.0,
            max_entries = None
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
            files = sorted(list(search_path.glob("*.pt")))
            if not files:
                continue
            for fp in files:
                try:
                    data = torch.load(fp, map_location="cpu", weights_only=False)  # force CPU to avoid device mismatch
                    if "x" not in data:
                        print("x not in data")
                        continue

                    x_data = data["x"]
                    if not torch.is_tensor(x_data):
                        x_data = torch.as_tensor(x_data)

                    x_data = x_data.to(dtype=torch.float32, device="cpu")  # [N, M, F]
                    N = x_data.shape[0]

                    # --- globals/mask ---
                    g = data.get("global")
                    if g is None:
                        g = data.get("globals")
                    msk = data.get("x_mask")
                    if g is None or msk is None:
                        print("globals or mask missing")
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
                    phys_w = raw_w * (xsec * lumi / nevents) * 2 #ratio split for 2
                    # if split == "train":
                    #     phys_w = phys_w.abs()

                    # --- labels ---
                    y = torch.ones(N, dtype=torch.float32, device="cpu") if ds.is_signal else torch.zeros(N,dtype=torch.float32, device="cpu")

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

        if max_entries is not None:
            # random get max entries
            N = final_data["x"].shape[0]
            max_n = min(int(max_entries), N)

            # sample indices (no replacement)
            idx = torch.randperm(N, device=final_data["x"].device)[:max_n]

            # torch tensors
            for k in ["x", "globals", "x_mask", "y", "w", "m"]:
                v = final_data[k]
                # if some tensor is on CPU and idx on GPU (or vice versa), move idx
                if isinstance(v, torch.Tensor) and v.device != idx.device:
                    idx_use = idx.to(v.device)
                else:
                    idx_use = idx
                final_data[k] = v.index_select(0, idx_use)

            # proc (numpy/object array)
            final_data["proc"] = final_data["proc"][idx.cpu().numpy()]


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
        # feats["globals"] = torch.cat([data_dict["globals"], data_dict["m"]], dim=1)

    return feats


def run_pipeline(args):
    if not HAS_EVENET:
        return

    if args.parameterize:
        mode_str = f"parametrized_reduce_factor_x_{args.param_mx_step}_y_{args.param_my_step}"
    else:
        mode_str = "individual"
    mass_target = "All" if args.parameterize else f"MX-{args.mX}_MY-{args.mY}"
    model_str = "evenet-pretrain" if args.pretrain else "evenet-scratch"
    out_dir = Path(args.out_dir) / model_str / mode_str / mass_target
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = out_dir / "checkpoints"
    ckpt_dir.mkdir(exist_ok=True)
    if args.continue_training:
        ckpt_old_dir = out_dir / "checkpoints_old"
        ckpt_old_dir.mkdir(exist_ok=True)
        os.system(f"cp {ckpt_dir}/*.pt {ckpt_old_dir}/.")

    if args.in_dir is not None:
        load_dir = Path(args.in_dir) / model_str / mode_str / mass_target
    else:
        load_dir = out_dir

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
            sig_datasets_train = [d for d in all_datasets if
                                  d.is_signal and d.mx in mx_filtered and d.my in my_filtered]
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
    ) # doesn't matter because we will do signal mass dynamic sampling in the trainer

    # ---- load training data ----
    dm = EveNetDatasetManager(cfg, parameterize=args.parameterize)

    classifier = None
    logger.info(">>> Loading Signal (Train)...")
    d_sig_tr = dm.load_data(sig_datasets_train, "train", lumi=args.lumi)
    d_sig_tr = dm.reweight_signals(d_sig_tr, logger=logger)

    logger.info(">>> Loading Background (Train)...")
    # Pass numpy if your loader expects numpy; otherwise pass torch and convert inside loader
    d_bkg_tr = dm.load_data(bkg_datasets, "train", target_masses=target_masses.cpu().numpy(), lumi=args.lumi, max_entries=args.max_bkg_entries)

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
    
    # Create a dedicated generator with a FIXED seed for splitting
    # This ensures Rank 0, 1, 2, 3 all generate the EXACT SAME indices
    g_split = torch.Generator()
    g_split.manual_seed(12345)  # Hardcoded seed for data splitting consistency
    
    indices = torch.randperm(N_full, generator=g_split)
    
    split_idx = int(N_full * 0.8)
    train_idx = indices[:split_idx]
    val_idx = indices[split_idx:]
    # ---------------------------------------------

    d_train = slice_data(train_data, train_idx)
    d_val = slice_data(train_data, val_idx)

    # No negative loss in training data
    d_train["w"] = d_train["w"].abs()
    # d_train = filter_dict(d_train, d_train["w"] > 0)
    # ---- features ----
    train_features = prepare_evenet_features(d_train, args.parameterize)
    val_features = prepare_evenet_features(d_val, args.parameterize)

    global_dim = train_features["globals"].shape[1]
    if args.parameterize:
        global_dim += train_features["params"].shape[1]

    # ---- feature names ----
    feature_names = {
        "x": ['energy', 'pt', 'eta', 'phi', 'isBTag', 'isLepton', 'Charge'],
        "globals": ['met', 'met_phi', 'nLepton', 'nbJet', 'nJet', 'HT', 'HT_lep', 'M_all', 'M_leps', 'M_bjets'],
    }
    if args.parameterize:
        feature_names["params"] = ['feature_0', 'feature_1']

    if "train" in args.stage:
        # ---- callbacks ----
        callbacks = []
        if args.parameterize:
            m_vals = d_train["m"]
            min_vals = m_vals.min(dim=0).values.tolist()
            max_vals = m_vals.max(dim=0).values.tolist()
            logger.info(f"Adding ParameterRandomizationCallback: Min={min_vals}, Max={max_vals}")
            callbacks.append(
                ParameterRandomizationCallback(min_values=min_vals, max_values=max_vals, pool_from_signal=True))

        ##########################
        ## Normalization Rules  ##
        ##########################

        normalize_pt = "norm/normalization_pretrain.pt"
        normalize_dict = torch.load(normalize_pt, map_location="cpu", weights_only=False)
        normalization_stats = {
            "x": {
                "mean": normalize_dict["input_mean"]["Source"],  # len == num object features
                "std": normalize_dict["input_std"]["Source"],
            },
            "globals": {
                "mean": normalize_dict["input_mean"]["Conditions"],  # len == num global features
                "std": normalize_dict["input_std"]["Conditions"],
            }
        }

        normalization_rules = {
            "x": {
                "energy": "log_normalize",
                "pt": "log_normalize",
                "eta": "normalize",
                "phi": "normalize_uniform"
            },
            "globals": {
                "met": "log_normalize",
                "met_phi": "normalize",
                "HT": "log_normalize",
                "HT_lep": "log_normalize",
                "M_all": "log_normalize",
                "M_leps": "log_normalize",
                "M_bjets": "log_normalize"
            },
        }
        if args.parameterize:
            normalization_stats["params"] = {
                "mean": d_train["m"].mean(axis=1),
                "std": d_train["m"].std(axis=1)
            }
            normalization_rules["params"] = {
                "feature_0": "normalize",
                "feature_1": "normalize"
            }

        learning_rate = args.learning_rate if hasattr(args, 'learning_rate') else 1e-3

        body_frozen_factor = 0.1 if args.freeze_type == "partial" else 0.0 if args.freeze_type == "all" else 1.0
        if args.use_adapter:
            body_frozen_factor = 0.3
        mediate_frozen_factor = 0.3 if args.freeze_type == "partial" else 0.1 if args.freeze_type == "all" else 1.0
        learning_rates = [learning_rate] if not args.pretrain else [body_frozen_factor * learning_rate, mediate_frozen_factor * learning_rate,
                                                                    learning_rate]

        module_lists = [["Classification", "ObjectEncoder", "PET", "GlobalEmbedding"]] if not args.pretrain else [
            ["PET"], ["ObjectEncoder"], ["GlobalEmbedding", "Classification"]]

        learning_rates_new = []
        module_lists_new = []

        for lr, ml in zip(learning_rates, module_lists):
            if lr < 1e-10:
                continue
            else:
                learning_rates_new.append(lr)
                module_lists_new.append(ml)

        learning_rates = learning_rates_new
        module_lists = module_lists_new



        print("learning for each module:", learning_rates, module_lists)
        weight_decay = [1e-2  for x in learning_rates]
        if args.pretrain:
            weight_decay[0] = 1e-4 # Do not let weight decay to kill pretrain weight
            weight_decay[1] = 1e-4 # Do not let weight decay to kill pretrain weight
        # ---- train ----
        world_size = int(os.environ.get("WORLD_SIZE", "1"))
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        if world_size > 1 and torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
        logger.info(f">>> Starting EveNet Training on GPU: {local_rank} [of World: {world_size}]")

        best_ckpt = None
        if args.continue_training:
            # load lastest checkpoint from ckpt_dir
            ckpt_files = list(ckpt_dir.glob("*.pt"))
            if not ckpt_files:
                logger.error(f"No checkpoints found in {ckpt_dir} for prediction!")
                raise SystemExit(1)
            best_ckpt = max(ckpt_files, key=os.path.getctime)

        start_time = time.time()
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
            sampler=args.sampler,

            checkpoint_path=str(ckpt_dir),
            resume_from=best_ckpt,
            save_top_k=1,
            monitor_metric="val_loss",
            sic_min_bkg_events=10,
            normalization_rules=normalization_rules if args.parameterize else None,
            normalization_stats=normalization_stats,
            use_wandb=True,
            wandb={
                'project': 'EveNet-GridSearch',
                'name': f"{model_str}-{mode_str}-{mass_target}{'-test' if args.wandb_test else ''}{args.wandb_tag}",
                'entity': "ytchou97-university-of-washington",
                'dir': "/pscratch/sd/t/tihsu/tmp/wandb"
            },
            pretrained=args.pretrain,
            pretrained_path="/global/cfs/cdirs/m5019/avencast/Checkpoints/checkpoints.20M.ablation.4.newcls/last.ckpt",
            pretrained_source="local",
            module_lists=module_lists,
            lr=learning_rates,
            weight_decay=weight_decay,
            early_stop_patience=args.early_stop,
            n_ensemble=args.ensemble,
            loss_gamma=args.gamma,
            use_adapter=args.use_adapter,
            use_peft=args.use_adapter
        )
        end_time = time.time()

        fitting_time = end_time - start_time
        logger.info(f"Training completed in {fitting_time / 60:.2f} minutes.")

        if (not dist.is_available()) or ((not dist.is_initialized()) or dist.get_rank() == 0):
            training_log = {
                "time": fitting_time,
                **vars(args)
            }
            with open(out_dir / f"training_log.json", "w") as f:
                json.dump(training_log, f, indent=4)
    predict_value = None
    if "predict" in args.stage:
        if classifier is None:
            classifier = EvenetLiteClassifier(
                class_labels=["background", "signal"],
                global_input_dim=global_dim,
                n_ensemble=args.ensemble,
                use_adapter=args.use_adapter
            )
            # load lastest checkpoint from ckpt_dir
            ckpt_files = list(ckpt_dir.glob("*.pt"))
            if not ckpt_files:
                logger.error(f"No checkpoints found in {ckpt_dir} for prediction!")
                raise SystemExit(1)
            best_ckpt = max(ckpt_files, key=os.path.getctime)
            classifier.load_checkpoint(best_ckpt, feature_names=feature_names)

        # ---- evaluation data ----
        def is_rank_zero():
            return (not dist.is_available()) or (not dist.is_initialized()) or dist.get_rank() == 0

        logger.info(">>> Loading Test Data...")
        d_sig_te = dm.load_data(sig_datasets_eval, "valid", lumi=args.lumi)
        # d_sig_te = dm.reweight_signals(d_sig_te, logger=logger)
        d_bkg_te = dm.load_data(bkg_datasets, "valid", lumi=args.lumi)

        # Unique mass points as torch [K,2]
        unique_masses = torch.unique(d_sig_te["m"], dim=0)

        for i in range(unique_masses.shape[0]):
            mx = unique_masses[i, 0]
            my = unique_masses[i, 1]

            if args.mX is not None:
                if int(mx.item()) != int(args.mX):
                    continue
            if args.mY is not None:
                if int(my.item()) != int(args.mY):
                    continue

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

            predict_value = {
                "y_true": y_eval.tolist(),
                "y_pred": y_pred.tolist(),
                "w": w_eval.tolist(),
                "proc": p_eval.tolist(),
                "mx": mx.item(),
                "my": my.item(),
            }

            with open(out_dir / f"predictions_MX-{int(round(mx.item()))}_MY-{int(round(my.item()))}.json", "w") as f:
                json.dump(predict_value, f, indent=4)

    if "evaluate" in args.stage:
        def is_rank_zero():
            return (not dist.is_available()) or (not dist.is_initialized()) or dist.get_rank() == 0

        all_masses = [(d_sig.mx, d_sig.my) for d_sig in sig_datasets_eval]
        for mx_val, my_val in all_masses:
            if args.mX is not None:
                if int(mx_val) != int(args.mX):
                    continue
            if args.mY is not None:
                if int(my_val) != int(args.mY):
                    continue

            with open(load_dir / f"predictions_MX-{int(round(mx_val))}_MY-{int(round(my_val))}.json", "r") as f:
                predict_value = json.load(f)

            if not is_rank_zero():
                continue

            y_eval = np.array(predict_value["y_true"])
            y_pred = np.array(predict_value["y_pred"])
            w_eval = np.array(predict_value["w"])
            p_eval = np.array(predict_value["proc"])

            nevents_by_name = {ds.category: ds.nevents if ds.category != 'signal' else 1.0 for ds in all_datasets }
            nevents_eval = np.array([nevents_by_name[p] for p in p_eval])
            # w_eval = w_eval / nevents_eval

            mx = predict_value["mx"]
            my = predict_value["my"]
            # ---- metrics ----

            metrics = calculate_physics_metrics(
                y_pred, y_eval, w_eval, training=False,
                min_bkg_events=10,
                log_plots=True,
                bins=1000,
                min_bkg_ratio=0.0001,
                f_name=str(out_dir / f"sic_plots_MX-{int(round(mx_val))}_MY-{int(round(my_val))}.png"),
                Zs=10,
                Zb=5,
                min_bkg_per_bin=3,
                min_mc_stats=0.2,
                include_signal_in_stat=False,
                # logger=logger,
            )

            key = f"MX-{int(round(mx_val))}_MY-{int(round(my_val))}"
            logger.info(
                f"Mass {key}: AUC={metrics['auc']:.4f}, Max SIC={metrics['max_sic']:.4f}, Bin SIG={metrics['trafo_bin_sig']:.4f}")

            # ---- plots ----

            plot_score_overlay(
                y_eval=y_eval,
                w_eval=w_eval,
                p_eval=p_eval,
                y_pred=y_pred,
                fname = out_dir / f"score_uniform_binning_MX-{int(mx)}_MY-{int(my)}.png"
            )
            plot_score_overlay(
                y_eval=y_eval,
                y_pred=y_pred,
                w_eval=w_eval,
                p_eval=p_eval,
                bins=metrics['trafo_edge'],
                uniform_bin_plot=True,
                fname = out_dir / f"score_auto_binning_flat_MX-{int(mx)}_MY-{int(my)}.png"
            )
            plot_score_overlay(
                y_eval=y_eval,
                y_pred=y_pred,
                w_eval=w_eval,
                p_eval=p_eval,
                bins=metrics['trafo_edge'],
                uniform_bin_plot=False,
                fname = out_dir / f"score_auto_binning_MX-{int(mx)}_MY-{int(my)}.png"
            )

            # ---- save metrics ----
            results = {
                "auc": float(metrics["auc"]),
                "max_sic": float(metrics["max_sic"]),
                "max_sic_unc": float(metrics["max_sic_unc"]),
                "trafo_bin_sig": float(metrics["trafo_bin_sig"]),
                "sic": metrics["sic"].tolist(),
                "sic_unc": metrics["sic_unc"].tolist(),
                "trafo_edge": metrics["trafo_edge"].tolist(),
                # "fitting_time": end_time - start_time,
            }
            with open(out_dir / f"eval_metrics_{key}.json", "w") as f:
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
    parser.add_argument("--sampler", type=str, default=None)

    # IO
    parser.add_argument("--out_dir", type=str, default="results")
    parser.add_argument("--in_dir", type=str, default=None, help="input directory that differs from out_dir")
    parser.add_argument("--wandb_tag", type=str, default="")
    parser.add_argument("--pretrain", action="store_true", help="Use pretrained model weights")
    parser.add_argument("--learning_rate", type=float, default=1e-3, help="Learning rate for training")
    parser.add_argument("--param-mx-step", type=int, default=1)
    parser.add_argument("--param-my-step", type=int, default=1)
    parser.add_argument("--lumi", type=float, default=300000)
    parser.add_argument("--early_stop", type=int, default=5)

    parser.add_argument("--ensemble", type=int, default=1, help="Number of ensemble models to train")
    parser.add_argument("--gamma", type=float, default=0.0, help="gamma for focal loss" )

    parser.add_argument("--stage", type=str, default=["train", "predict", "evaluate"], nargs="+",
                        help="Pipeline stages to run")
    parser.add_argument("--freeze_type", type=str, default="partial", choices=["none", "partial", "all"],)
    parser.add_argument("--max_bkg_entries", type=int, default=None, help="Max entries to load for training/testing")
    # logging
    parser.add_argument("--wandb_test", action="store_true")
    parser.add_argument("--use_adapter", action="store_true")
    parser.add_argument("--continue_training", action="store_true")
    args = parser.parse_args()

    if not args.parameterize and (args.mX is None or args.mY is None):
        parser.error("Specify -mX and -mY, or use --parameterize for mass parameterization.")

    run_pipeline(args)
