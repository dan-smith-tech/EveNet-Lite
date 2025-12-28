import sys
import os
import argparse
import re
import json
import yaml
import numpy as np
import logging
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
from pathlib import Path
import time

import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from shared_metrics import plot_score_overlay
# --- Optional Imports ---
try:
    from tabpfn import TabPFNClassifier

    HAS_TABPFN = True
except ImportError:
    HAS_TABPFN = False

import matplotlib.pyplot as plt
try:
    import mplhep as hep
    plt.style.use(hep.style.CMS)
except ImportError:
    print("mplhep not found, using default style")

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

# --- Physics Metric Fallback ---
try:
    from evenet_lite.metrics import calculate_physics_metrics
except ImportError as e:
    logger.warning("evenet_lite not found, using simplified physics metrics.")
    print("error", e)
    def calculate_physics_metrics(probs, targets, weights):
        return {
            'max_sic_unc': 0.0, 'max_sic': 0.0,
            'auc': roc_auc_score(targets, probs, sample_weight=weights)
        }


# ==========================================
# 1. Configuration & Data Structures
# ==========================================

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
        """Extracts MX, MY from folder name strings like 'MX-900_MY-100'."""
        match = re.search(r"MX-(\d+)_MY-(\d+)", folder_name)
        if match:
            return float(match.group(1)), float(match.group(2))
        return 0.0, 0.0

    def discover_datasets(self) -> List[DatasetInfo]:
        """
        Scans base_dir and matches against YAML definitions.
        Returns a concrete list of DatasetInfo objects.
        """
        found_datasets = []
        existing_folders = [f for f in self.base_dir.iterdir() if f.is_dir()]

        # 1. Discover Backgrounds (Explicit Match)
        for name, cfg in self.bkg_config.items():
            matched = [f for f in existing_folders if name in f.name]

            if not matched:
                logger.warning(f"Background '{name}' not found in {self.base_dir}")
                continue

            # Assume strict 1-to-1 match for background definitions in YAML
            target = matched[0]
            found_datasets.append(DatasetInfo(
                name=name,
                path=target,
                is_signal=False,
                xsec=cfg.get('xsec', 1.0),
                nevents=cfg.get('nEvent', 1.0),
                category=cfg.get("name", "background")
            ))

        # 2. Discover Signals (Regex Pattern Match)
        # Convert glob wildcard to rough regex for safety or just use string check
        # Here we rely on the specific format "NMSSM..."
        for folder in existing_folders:
            if "MX-" in folder.name:
                mx, my = self.parse_mass(folder.name)
                # Optional: Filter by YAML range if needed
                # mx_range = self.signal_config.get('mx', [0, 99999])
                # if not (mx_range[0] <= mx <= mx_range[1]): continue

                found_datasets.append(DatasetInfo(
                    name=folder.name,
                    path=folder,
                    is_signal=True,
                    xsec=1.0,  # Signal often normalized differently, usually 1.0 relative
                    nevents=184000,  #TODO: make configurable, now hardcoded
                    mx=mx,
                    my=my,
                    category="signal"
                ))

        logger.info(
            f"Discovered {len(found_datasets)} datasets ({len([d for d in found_datasets if d.is_signal])} Signal).")
        return found_datasets


# ==========================================
# 2. Data Management
# ==========================================

class DatasetManager:
    def __init__(self, config_loader: ConfigLoader, parameterize: bool = False, features: List[str] = None):
        self.cfg = config_loader
        self.parameterize = parameterize
        self.features = features
        self.feature_indices = None
        self.feature_names_loaded = None

    def load_data(self, datasets: List[DatasetInfo], split: str = "train",
                  target_masses: Optional[np.ndarray] = None, lumi:float = 1.0) -> Dict[str, np.ndarray]:
        """
        Loads .npz files for the given list of datasets and split (train/valid).
        Handles:
          - Weights Calculation (xsec/nEvent)
          - Mass Parameterization (Random injection for Bkg)
          - Feature Selection
        """
        X_list, y_list, w_list, m_list, p_list = [], [], [], [], []

        for ds in datasets:
            search_path = ds.path / "xgb" / split
            files = list(search_path.glob("*.npz"))

            if not files:
                continue

            for fp in files:
                try:
                    with np.load(fp, allow_pickle=True) as data:
                        if 'X' not in data: continue
                        arr = data['X']
                        if len(arr) == 0: continue

                        # --- Feature Management ---
                        # Initialize feature mapping on first successful load
                        if self.feature_names_loaded is None and 'features' in data:
                            self.feature_names_loaded = list(data['features'])
                            if self.features:
                                self.feature_indices = [self.feature_names_loaded.index(f) for f in self.features if
                                                        f in self.feature_names_loaded]
                            else:
                                self.feature_indices = list(range(len(arr[0])))

                        # Select Columns
                        if self.feature_indices:
                            arr = arr[:, self.feature_indices]

                        # --- Weights ---
                        # Weight = (Sign of genWeight) * (xsec / total_nevents)
                        raw_w = data['weights'] if 'weights' in data else np.ones(len(arr))
                        phys_w = raw_w * (ds.xsec * lumi / ds.nevents)
                        # if split == "train":
                        #     phys_w = abs(phys_w)  # Use absolute weights for training
                        #
                        # --- Mass Injection ---
                        N = len(arr)
                        if ds.is_signal:
                            mass_arr = np.column_stack([np.full(N, ds.mx), np.full(N, ds.my)])
                        else:
                            # For Background Training: Inject random mass hypotheses
                            if self.parameterize and split == "train":
                                if target_masses is None:
                                    raise ValueError("Target masses required for Background parameterization")
                                rand_idx = np.random.randint(0, len(target_masses), size=N)
                                mass_arr = target_masses[rand_idx]
                            else:
                                mass_arr = np.zeros((N, 2))

                        X_list.append(arr)
                        y_list.append(np.ones(N) if ds.is_signal else np.zeros(N))
                        w_list.append(phys_w)
                        m_list.append(mass_arr)
                        p_list.append([ds.category] * N)

                except Exception as e:
                    logger.warning(f"Corrupt file {fp}: {e}")

        if not X_list:
            logger.error(f"No data loaded for split {split}!")
            return {}

        return {
            'X': np.concatenate(X_list),
            'y': np.concatenate(y_list),
            'w': np.concatenate(w_list),
            'm': np.concatenate(m_list),
            'proc': np.concatenate(p_list)
        }

    def reweight_signals(self, data: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Renormalize signal weights so each mass point contributes equally."""
        if 'm' not in data: return data

        w = data['w']
        m = data['m']
        unique_masses = np.unique(m, axis=0)
        if len(unique_masses) == 0: return data

        target_w = np.sum(w) / len(unique_masses)

        logger.info(f"Reweighting {len(unique_masses)} signal points to target weight {target_w:.2e}")

        for mx, my in unique_masses:
            mask = (m[:, 0] == mx) & (m[:, 1] == my)
            current_sum = np.sum(w[mask])
            if current_sum > 0:
                w[mask] *= (target_w / current_sum)

        data['w'] = w
        return data

    def downsample_for_tabpfn(self, X, y, w, limit=20000):
        """Probabilistic downsampling based on weights."""
        if len(X) <= limit: return X, y, w
        logger.info(f"TabPFN Downsampling: {len(X)} -> {limit}")

        prob = np.abs(w) / np.sum(np.abs(w))
        idx = np.random.choice(len(X), limit, replace=False, p=prob)
        return X[idx], y[idx], w[idx]

# ==========================================
# 3. Plotting Helpers
# ==========================================

def plot_overtraining(model, X_tr, y_tr, w_tr, X_val, y_val, w_val, out_dir):
    """Checks score distribution for Train vs Validation to detect overtraining."""
    print(">>> Plotting Overtraining Check...")

    # Handle TabPFN or large datasets to prevent OOM during prediction
    if hasattr(model, "predict_proba"):
        # Downsample for plotting if dataset is too large (>20k)
        if len(X_tr) > 200000:
            idx = np.random.choice(len(X_tr), 200000, replace=False)
            X_tr, y_tr, w_tr = X_tr[idx], y_tr[idx], w_tr[idx]

        # XGBoost handles this fast, TabPFN needs small batches if not downsampled
        # --- FIX: Batch prediction for TabPFN to avoid CUDA errors ---
        # Check if the dataset is large (e.g. > 2000 samples)
        if len(X_tr) >50000:
            batch_size = 50000
            preds = []
            for i in range(0, len(X_tr), batch_size):
                # Predict in chunks
                preds.append(model.predict_proba(X_tr[i:i + batch_size])[:, 1])
            tr_scores = np.concatenate(preds)
        else:
            # Small dataset, run normally
            tr_scores = model.predict_proba(X_tr)[:, 1]

        # Repeat the same for Validation set if it is also large
        if len(X_val) > 50000:
            batch_size = 50000
            preds_val = []
            for i in range(0, len(X_val), batch_size):
                preds_val.append(model.predict_proba(X_val[i:i + batch_size])[:, 1])
            val_scores = np.concatenate(preds_val)
        else:
            val_scores = model.predict_proba(X_val)[:, 1]

    else:
        return

    plt.figure(figsize=(10, 8))
    bins = np.linspace(0, 1, 40)

    # Train (Filled Histogram)
    plt.hist(tr_scores[y_tr == 0], bins=bins, weights=w_tr[y_tr == 0], density=True,
             alpha=0.3, color='blue', label='Train Bkg')
    plt.hist(tr_scores[y_tr == 1], bins=bins, weights=w_tr[y_tr == 1], density=True,
             alpha=0.3, color='red', label='Train Sig')

    # Valid (Dots / Error bars)
    h_b, _ = np.histogram(val_scores[y_val == 0], bins=bins, weights=w_val[y_val == 0], density=True)
    h_s, _ = np.histogram(val_scores[y_val == 1], bins=bins, weights=w_val[y_val == 1], density=True)
    ct = (bins[:-1] + bins[1:]) / 2

    plt.plot(ct, h_b, 'o', color='blue', label='Valid Bkg')
    plt.plot(ct, h_s, 'o', color='red', label='Valid Sig')

    plt.xlabel("Model Score")
    plt.ylabel("Density")
    plt.legend()
    plt.title("Overtraining Check")
    plt.savefig(out_dir / "overtraining.png")
    plt.close()





# ==========================================
# 3. Execution Flow
# ==========================================

def run_pipeline(args):
    # 1. Setup
    mode_str = "parametrized" if args.parameterize else "individual"
    mass_target = "All" if args.parameterize else f"MX-{args.mX}_MY-{args.mY}"
    out_dir = Path(args.out_dir) / args.model / mode_str / mass_target
    out_dir.mkdir(parents=True, exist_ok=True)

    # 2. Config & Discovery
    cfg = ConfigLoader(args.yaml_path, args.base_dir)
    all_datasets = cfg.discover_datasets()

    # Filter Signals based on args
    if args.parameterize:
        sig_datasets = [d for d in all_datasets if d.is_signal]
    else:
        sig_datasets = [d for d in all_datasets if d.is_signal and d.mx == args.mX and d.my == args.mY]
        if not sig_datasets:
            logger.error(f"Signal MX={args.mX}, MY={args.mY} not found!")
            sys.exit(1)

    bkg_datasets = [d for d in all_datasets if not d.is_signal]

    # Collect all available mass points for parametrization logic
    target_masses = np.array([[d.mx, d.my] for d in all_datasets if d.is_signal])

    # 3. Load Training Data
    dm = DatasetManager(cfg, parameterize=args.parameterize, features=args.features)
    model = None
    if "train" in args.stage:
        logger.info(">>> Loading Signal (Train)...")
        d_sig_tr = dm.load_data(sig_datasets, "train", lumi=args.lumi)
        d_sig_tr = dm.reweight_signals(d_sig_tr)  # Equalize mass points

        logger.info(">>> Loading Background (Train)...")
        d_bkg_tr = dm.load_data(bkg_datasets, "train", target_masses=target_masses, lumi=args.lumi)

        # Global Balance: Sum(Bkg Weights) = Sum(Sig Weights)
        # ---- global balance: scale background to match total signal weight ----
        sig_sum = d_sig_tr["w"].sum()
        bkg_sum = d_bkg_tr["w"].sum()

        if bkg_sum > 0:
            num_bkg = d_bkg_tr["w"].shape[0]
            d_bkg_tr["w"] = d_bkg_tr["w"] * (num_bkg / bkg_sum)
            d_sig_tr["w"] = d_sig_tr["w"] * (num_bkg / sig_sum)
        #
        # scale = np.sum(d_sig_tr['w']) / np.sum(d_bkg_tr['w'])
        # d_bkg_tr['w'] *= scale

        # Merge
        X_full = np.concatenate([d_bkg_tr['X'], d_sig_tr['X']])
        y_full = np.concatenate([d_bkg_tr['y'], d_sig_tr['y']])
        w_full = np.concatenate([d_bkg_tr['w'], d_sig_tr['w']])

        w_full = abs(w_full)

        if args.parameterize:
            m_full = np.concatenate([d_bkg_tr['m'], d_sig_tr['m']])
            X_full = np.hstack([X_full, m_full])

        # 4. Training
        X_tr, X_val, y_tr, y_val, w_tr, w_val = train_test_split(
            X_full, y_full, w_full, test_size=0.2, stratify=y_full, random_state=42
        )

        start_time = time.time()

        # positive_weight_mask = w_tr > 0
        #
        # X_tr = X_tr[positive_weight_mask]
        # y_tr = y_tr[positive_weight_mask]
        # w_tr = w_tr[positive_weight_mask]

        if args.model == 'xgb':
            logger.info("Training XGBoost...")
            model = xgb.XGBClassifier(
                n_estimators=1000, learning_rate=0.05, max_depth=6,
                early_stopping_rounds=50,
                device= 'cuda' if os.environ.get('CUDA_VISIBLE_DEVICES') else 'cpu'
            )
            model.fit(
                X_tr, y_tr, sample_weight=w_tr,
                eval_set=[(X_val, y_val)], sample_weight_eval_set=[w_val],
                verbose=100
            )
            model.save_model(out_dir / "model.json")

        elif args.model == 'tabpfn':
            if not HAS_TABPFN:
                logger.error("TabPFN requested but not installed.")
                sys.exit(1)
            logger.info("Training TabPFN...")
            X_sub, y_sub, _ = dm.downsample_for_tabpfn(X_tr, y_tr, w_tr, limit=args.tabpfn_limit)
            model = TabPFNClassifier(n_estimators=1,balance_probabilities=True) #'cuda' if os.environ.get('CUDA_VISIBLE_DEVICES') else 'cpu')
            print("cuda:", os.environ.get('CUDA_VISIBLE_DEVICES'))
            model.fit(X_sub, y_sub)

        finish_time = time.time()
        fitting_time = finish_time - start_time

        # =========================================================
        # [INSERT 1] Plot Overtraining Check (Right after training)
        # =========================================================
        plot_overtraining(model, X_tr, y_tr, w_tr, X_val, y_val, w_val, out_dir)

    if "predict" in args.stage:
        # 5. Inference (Evaluation)
        if model is None:
            logger.info(">>> Loading Trained Model...")
            if args.model == 'xgb':
                model = xgb.XGBClassifier()
                model.load_model(out_dir / "model.json")
            elif args.model == 'tabpfn':
                if not HAS_TABPFN:
                    logger.error("TabPFN requested but not installed.")
                    sys.exit(1)
                model = TabPFNClassifier(n_estimators=1,balance_probabilities=True) # TODO add evaluation check point
        logger.info(">>> Loading Test Data...")
        d_sig_te = dm.load_data(sig_datasets, "valid", lumi=args.lumi)
        d_bkg_te = dm.load_data(bkg_datasets, "valid", lumi=args.lumi)
        d_sig_te = dm.reweight_signals(d_sig_te)

        # Prepare for parametrized inference loop
        unique_masses = np.unique(d_sig_te['m'], axis=0)

        for mx, my in unique_masses:
            # A. Get Signal Subset
            if args.mX is not None and (int(mx) != int(args.mX)):
                continue
            if args.mY is not None and (int(my) != int(args.mY)):
                continue
            mask_s = (d_sig_te['m'][:, 0] == mx) & (d_sig_te['m'][:, 1] == my)
            X_s = d_sig_te['X'][mask_s]
            # B. Get Background (Parameterize Injection)
            X_b = d_bkg_te['X'].copy()
            if args.parameterize:
                # Overwrite Bkg mass to current signal hypothesis
                m_b_inj = np.column_stack([np.full(len(X_b), mx), np.full(len(X_b), my)])
                m_s_act = np.column_stack([np.full(len(X_s), mx), np.full(len(X_s), my)])
                X_eval = np.concatenate([
                    np.hstack([X_b, m_b_inj]),
                    np.hstack([X_s, m_s_act])
                ])
            else:
                X_eval = np.concatenate([X_b, X_s])

            y_eval = np.concatenate([d_bkg_te['y'], d_sig_te['y'][mask_s]])
            w_eval = np.concatenate([d_bkg_te['w'], d_sig_te['w'][mask_s]])
            p_eval = np.concatenate([d_bkg_te['proc'], d_sig_te['proc'][mask_s]])

            # C. Predict
            if args.model == 'tabpfn' and len(X_eval) > 50000:
                # Batch prediction
                batch = 50000
                preds = []
                for i in range(0, len(X_eval), batch):
                    preds.append(model.predict_proba(X_eval[i:i + batch])[:, 1])
                y_pred = np.concatenate(preds)
            else:
                y_pred = model.predict_proba(X_eval)[:, 1]


            predict_value = {
                "y_true": y_eval.tolist(),
                "y_pred": y_pred.tolist(),
                "w": w_eval.tolist(),
                "proc": p_eval.tolist(),
                "mx": mx,
                "my": my,
            }

            with open(out_dir / f"predictions_MX-{int(round(mx))}_MY-{int(round(my))}.json", "w") as f:
                json.dump(predict_value, f, indent=4)

    if "evaluate" in args.stage:
        logger.info(">>> Evaluating Predictions...")
        # Load predictions
        all_masses = [(d_sig.mx, d_sig.my) for d_sig in sig_datasets]
        for mx, my in all_masses:
            if args.mX is not None and (int(mx) != int(args.mX)):
                continue
            if args.mY is not None and (int(my) != int(args.mY)):
                continue

            pred_file = out_dir / f"predictions_MX-{int(mx)}_MY-{int(my)}.json"
            if not pred_file.exists():
                logger.warning(f"Prediction file not found: {pred_file}")
                continue

            with open(pred_file) as f:
                pred_data = json.load(f)

            y_eval = np.array(pred_data['y_true'])
            y_pred = np.array(pred_data['y_pred'])
            w_eval = np.array(pred_data['w'])
            p_eval = np.array(pred_data['proc'])

            # D. Metrics
            metrics = calculate_physics_metrics(
                y_pred, y_eval, w_eval, training=False,
                min_bkg_events=10,
                log_plots=True,
                bins=1000,
                min_bkg_ratio=0.0001,
                f_name=f"{out_dir}/sic_MX-{int(mx)}_MY-{int(my)}.png",
                Zs=10,
                Zb=5,
                min_bkg_per_bin=3,
                min_mc_stats=1.0,
            )

            key = f"MX-{int(mx)}_MY-{int(my)}"
            results = {
                "auc": float(metrics['auc']),
                "max_sic": float(metrics['max_sic']),
                "max_sic_unc": float(metrics['max_sic_unc']),
                # "fitting_time": fitting_time
            }
            logger.info(f"Mass {key}: AUC={metrics['auc']:.4f}, Max SIC={metrics['max_sic']:.4f}")

            plot_score_overlay(
                y_eval=y_eval,
                w_eval=w_eval,
                p_eval=p_eval,
                y_pred=y_pred,
                fname = out_dir / f"score_MX-{int(mx)}_MY-{int(my)}.png"
            )
            with open(out_dir / f"eval_metrics_MX-{int(mx)}_MY-{int(my)}.json", "w") as f:
                json.dump(results, f, indent=4)
    logger.info(f"Done. Results saved to {out_dir}")


# ==========================================
# 4. Entry Point
# ==========================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Advanced Grid Search Trainer")

    # Data Selection
    parser.add_argument("--base_dir", type=str, default="/pscratch/sd/t/tihsu/database/GridStudy_v2")
    parser.add_argument("--yaml_path", type=str, default="sample.yaml")
    parser.add_argument("--features_yaml", type=str, default=None, help="YAML file specifying features to use")
    parser.add_argument("--mX", type=float, default=None)
    parser.add_argument("--mY", type=float, default=None)
    parser.add_argument("--lumi", type=float, default=36000)

    # Model Config
    parser.add_argument("--model", type=str, default="xgb", choices=["xgb", "tabpfn"])
    parser.add_argument("--parameterize", action="store_true", help="Include Mass as input")
    parser.add_argument("--features", nargs="+", help="Explicit list of features to use")
    parser.add_argument("--tabpfn_limit", type=int, default=50000)

    # IO
    parser.add_argument("--out_dir", type=str, default="results")
    parser.add_argument("--stage", type=str, default=["train", "predict", "evaluate"], nargs="+", help="Pipeline stages to run")

    args = parser.parse_args()

    if not args.parameterize and (args.mX is None or args.mY is None):
        parser.error("Specify -mX and -mY, or use --parameterize for mass parameterization.")

    if args.features_yaml:
        with open(args.features_yaml) as f:
            feat_cfg = yaml.safe_load(f)
            args.features = feat_cfg.get('features', [])

    run_pipeline(args)