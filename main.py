import logging
from glob import glob

import numpy as np
from evenet_lite import EvenetLiteClassifier
import torch
from evenet_lite.callbacks import ParameterRandomizationCallback
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple
import regex as re


def _resolve_paths(path_str: str, description: str) -> List[Path]:
    """Resolve one or more concrete files from a path or glob pattern."""

    candidate = Path(path_str)
    if candidate.is_file():
        return [candidate]

    def _expand_braces(pattern: str) -> List[str]:
        match = re.search(r"\{([^{}]+)\}", pattern)
        if not match:
            return [pattern]

        options = match.group(1).split(",")
        prefix = pattern[: match.start()]
        suffix = pattern[match.end():]
        expanded: List[str] = []
        for option in options:
            expanded.extend(_expand_braces(prefix + option + suffix))
        return expanded

    expanded_patterns = _expand_braces(path_str)
    matches = sorted({Path(p) for pat in expanded_patterns for p in glob(pat)})
    if not matches:
        raise FileNotFoundError(
            f"No files matched {description} pattern: {path_str} (expanded to {expanded_patterns})"
        )

    return matches


def _concat_tensors(data_iter: Iterable[torch.Tensor]) -> torch.Tensor:
    return torch.cat(list(data_iter), dim=0)


BKG_META = {
    "tt1l": {
        "xsec": 365.35,
        "nEvent": 144_722_000,
    },
    "DYBJets_pt100to200": {
        "xsec": 3.222,
        "nEvent": 8_848_155,
    },
    "DYBJets_pt200toInf": {
        "xsec": 0.6181,
        "nEvent": 887_122,
    },
    "ggHtautau": {
        "xsec": 3.08,
        "nEvent": 6_439_000,
    },
    "VBFHtautau": {
        "xsec": 0.237,
        "nEvent": 1_500_000,
    },
}

def _match_bkg_sample(path: Path) -> str:
    path_str = str(path)
    for name in BKG_META:
        if name in path_str:
            return name
    raise ValueError(f"Cannot match background sample for path: {path}")

def _make_sample_weights(path: Path, n_events: int) -> torch.Tensor:
    sample = _match_bkg_sample(path)
    meta = BKG_META[sample]
    w = meta["xsec"] / meta["nEvent"] * 1000 * 36
    return torch.full((n_events,), w, dtype=torch.float32)

def _load_split(sig_paths: List[Path], bkg_paths: List[Path]):
    sig_parts = [torch.load(p, weights_only=False, map_location="cpu") for p in sig_paths]
    bkg_parts = [torch.load(p, weights_only=False, map_location="cpu") for p in bkg_paths]

    available_keys = {key for part in [*sig_parts, *bkg_parts] for key in part.keys()}
    requested_keys = ["x", "x_mask", "global", "params"]

    features = {}
    for key in requested_keys:
        if key not in available_keys or any(key not in part for part in [*sig_parts, *bkg_parts]):
            continue
        alias = "globals" if key == "global" else key
        features[alias] = _concat_tensors(
            [*(part[key] for part in sig_parts),
             *(part[key] for part in bkg_parts)]
        )

    # labels
    n_sig = sum(len(part["x"]) for part in sig_parts)
    n_bkg = sum(len(part["x"]) for part in bkg_parts)

    labels = torch.cat(
        [
            torch.ones(n_sig),
            torch.zeros(n_bkg),
        ],
        dim=0,
    )

    # ---- weights (physics-correct) ----
    sig_weights = torch.ones(n_sig, dtype=torch.float32)

    bkg_weights = []
    for path, part in zip(bkg_paths, bkg_parts):
        n = len(part["x"])
        bkg_weights.append(_make_sample_weights(path, n))

    weights = torch.cat([sig_weights, *bkg_weights], dim=0)

    return features, labels, weights


if __name__ == '__main__':
    # train_data = {
    #     'sig': torch.load(
    #         "workspace/data/46915_NMSSM_XToYHTo2B2Tau_MX-600_MY-300_TuneCP5_13TeV-madgraph-pythia8_RunIISummer20UL16NanoAODv9-106X_mcRun2_asymptotic_v17-v1_NANOAODSIM/evenet/train/skim_32F37BC4-BDDF-BF4F-91ED-3CF4F049586E.pt"),
    #     'bkg': torch.load(
    #         "/Users/avencastmini/PycharmProjects/EveNet-Lite/workspace/data/67993_TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8_RunIISummer20UL16NanoAODv9-106X_mcRun2_asymptotic_v17-v1_NANOAODSIM/evenet/train/skim_0AFFFA0B-438F-0B4E-A74E-344E41B69B2D.pt")
    # }
    #
    # valid_data = {
    #     'sig': torch.load(
    #         "workspace/data/46915_NMSSM_XToYHTo2B2Tau_MX-600_MY-300_TuneCP5_13TeV-madgraph-pythia8_RunIISummer20UL16NanoAODv9-106X_mcRun2_asymptotic_v17-v1_NANOAODSIM/evenet/valid/skim_32F37BC4-BDDF-BF4F-91ED-3CF4F049586E.pt"),
    #     'bkg': torch.load(
    #         "workspace/data/67993_TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8_RunIISummer20UL16NanoAODv9-106X_mcRun2_asymptotic_v17-v1_NANOAODSIM/evenet/valid/skim_0AFFFA0B-438F-0B4E-A74E-344E41B69B2D.pt")
    # }

    # X_train = {
    #     k if k != 'global' else 'globals': torch.cat([train_data['sig'][k], train_data['bkg'][k]], dim=0)
    #     for k in ['x', 'x_mask', 'global']
    # }
    # y_train = torch.cat([
    #     torch.ones(len(train_data['sig']["x"])),
    #     torch.zeros(len(train_data['bkg']["x"])),
    # ], dim=0)
    #
    # X_val = {
    #     k if k != 'global' else 'globals': torch.cat([valid_data['sig'][k], valid_data['bkg'][k]], dim=0)[:1000, ...]
    #     for k in ['x', 'x_mask', 'global']
    # }
    # y_val = torch.cat([
    #     torch.ones(len(valid_data['sig']["x"]))[:500],
    #     torch.zeros(len(valid_data['bkg']["x"]))[:500],
    # ], dim=0)
    #
    # N = len(X_train["x"])
    # X_train['params'] = torch.tensor(np.vstack([
    #     np.tile([350, 700], (N // 2, 1)),
    #     np.tile([200, 500], (N - N // 2, 1)),
    # ]).astype(np.float32))
    #
    # N = len(X_val["x"])
    # X_val['params'] = torch.tensor(np.vstack([
    #     np.tile([350, 700], (N // 2, 1)),
    #     np.tile([200, 500], (N - N // 2, 1)),
    # ]).astype(np.float32))

    (train_features, train_labels, train_weights) = _load_split(
        _resolve_paths("/Users/avencastmini/PycharmProjects/EveNet-Lite/workspace/grid/MX-500_MY-90/evenet/train/*.pt",
                       "train signal"),
        _resolve_paths(
            "/Users/avencastmini/PycharmProjects/EveNet-Lite/workspace/grid/{DYBJets_pt100to200,DYBJets_pt200toInf,ggHtautau,VBFHtautau,tt1l}/evenet/train/*.pt",
            "train background"),
    )

    N = train_labels.shape[0]
    train_size = int(0.7 * N)
    val_size = N - train_size
    generator = torch.Generator().manual_seed(42)  # remove seed if you want
    perm = torch.randperm(N, generator=generator)

    train_idx = perm[:train_size]
    val_idx = perm[train_size:]


    def split_features(features, idx):
        out = {}
        for k, v in features.items():
            out[k] = v[idx]
        return out


    val_features = split_features(train_features, val_idx)
    train_features = split_features(train_features, train_idx)

    val_labels = train_labels[val_idx]
    train_labels = train_labels[train_idx]

    val_weights = train_weights[val_idx]
    train_weights = train_weights[train_idx]

    obj_feature_names = ['energy', 'pt', 'eta', 'phi', 'isBTag', 'isLepton', 'Charge']
    global_feature_names = ['met', 'met_phi', 'nLepton', 'nbJet', 'nJet', 'HT', 'HT_lep', 'M_all', 'M_leps', 'M_bjets']

    clf = EvenetLiteClassifier(
        class_labels=['bkg', 'signal'],
        device="mps",  # "cpu", "cuda", or "auto"
        lr=[5e-5, 1e-5, 5e-6],
        weight_decay=1e-2,
        module_lists=[
            ["Classification"],
            ["ObjectEncoder"],
            ["PET", "GlobalEmbedding"],
        ],
        grad_clip=1.0,
        warmup_epochs=1,
        warmup_ratio=0.1,
        warmup_start_factor=0.1,
        pretrained=True,
        log_level=logging.INFO,
        use_wandb=True,
        wandb={
            'project': 'EvenetLite',
            'name': 'test',
        },
        global_input_dim=10,
        num_workers=0,
        n_ensemble=3,
        ensemble_mode="shared_backbone",
    )

    clf.fit(
        train_data=(train_features, train_labels, None),
        val_data=(val_features, val_labels, val_weights),
        eval_data=(val_features, val_labels, val_weights),
        # callbacks=[ParameterRandomizationCallback(min_values=[300, 500], max_values=[800, 1200])],
        callbacks=[],
        epochs=1,
        batch_size=128,
        sampler="weighted",  # or None
        epoch_size=1280,  # or None,
        save_top_k=1,
        checkpoint_every=1,
        # checkpoint_path="./checkpoint",
        feature_names={
            "x": obj_feature_names,
            "globals": global_feature_names,
            # "params": ['m_X', 'm_Y'],
        },
        normalization_rules={
            "x": {
                "energy": "log_normalize",
                "pt": "log_normalize",
                "eta": "normalize",
                "phi": "normalize_uniform",
                "isBTag": "none",
                "isLepton": "none",
                "Charge": "none",
            },
            "globals": {
                "met": "log_normalize",
                "met_phi": "normalize",
                "nLepton": "none",
                "nbJet": "none",
                "nJet": "none",
                "HT": "log_normalize",
                "HT_lep": "log_normalize",
                "M_all": "log_normalize",
                "M_leps": "log_normalize",
                "M_bjets": "log_normalize",
            }
        },
        # normalization_stats={
        #     "x": {
        #         "mean": [0.12, -0.03, 0.5],  # len == num object features
        #         "std": [1.1, 0.95, 0.8],
        #     },
        #     "globals": {
        #         "mean": [-0.05, 10.0],  # len == num global features
        #         "std": [1.0, 1.0],
        #     },
        # }
    )
    #
    # probs = clf.predict(X_val, batch_size=256)
    # metrics = clf.evaluate(X_test, y_test, w_test, batch_size=256)
