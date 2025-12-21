import argparse
import logging

import torch

from evenet_lite import EvenetLiteClassifier


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train EveNet-Lite with optional pretrained weights.")
    parser.add_argument(
        "--pretrained",
        choices=["none", "hf", "local"],
        default="none",
        help="Select pretrained weight source. 'hf' pulls from Hugging Face via hf_utils; 'local' loads a checkpoint path.",
    )
    parser.add_argument(
        "--pretrained-path",
        default=None,
        help="Path to a local checkpoint when --pretrained local is selected.",
    )
    parser.add_argument(
        "--pretrained-repo-id",
        default="Evenet-Lite/evenet-lite",
        help="Hugging Face repo id used when --pretrained hf is selected.",
    )
    parser.add_argument(
        "--pretrained-filename",
        default="model.pt",
        help="Filename inside the Hugging Face repo to download when --pretrained hf is selected.",
    )
    parser.add_argument(
        "--pretrained-cache-dir",
        default=None,
        help="Optional cache directory for Hugging Face downloads.",
    )
    return parser.parse_args()

def main() -> None:
    args = parse_args()
    train_data = {
        'sig': torch.load(
            "workspace/data/46915_NMSSM_XToYHTo2B2Tau_MX-600_MY-300_TuneCP5_13TeV-madgraph-pythia8_RunIISummer20UL16NanoAODv9-106X_mcRun2_asymptotic_v17-v1_NANOAODSIM/evenet/train/skim_32F37BC4-BDDF-BF4F-91ED-3CF4F049586E.pt"),
        'bkg': torch.load(
            "/Users/avencastmini/PycharmProjects/EveNet-Lite/workspace/data/67993_TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8_RunIISummer20UL16NanoAODv9-106X_mcRun2_asymptotic_v17-v1_NANOAODSIM/evenet/train/skim_0AFFFA0B-438F-0B4E-A74E-344E41B69B2D.pt")
    }

    valid_data = {
        'sig': torch.load(
            "workspace/data/46915_NMSSM_XToYHTo2B2Tau_MX-600_MY-300_TuneCP5_13TeV-madgraph-pythia8_RunIISummer20UL16NanoAODv9-106X_mcRun2_asymptotic_v17-v1_NANOAODSIM/evenet/valid/skim_32F37BC4-BDDF-BF4F-91ED-3CF4F049586E.pt"),
        'bkg': torch.load(
            "workspace/data/67993_TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8_RunIISummer20UL16NanoAODv9-106X_mcRun2_asymptotic_v17-v1_NANOAODSIM/evenet/valid/skim_0AFFFA0B-438F-0B4E-A74E-344E41B69B2D.pt")
    }

    X_train = {
        k if k != 'global' else 'globals': torch.cat([train_data['sig'][k], train_data['bkg'][k]], dim=0)
        for k in ['x', 'x_mask', 'global']
    }
    y_train = torch.cat([
        torch.ones(len(train_data['sig']["x"])),
        torch.zeros(len(train_data['bkg']["x"])),
    ], dim=0)

    X_val = {
        k if k != 'global' else 'globals': torch.cat([valid_data['sig'][k], valid_data['bkg'][k]], dim=0)[:1000, ...]
        for k in ['x', 'x_mask', 'global']
    }
    y_val = torch.cat([
        torch.ones(len(valid_data['sig']["x"]))[:500],
        torch.zeros(len(valid_data['bkg']["x"]))[:500],
    ], dim=0)

    obj_feature_names = ['energy', 'pt', 'eta', 'phi', 'isBTag', 'isLepton', 'Charge']
    global_feature_names = ['met', 'met_phi', 'nLepton', 'nbJet', 'HT', 'HT_lep', 'M_all', 'M_leps', 'M_bjets']

    clf = EvenetLiteClassifier(
        class_labels=['bkg', 'signal'],
        device="auto",  # "cpu", "cuda", or "auto"
        body_lr=1e-4,
        head_lr=1e-3,
        weight_decay=1e-3,
        grad_clip=1.0,
        warmup_epochs=1,
        warmup_ratio=0.1,
        warmup_start_factor=0.1,
        log_level=logging.INFO,
        use_wandb=True,
        wandb={
            'project': 'EvenetLite',
            'name': 'test',
        },
        pretrained=args.pretrained != "none",
        pretrained_source=args.pretrained if args.pretrained != "none" else "hf",
        pretrained_path=args.pretrained_path,
        pretrained_repo_id=args.pretrained_repo_id,
        pretrained_filename=args.pretrained_filename,
        pretrained_cache_dir=args.pretrained_cache_dir,
    )

    clf.fit(
        train_data=(X_train, y_train, None),
        val_data=(X_val, y_val, None),
        callbacks=[],  # custom callbacks optional; normalization auto-injected
        epochs=10,
        batch_size=256,
        sampler="weighted",  # or None
        epoch_size=1280, # or None,
        save_top_k=2,
        checkpoint_every=1,
        # checkpoint_path="./checkpoint",
        feature_names={"objects": obj_feature_names, "globals": global_feature_names},
        normalization_rules={
            "x": {
                "energy": "log_normalize",
                "pt": "log_normalize",
                "eta": "normalize",
                "phi": "normalize_uniform",
                "btag": "none",
                "isLepton": "none",
                "charge": "none",
            },
            "global": {
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
        }
    )
    #
    # probs = clf.predict(X_test, batch_size=256)
    # metrics = clf.evaluate(X_test, y_test, w_test, batch_size=256)

if __name__ == '__main__':
    main()
