import logging
import numpy as np
from evenet_lite import EvenetLiteClassifier
import torch
from evenet_lite.callbacks import ParameterRandomizationCallback

if __name__ == '__main__':
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

    N = len(X_train["x"])
    X_train['params'] = torch.tensor(np.vstack([
        np.tile([350, 700], (N // 2, 1)),
        np.tile([200, 500], (N - N // 2, 1)),
    ]).astype(np.float32))

    N = len(X_val["x"])
    X_val['params'] = torch.tensor(np.vstack([
        np.tile([350, 700], (N // 2, 1)),
        np.tile([200, 500], (N - N // 2, 1)),
    ]).astype(np.float32))

    obj_feature_names = ['energy', 'pt', 'eta', 'phi', 'isBTag', 'isLepton', 'Charge']
    global_feature_names = ['met', 'met_phi', 'nLepton', 'nbJet', 'nJet', 'HT', 'HT_lep', 'M_all', 'M_leps', 'M_bjets']

    clf = EvenetLiteClassifier(
        class_labels=['bkg', 'signal'],
        device="mps",  # "cpu", "cuda", or "auto"
        body_lr=1e-4,
        head_lr=1e-3,
        weight_decay=1e-3,
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
        global_input_dim=12,
        num_workers=0,
        body_modules=["GlobalEmbedding", "PET", "ObjectEncoder"],
        head_modules=["Classification"],
    )

    clf.fit(
        train_data=(X_train, y_train, None),
        val_data=(X_val, y_val, None),
        eval_data=(X_val, y_val, None),
        callbacks=[ParameterRandomizationCallback(min_values=[300, 500], max_values=[800, 1200])],
        epochs=5,
        batch_size=128,
        sampler="weighted",  # or None
        epoch_size=1280,  # or None,
        save_top_k=1,
        checkpoint_every=1,
        # checkpoint_path="./checkpoint",
        feature_names={
            "x": obj_feature_names,
            "globals": global_feature_names,
            "params": ['m_X', 'm_Y'],
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
        normalization_stats={
            "x": {
                "mean": [0.12, -0.03, 0.5],  # len == num object features
                "std": [1.1, 0.95, 0.8],
            },
            "globals": {
                "mean": [-0.05, 10.0],  # len == num global features
                "std": [1.0, 1.0],
            },
        }
    )
    #
    # probs = clf.predict(X_val, batch_size=256)
    # metrics = clf.evaluate(X_test, y_test, w_test, batch_size=256)
