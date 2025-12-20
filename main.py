from evenet_lite import EvenetLiteClassifier
import torch

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
        k if k != 'global' else 'globals': torch.cat([valid_data['sig'][k], valid_data['bkg'][k]], dim=0)
        for k in ['x', 'x_mask', 'global']
    }
    y_val = torch.cat([
        torch.ones(len(valid_data['sig']["x"])),
        torch.zeros(len(valid_data['bkg']["x"])),
    ], dim=0)

    obj_feature_names = ['pt', 'eta', 'phi', 'E', 'isBTag', 'isLepton', 'Charge']
    global_feature_names = ['met', 'met_phi', 'nLepton', 'nbJet', 'HT', 'HT_lep', 'M_all', 'M_leps', 'M_bjets']

    clf = EvenetLiteClassifier(
        class_labels=['signal', 'bkg'],
        device="auto",  # "cpu", "cuda", or "auto"
        lr=1e-3,
        weight_decay=0.01,
        grad_clip=1.0,  # optional gradient clipping
    )

    clf.fit(
        train_data=(X_train, y_train, None),
        val_data=(X_val, y_val, None),
        feature_names={"objects": obj_feature_names, "globals": global_feature_names},
        callbacks=[],  # custom callbacks optional; normalization auto-injected
        epochs=10,
        batch_size=256,
        sampler="weighted",  # or None
        epoch_size=None,
    )
    #
    # probs = clf.predict(X_test, batch_size=256)
    # metrics = clf.evaluate(X_test, y_test, w_test, batch_size=256)


