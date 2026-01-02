import numpy as np
import awkward as ak
# ==========================================
# 0. Histogram Definitions (Full List)
# ==========================================
def _edges(nbins, lo, hi):
    return np.linspace(lo, hi, nbins + 1)

HIST_DEFS = {
    # event weight
    "event_weight": _edges(60, -1, 9),
    # EveNet Point Cloud
    "x_E": _edges(60, 0, 1000), "x_pt": _edges(60, 0, 600),
    "x_eta": _edges(50, -3, 3), "x_phi": _edges(50, -3.14, 3.14),
    "x_isbtag": np.array([-0.5, 0.5, 1.5, 2.5], dtype=np.float64),
    "x_isLepton": np.array([-1.5, -0.5, 0.5, 1.5], dtype=np.float64),
    "x_charge": np.array([-1.5, -0.5, 0.5, 1.5], dtype=np.float64),

    # EveNet Global (Full 10 vars)
    "g_met": _edges(60, 0, 600), "g_met_phi": _edges(50, -3.14, 3.14),
    "g_nLepton": np.arange(-0.5, 6.5, 1.0), "g_nbJet": np.arange(-0.5, 10.5, 1.0), "g_nJet": np.arange(-0.5, 16.5, 1.0),
    "g_HT": _edges(60, 0, 2500), "g_HT_lep": _edges(60, 0, 1000),
    "g_M_all": _edges(60, 0, 4000), "g_M_leps": _edges(60, 0, 1500), "g_M_bjets": _edges(60, 0, 2000),

    # XGB (Full List)
    "xgb_mHH_proxy": _edges(60, 0, 3000), "xgb_mbb": _edges(60, 0, 1000),
    "xgb_dR_ltau": _edges(50, 0, 6), "xgb_dR_bb": _edges(50, 0, 6),
    "xgb_dpt_ltau": _edges(60, 0, 500), "xgb_b2_pt": _edges(60, 0, 500),
    "xgb_mT_W": _edges(60, 0, 400), "xgb_met": _edges(60, 0, 600),
    "xgb_dphi_ltau_bb": _edges(50, 0, 3.14), "xgb_dphi_l_met": _edges(50, 0, 3.14),
    "xgb_mtautau_visible": _edges(60, 0, 1500), "xgb_b1_pt": _edges(60, 0, 500),
    "l1_pt": _edges(60, 0, 600), "t1_pt": _edges(60, 0, 600),
    "xgb_mb1l": _edges(60, 0, 1000), "xgb_mb2l": _edges(60, 0, 1000),
    "xgb_mb1t": _edges(60, 0, 1000), "xgb_mb2t": _edges(60, 0, 1000),
    "xgb_dr_b1l": _edges(50, 0, 6), "xgb_dr_b2l": _edges(50, 0, 6),
    "xgb_dr_b1t": _edges(50, 0, 6), "xgb_dr_b2t": _edges(50, 0, 6),
}


# ==========================================
# 1. DQM Accumulator
# ==========================================
class DQMAccumulator:
    def __init__(self, hist_defs):
        self.hist_defs = hist_defs
        self.hists = {}
        self.meta = {"n_train": 0, "n_valid": 0}

        for name, edges in self.hist_defs.items():
            nbins = len(edges) - 1
            self.hists[name] = {
                "train": np.zeros(nbins, dtype=np.int64),
                "valid": np.zeros(nbins, dtype=np.int64)
            }

    def __add__(self, other):
        if not isinstance(other, DQMAccumulator):
            raise ValueError("Identity mismatch")

        new_acc = DQMAccumulator(self.hist_defs)

        new_acc.meta["n_train"] = self.meta["n_train"] + other.meta["n_train"]
        new_acc.meta["n_valid"] = self.meta["n_valid"] + other.meta["n_valid"]

        for name in self.hists:
            new_acc.hists[name]["train"] = self.hists[name]["train"] + other.hists[name]["train"]
            new_acc.hists[name]["valid"] = self.hists[name]["valid"] + other.hists[name]["valid"]

        return new_acc

    def fill(self, name, split, values):
        if name not in self.hists: return
        arr = ak.to_numpy(ak.flatten(values, axis=None))
        arr = arr[~np.isnan(arr)]
        if len(arr) == 0: return
        counts, _ = np.histogram(arr, bins=self.hist_defs[name])
        self.hists[name][split] += counts.astype(np.int64)
