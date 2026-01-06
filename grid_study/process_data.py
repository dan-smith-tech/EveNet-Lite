import argparse
import json
import numpy as np
import awkward as ak
import torch
import vector
from pathlib import Path
from coffea import processor
from coffea.nanoevents import NanoEventsFactory, NanoAODSchema
from coffea.analysis_tools import PackedSelection
from coffea.processor import Runner
from coffea.processor import FuturesExecutor
from coffea.nanoevents import NanoEventsFactory, NanoAODSchema
import warnings
import uuid
import collections
from accumulators import DQMAccumulator, HIST_DEFS

# Suppress performance warnings
warnings.filterwarnings("ignore", module="coffea")
# Enable vector behavior
vector.register_awkward()
import matplotlib.pyplot as plt
import mplhep as hep  # Recommended for HEP style plots: pip install mplhep

try:
    plt.style.use(hep.style.CMS)
except Exception:
    pass


# --- Add these OUTSIDE your class, near imports ---
def cutflow_factory():
    return processor.defaultdict_accumulator(int)

def nested_dict_int():
    """Helper to replace lambda: collections.defaultdict(int)"""
    return collections.defaultdict(int)

def dqm_factory():
    """Helper to replace lambda: DQMAccumulator(HIST_DEFS)"""
    # Assuming HIST_DEFS is a global variable defined earlier in the file
    return DQMAccumulator(HIST_DEFS)

def plot_dqm(dqm_data, dataset_name, out_dir):

    plot_dir = Path(out_dir) / dataset_name
    plot_dir.mkdir(parents=True, exist_ok=True)

    hists = dqm_data["hists"]
    hist_defs = dqm_data["hist_defs"]

    for name, counts_dict in hists.items():
        edges = hist_defs[name]
        centers = 0.5 * (edges[:-1] + edges[1:])
        width = edges[1:] - edges[:-1]

        counts_tr = counts_dict["train"].astype(np.float64)
        counts_va = counts_dict["valid"].astype(np.float64)

        # --- Density ---
        sum_tr = np.sum(counts_tr)
        sum_va = np.sum(counts_va)

        if sum_tr == 0 or sum_va == 0:
            continue

        dens_tr = counts_tr / (sum_tr * width)
        dens_va = counts_va / (sum_va * width)

        # --- (Main Pad + Ratio Pad) ---
        fig, (ax_main, ax_ratio) = plt.subplots(
            2, 1,
            gridspec_kw={'height_ratios': [3, 1]},
            figsize=(10, 8),
            sharex=True
        )

        # 1. Main Plot (Density)
        ax_main.step(centers, dens_tr, where='mid', label=f'Train ({int(sum_tr)})', color='black', linewidth=1.5)
        ax_main.step(centers, dens_va, where='mid', label=f'Valid ({int(sum_va)})', color='red', linestyle='--',
                     linewidth=1.5)

        # Error bars (Poisson approximation)
        err_tr = np.sqrt(counts_tr) / (sum_tr * width)
        err_va = np.sqrt(counts_va) / (sum_va * width)

        ax_main.errorbar(centers, dens_tr, yerr=err_tr, fmt='none', color='black')
        ax_main.errorbar(centers, dens_va, yerr=err_va, fmt='none', color='red')

        ax_main.set_ylabel("Density")
        ax_main.set_title(f"[{dataset_name}] {name}", loc='left', fontsize=16, fontweight='bold')
        ax_main.legend()
        ax_main.grid(True, alpha=0.3)

        # 2. Ratio Plot (Valid / Train)
        # Avoid divide by zero
        ratio = np.divide(dens_va, dens_tr, out=np.ones_like(dens_va), where=dens_tr != 0)

        ax_ratio.step(centers, ratio, where='mid', color='blue', linewidth=1)
        ax_ratio.axhline(1.0, color='gray', linestyle='--', linewidth=1)
        ax_ratio.set_ylim(0.5, 1.5)
        ax_ratio.set_ylabel("Valid / Train")
        ax_ratio.set_xlabel(name)
        ax_ratio.grid(True, alpha=0.3)

        # Save
        plt.tight_layout()
        fig.savefig(plot_dir / f"{name}.png", dpi=100)
        plt.close(fig)

    print(f"  Plots saved to {plot_dir}")

# ==========================================
# 2. Processor
# ==========================================
class FullLogicProcessor(processor.ProcessorABC):
    def __init__(self, config):
        self.cfg = config
        self._accumulator = processor.dict_accumulator({
            "cutflow": processor.defaultdict_accumulator(
                lambda: processor.defaultdict_accumulator(int)
            ),
            "dqm": processor.defaultdict_accumulator(lambda: DQMAccumulator(HIST_DEFS))
        })

    @property
    def accumulator(self):
        return self._accumulator

    def process(self, events):
        # 1. You can use lambda here freely for convenience while processing
        #    (It's fine as long as we don't try to return it directly)
        temp_cutflow = collections.defaultdict(lambda: collections.defaultdict(int))
        temp_dqm = collections.defaultdict(lambda: DQMAccumulator(HIST_DEFS))

        dataset = events.metadata["dataset"]
        filename = events.metadata["filename"]
        temp_cutflow[dataset]["total"] += len(events)

        # --- 1. Object Prep (Keep needed branches) ---
        # Attach 'iso' explicitly for XGB sorting later
        ele = events.Electron
        ele["iso"] = ele.pfRelIso03_all
        mu = events.Muon
        mu["iso"] = mu.pfRelIso04_all

        # ID Selection
        good_ele = ele[(ele.pt > 25) & (abs(ele.eta) < 2.1) & (ele.mvaFall17V2Iso_WP90) & (abs(ele.dxy) < 0.045) & (abs(ele.dz)<0.2) & (ele.iso < 0.15)]
        good_mu = mu[(mu.pt > 25) & (abs(mu.eta) < 2.1) & (mu.iso < 0.15) & (mu.mediumId) & (abs(mu.dxy) < 0.045) & (abs(mu.dz)<0.2)]
        good_tau = events.Tau[
            (events.Tau.pt > 30) & (abs(events.Tau.eta) < 2.3) & (abs(events.Tau.dz) < 0.2) & ((events.Tau.decayMode < 5) | (events.Tau.decayMode >= 10))
            & (events.Tau.idDeepTau2017v2p1VSjet >= 16) & (events.Tau.idDeepTau2017v2p1VSe >= 32) & (events.Tau.idDeepTau2017v2p1VSmu >= 1)]
        good_jet = events.Jet[(events.Jet.pt > 20) & (abs(events.Jet.eta) < 2.4)
                            & (events.Jet.jetId >= 4) & ~((events.Jet.pt < 50) & ~(events.Jet.puId == 7))]

        def dr_clean(obj, ref, dr=0.4):
            nearest = obj.nearest(ref)
            dR = obj.delta_r(nearest)
            return obj[ak.fill_none(dR, 999) > dr]  # ref 空 -> dR=None -> 當作 999 -> 保留 obj

        # Cross Cleaning
        good_tau = dr_clean(good_tau, good_ele, 0.4)
        good_tau = dr_clean(good_tau, good_mu, 0.4)
        good_jet = dr_clean(good_jet, good_ele, 0.4)
        good_jet = dr_clean(good_jet, good_mu, 0.4)
        good_jet = dr_clean(good_jet, good_tau, 0.4)

        good_tau = ak.drop_none(good_tau, axis=1)
        good_jet = ak.drop_none(good_jet, axis=1)

        good_bjet = good_jet[good_jet.btagDeepFlavB > self.cfg['btag_wp']]

        def make_lep_p4(coll):
            return ak.zip({
                "pt": coll.pt,
                "eta": coll.eta,
                "phi": coll.phi,
                "mass": coll.mass,
                "charge": coll.charge,
                "iso": coll.iso
            }, with_name="PtEtaPhiMLorentzVector")
        good_ele = make_lep_p4(good_ele)
        good_mu = make_lep_p4(good_mu)
        # Lepton Merging (Keep 'iso' field)
        leptons = ak.concatenate([good_ele, good_mu], axis=1)
        # Sort by pT for standard selection logic
        leptons = leptons[ak.argsort(leptons.pt, axis=1, ascending=False)]
        lep_q0 = ak.firsts(leptons.charge)  # shape: (nEvents,), empty -> None
        tau_q0 = ak.firsts(good_tau.charge)  # shape: (nEvents,), empty -> None

        opposite_sign = ak.fill_none((lep_q0 * tau_q0) < 0, False)

        # --- 2. Event Selection ---
        selection = PackedSelection()
        selection.add("one_lep", ak.num(leptons) == 1)
        selection.add("one_tau", ak.num(good_tau) == 1)
        selection.add("two_jets", ak.num(good_jet) >= 2)
        selection.add("one_bjets", ak.num(good_bjet) >= 1)
        selection.add("opposite_sign", opposite_sign)
        cut = selection.all("one_lep", "one_tau", "two_jets", "one_bjets", "opposite_sign")

        sel_ev = events[cut]
        if len(sel_ev) == 0: return {
            "cutflow": {k: dict(v) for k, v in temp_cutflow.items()},
            "dqm": dict(temp_dqm)
        }
        temp_cutflow[dataset]["passed"] += len(sel_ev)

        # Slice collections
        sel_leps = leptons[cut]
        sel_taus = good_tau[cut]
        sel_jets = good_jet[cut]
        sel_met = events.MET[cut]

        # --- 3. Train/Valid Split ---
        fhash = abs(hash(filename)) % (2 ** 32 - 1)
        rng = np.random.default_rng(self.cfg['seed'] + fhash)
        is_train = rng.random(len(sel_ev)) < self.cfg['train_frac']

        temp_dqm[dataset].meta["n_train"] += int(np.sum(is_train))
        temp_dqm[dataset].meta["n_valid"] += int(np.sum(~is_train))

        # --- 4. Logic Execution ---

        entry_start = events.metadata['entrystart']
        entry_stop = events.metadata['entrystop']
        output_filename = f"{filename.replace('.root','')}_{entry_start}_{entry_stop}.root"


        # A. EveNet (Full 10 Globals, Max 18 Objs)
        x_ev, mask_ev, glob_ev, dqm_ev = self.get_evenet_features(sel_leps, sel_taus, sel_jets, sel_met)

        dqm_ev["event_weight"] = ak.to_numpy(sel_ev.genWeight / abs(sel_ev.genWeight))

        # B. XGB (Complex Sorting & Logic)
        x_xgb, name_xgb, dqm_xgb = self.get_xgb_features(sel_leps, sel_taus, sel_jets, sel_met)

        # --- 5. Saving ---
        self.save_file(dataset, output_filename, "evenet", ".pt", is_train,
                       {"x": x_ev, "x_mask": mask_ev, "global": glob_ev, "weights": ak.to_numpy(sel_ev.genWeight/abs(sel_ev.genWeight))})

        self.save_file(dataset, output_filename, "xgb", ".npz", is_train,
                       {"X": x_xgb, "features": name_xgb, "weights": ak.to_numpy(sel_ev.genWeight/abs(sel_ev.genWeight))})

        # --- 6. Filling DQM ---
        def fill_dqm(source, mask, split):
            for k, v in source.items():
                if k in HIST_DEFS:
                    temp_dqm[dataset].fill(k, split, v[mask])

        fill_dqm(dqm_ev, is_train, "train")
        fill_dqm(dqm_ev, ~is_train, "valid")
        fill_dqm(dqm_xgb, is_train, "train")
        fill_dqm(dqm_xgb, ~is_train, "valid")

        final_cutflow = {k: dict(v) for k, v in temp_cutflow.items()}
        final_dqm = dict(temp_dqm)

        return {
            "cutflow": final_cutflow,
            "dqm": final_dqm
        }

    def postprocess(self, accumulator):
        return accumulator
    # ==========================================
    # LOGIC A: EveNet (Exact Physics)
    # ==========================================
    def get_evenet_features(self, leptons, taus, jets, met):
        MAX_JETS = self.cfg['max_jets']  # 16
        MAX_OBJS = self.cfg['max_objs']  # 18

        # 1. Jet sorting & Cut (Max 16)
        jets_sorted = jets[ak.argsort(jets.pt, axis=1, ascending=False)]
        jet_btag = ak.values_astype((jets_sorted.btagDeepFlavB > self.cfg['btag_wp']), np.float32)

        # 2. Record Building
        def make_rec(coll, is_lep, is_btag_val, charge=None):
            coll_p4 = ak.zip({
                "pt": coll.pt,
                "eta": coll.eta,
                "phi": coll.phi,
                "mass": coll.mass
            }, with_name="PtEtaPhiMLorentzVector")
            return ak.zip({
                "E": coll_p4.E, "pt": coll.pt, "eta": coll.eta, "phi": coll.phi,
                "btag": is_btag_val,
                "isLepton": ak.full_like(coll.pt, is_lep, dtype=np.float32),
                "charge": ak.values_astype(charge, np.float32) if charge is not None else ak.full_like(coll.pt, 0.0, dtype=np.float32)
            })

        jet_rec = make_rec(jets_sorted, 0.0, jet_btag, ak.full_like(jets_sorted.pt, 0.0, dtype=np.float32))  # Jet: btag=0/1, isLep=0
        tau_rec = make_rec(taus, 0.0, ak.full_like(taus.pt, 2, dtype=np.float32), taus.charge)  # Tau: btag=2,   isLep=0
        lep_rec = make_rec(leptons, 1.0, ak.full_like(leptons.pt, 0, dtype=np.float32), leptons.charge)  # Lep: btag=0,   isLep=1

        # 3. Merge -> Sort -> Pad 18
        all_objs = ak.concatenate([lep_rec, tau_rec, jet_rec], axis=1)
        all_objs = all_objs[ak.argsort(all_objs.pt, axis=1, ascending=False)]

        padded = ak.pad_none(all_objs, MAX_OBJS, axis=1, clip=True)
        mask = (ak.fill_none(padded.pt, -1.0)) > -0.05
        # 4. Safe Numpy Conversion

        def safe(arr): return ak.to_numpy(ak.fill_none(arr, 0.0)).astype(np.float32)

        x_np = np.stack([
            safe(padded.E), safe(padded.pt), safe(padded.eta), safe(padded.phi),
            safe(padded.btag), safe(padded.isLepton), safe(padded.charge)
        ], axis=2)

        # 5. Global Features (Calculated from PADDED array to match tensor)
        # Re-construct 4-vectors from padded arrays
        pad_p4 = ak.zip({
            "pt": ak.fill_none(padded.pt, 0.0),
            "eta": ak.fill_none(padded.eta, 0.0),
            "phi": ak.fill_none(padded.phi, 0.0),
            "E": ak.fill_none(padded.E, 0.0)
        }, with_name="PtEtaPhiELorentzVector")

        is_lepton_padded = ak.fill_none(padded.isLepton, 0.0)
        is_btag_padded = ak.fill_none(padded.btag, 0.0)

        # Masks on padded array
        is_lep_mask = is_lepton_padded > 0.5
        # btag logic: > 0.5 and < 1.5 (Strictly jets, excludes Taus=2.0)
        is_bjet_mask = (is_btag_padded > 0.5) & (is_btag_padded < 1.5)
        # Valid objects (exists and not None)
        valid_mask = ak.fill_none(mask, False)

        # Summation
        def calc_mass(vec_slice):
            return ak.to_numpy(vec_slice.sum(axis=1).mass).astype(np.float32)

        g_met = safe(met.pt)
        g_met_phi = safe(met.phi)
        g_nLep = safe(ak.sum(is_lep_mask, axis=1))
        g_nbJet = safe(ak.sum(is_bjet_mask, axis=1))
        g_nJet = safe(ak.num(jets_sorted) + ak.num(taus))  # Original logic: Jets + Taus count
        g_HT = safe(ak.sum(pad_p4.pt * (~is_lep_mask), axis=1))
        g_HT_lep = safe(ak.sum(pad_p4.pt * is_lep_mask, axis=1))
        g_Mall = calc_mass(ak.mask(pad_p4, valid_mask))  # Sum all valid
        g_Mlep = calc_mass(ak.mask(pad_p4, is_lep_mask))
        g_Mbjet = calc_mass(ak.mask(pad_p4, is_bjet_mask))

        g_Mall = np.where(g_Mall > 0.0, g_Mall, 0.0)
        g_Mlep = np.where(g_Mlep > 0.0, g_Mlep, 0.0)
        g_Mbjet = np.where(g_Mbjet > 0.0, g_Mbjet, 0.0)

        g_np = np.stack([
            g_met, g_met_phi, g_nLep, g_nbJet, g_nJet,
            g_HT, g_HT_lep, g_Mall, g_Mlep, g_Mbjet
        ], axis=1)

        dqm_dict = {
            "x_E": padded.E, "x_pt": padded.pt, "x_eta": padded.eta, "x_phi": padded.phi,
            "x_isLepton": padded.isLepton, "x_isbtag": padded.btag, "x_charge": padded.charge,
            "g_met": g_met, "g_met_phi": g_met_phi, "g_nLepton": g_nLep,
            "g_nbJet": g_nbJet, "g_nJet": g_nJet,
            "g_HT": g_HT, "g_HT_lep": g_HT_lep,
            "g_M_all": g_Mall,
            "g_M_leps": g_Mlep,
            "g_M_bjets": g_Mbjet
        }

        return torch.from_numpy(x_np), torch.from_numpy(safe(mask).astype(bool)), torch.from_numpy(g_np), dqm_dict

    # ==========================================
    # LOGIC B: XGBoost (Exact Complex Logic)
    # ==========================================
    def get_xgb_features(self, leptons, taus, jets, met):
        # 1. Lepton Selection: Iso Sort (Iso*1e6 - Pt)
        # Note: 'leptons' passed here has 'iso' attached from process()
        l1 = ak.firsts(leptons)

        # 2. Tau Selection: Maximize Pt(lep + tau)
        # Need to broadcast l1 to all taus to calc pair pt
        l1_p4 = ak.zip({"pt": l1.pt, "eta": l1.eta, "phi": l1.phi, "mass": l1.mass}, with_name="PtEtaPhiMLorentzVector")
        taus_p4 = ak.zip({"pt": taus.pt, "eta": taus.eta, "phi": taus.phi, "mass": taus.mass},
                         with_name="PtEtaPhiMLorentzVector")

        l1_broad, taus_broad = ak.broadcast_arrays(l1_p4, taus_p4)
        pair_pt = (l1_broad + taus_broad).pt
        # Sort taus by this pair_pt descending
        t1 = ak.firsts(taus[ak.argsort(pair_pt, axis=1, ascending=False)])

        # 3. Jet Selection (The Tree Logic)
        jets_pt_sort = jets[ak.argsort(jets.pt, axis=1, ascending=False)]
        is_btag = jets_pt_sort.btagDeepFlavB > self.cfg['btag_wp']
        bjets = jets_pt_sort[is_btag]
        ljets = jets_pt_sort[~is_btag]  # Light jets

        n_b = ak.num(bjets)

        # Prepare candidates (Pad to 2)
        b_pad = ak.pad_none(bjets, 2)
        l_pad = ak.pad_none(ljets, 2)
        all_pad = ak.pad_none(jets_pt_sort, 2)

        # Logic Branching
        # Case 2: >= 2 bjets -> b1=b[0], b2=b[1]
        # Case 1: == 1 bjets -> b1=b[0], b2=light[0]
        # Case 0: == 0 bjets -> b1=all[0], b2=all[1]

        b1 = ak.where(n_b >= 2, b_pad[:, 0], ak.where(n_b == 1, b_pad[:, 0], all_pad[:, 0]))
        b2 = ak.where(n_b >= 2, b_pad[:, 1], ak.where(n_b == 1, l_pad[:, 0], all_pad[:, 1]))

        # Fallback: If b2 is None (e.g. n_b=1 but no light jets), fallback to 2nd jet overall
        b2 = ak.where(ak.is_none(b2), all_pad[:, 1], b2)

        # 4. Feature Calculation
        # Wrap selected objects
        def get_p4(obj):
            return ak.zip({"pt": obj.pt, "eta": obj.eta, "phi": obj.phi, "mass": obj.mass},
                          with_name="PtEtaPhiMLorentzVector")

        t1_p4 = get_p4(t1)
        b1_p4 = get_p4(b1)
        b2_p4 = get_p4(b2)
        met_p4 = ak.zip({"pt": met.pt, "eta": ak.zeros_like(met.pt), "phi": met.phi, "mass": ak.zeros_like(met.pt)},
                        with_name="PtEtaPhiMLorentzVector")

        # Variables
        mbb = (b1_p4 + b2_p4).mass
        # mHH proxy: l + t + b + b + met
        mHH = (l1_p4 + t1_p4 + b1_p4 + b2_p4 + met_p4).mass
        m_tautau_visible = (l1_p4 + t1_p4).mass

        dr_ltau = l1_p4.delta_r(t1_p4)
        dr_bb = b1_p4.delta_r(b2_p4)
        dpt_ltau = abs(l1_p4.pt - t1_p4.pt)
        b2_pt = b2.pt
        b1_pt = b1.pt
        l1_pt = l1.pt
        t1_pt = t1.pt

        dphi_l_met = abs(l1_p4.delta_phi(met_p4))
        mT_W = np.sqrt(2 * l1.pt * met.pt * (1 - np.cos(dphi_l_met)))

        ltau_sys = l1_p4 + t1_p4
        bb_sys = b1_p4 + b2_p4
        dphi_ltau_bb = abs(ltau_sys.delta_phi(bb_sys))

        mb1l = (b1_p4 + l1_p4).mass
        mb1t = (b1_p4 + t1_p4).mass
        mb2l = (b2_p4 + l1_p4).mass
        mb2t = (b2_p4 + t1_p4).mass
        dr_b1l = b1_p4.delta_r(l1_p4)
        dr_b1t = b1_p4.delta_r(t1_p4)
        dr_b2l = b2_p4.delta_r(l1_p4)
        dr_b2t = b2_p4.delta_r(t1_p4)


        def safe(arr): return ak.to_numpy(ak.fill_none(arr, -999.0)).astype(np.float32)

        data = {
            "xgb_mHH_proxy": safe(mHH), "xgb_mbb": safe(mbb),
            "xgb_mtautau_visible": safe(m_tautau_visible),
            "xgb_dR_ltau": safe(dr_ltau), "xgb_dR_bb": safe(dr_bb),
            "xgb_dpt_ltau": safe(dpt_ltau), "xgb_b2_pt": safe(b2_pt), "xgb_b1_pt": safe(b1_pt),
            "xgb_mT_W": safe(mT_W), "xgb_met": safe(met.pt),
            "xgb_dphi_ltau_bb": safe(dphi_ltau_bb), "xgb_dphi_l_met": safe(dphi_l_met),
            "l1_pt": safe(l1_pt), "t1_pt": safe(t1_pt),
            "xgb_mb1l": safe(mb1l), "xgb_mb1t": safe(mb1t),
            "xgb_mb2l": safe(mb2l), "xgb_mb2t": safe(mb2t),
            "xgb_dr_b1l": safe(dr_b1l), "xgb_dr_b1t": safe(dr_b1t),
            "xgb_dr_b2l": safe(dr_b2l), "xgb_dr_b2t": safe(dr_b2t)
        }

        matrix = np.stack(list(data.values()), axis=1)
        # Raw arrays for DQM
        dqm_dict = {
            "xgb_mHH_proxy": mHH, "xgb_mbb": mbb, "xgb_dR_ltau": dr_ltau, "xgb_dR_bb": dr_bb,
            "xgb_dpt_ltau": dpt_ltau, "xgb_b2_pt": b2_pt, "xgb_mT_W": mT_W, "xgb_met": met.pt,
            "xgb_dphi_ltau_bb": dphi_ltau_bb, "xgb_dphi_l_met": dphi_l_met, "xgb_mtautau_visible": m_tautau_visible, "xgb_b1_pt": b1_pt,
            "l1_pt": l1_pt, "t1_pt": t1_pt,
            "xgb_mb1l": mb1l, "xgb_mb1t": mb1t,
            "xgb_mb2l": mb2l, "xgb_mb2t": mb2t,
            "xgb_dr_b1l": dr_b1l, "xgb_dr_b1t": dr_b1t,
            "xgb_dr_b2l": dr_b2l, "xgb_dr_b2t": dr_b2t
        }

        return matrix, list(data.keys()), dqm_dict

    # ==========================================
    # Helper: Save
    # ==========================================
    def save_file(self, dataset, filename, subdir, ext, train_mask, data_dict):
        base = Path(self.cfg['outdir']) / dataset / subdir
        base.mkdir(parents=True, exist_ok=True)
        stem = Path(filename).stem
        # uid = str(uuid.uuid4())[:6]

        for split, mask in [("train", train_mask), ("valid", ~train_mask)]:
            out_dir = base / split
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / f"{stem}_{ext}"

            sliced = {}
            for k, v in data_dict.items():
                if isinstance(v, list):
                    sliced[k] = v
                elif isinstance(v, torch.Tensor):
                    sliced[k] = v[torch.from_numpy(mask)]
                else:
                    sliced[k] = v[mask]

            if ext == ".pt":
                torch.save(sliced, out_path)
            else:
                np.savez_compressed(out_path, **sliced)


# ==========================================
# Main
# ==========================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("json", help="Input JSON {dataset: [files]}")
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    with open(args.json) as f: fileset = json.load(f)

    fileset_for_coffea = {}
    for category, sample_dict in fileset.items():
        for name, sample_list in sample_dict.items():
            fileset_for_coffea[name] = {
                "files": sample_list,
                "metadata": {"sample_name": name}
            }

    config = {
        "outdir": args.outdir,
        "btag_wp": 0.2489,  # 2016postapv WPs, medium
        "max_objs": 18,
        "max_jets": 16,
        "seed": 123,
        "train_frac": 0.5
    }

    print(f"Running Full Logic Processor with {args.workers} workers...")
    runner = Runner(
        executor=FuturesExecutor(workers=args.workers),
        schema=NanoAODSchema,
        chunksize=100_000
    )

    # Run by calling the runner instance
    output = runner(
        fileset_for_coffea,
        treename="Events",
        processor_instance=FullLogicProcessor(config)
    )


    print("\nProcessing Complete.")
    print("\n=== Cutflow Summary ===")
    print(f"{'Dataset':<20} | {'Total':<10} | {'Passed':<10} | {'Eff (%)':<10}")
    print("-" * 60)

    for dataset, counts in output["cutflow"].items():
        total = counts["total"]
        passed = counts["passed"]
        eff = 100 * passed / total if total > 0 else 0
        print(f"{dataset:<20} | {total:<10} | {passed:<10} | {eff:<10.2f}")

    # --- Save Cutflow JSON ---
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    cutflow_out = {
        ds: {k: int(v) for k, v in counts.items()}
        for ds, counts in output["cutflow"].items()
    }

    cutflow_path = outdir / "cutflow.json"
    with open(cutflow_path, "w") as f:
        json.dump(cutflow_out, f, indent=2, sort_keys=True)

    print(f"\nCutflow saved to: {cutflow_path}")

    # --- Save DQM (NPZ + Plots) ---
    dqm_dir = Path(args.outdir) / "dqm_summary"
    dqm_dir.mkdir(parents=True, exist_ok=True)

    print("\n=== Saving DQM and Generating Plots ===")

    for name, obj in output["dqm"].items():
        print(f"Processing {name}...")

        # save npz
        final_dqm_data = {
            "hists": obj.hists,
            "meta": obj.meta,
            "hist_defs": obj.hist_defs
        }
        np.savez(dqm_dir / f"dqm_{name}.npz", **final_dqm_data)

        # generate plots
        plot_dqm(final_dqm_data, name, dqm_dir)

    print(f"\nAll done! Check plots in: {dqm_dir}")
if __name__ == "__main__":
    main()