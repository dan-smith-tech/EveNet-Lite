import re
from pathlib import Path
from typing import Optional, Dict, Union
import awkward as ak
import json
from collections import OrderedDict
from multiprocessing import Pool

import torch
from tqdm import tqdm
import numpy as np
import uproot
from collections import defaultdict
import vector

vector.register_awkward()

SIG_PROCESS = "NMSSM_XToYHTo2B2Tau"

SIG_PATTERN = re.compile(
    r".*_MX(?P<mx>\d+).*_m35-(?P<m35>\d+).*\.root$"
)
GEN_PATTERN = re.compile(r"(?P<proc>.+)_(?P<seed>\d+)\.root$")


def parse_filename(path) -> Dict[str, object]:
    p = Path(path)  # normalize: works for str or Path
    name = p.name  # only the filename, no directory

    m = SIG_PATTERN.match(name)
    if m:
        out = {
            "is_signal": True, "path": str(p), "proc": 'NMSSM_XToYHTo2B2Tau',
            "m1": int(m.group(1)),
            "m2": int(m.group(2))
        }
        return out

    m = GEN_PATTERN.match(name)
    if m:
        out = m.groupdict()
        out["is_signal"] = False
        out["path"] = str(p)
        out["proc"] = str(out["proc"])
        return out

    raise ValueError(f"Unrecognized filename: {p}")


def vec(
        arr, pre, *, m=0.0, mom_names=("PT", "Eta", "Phi"), pdg=None, q=None, extra=(),
        pt_cut=None, eta_cut=None,
):
    pt, eta, phi = (arr[f"{pre}.{k}"] for k in mom_names)
    mass = arr[f"{pre}.Mass"] if f"{pre}.Mass" in arr.fields else ak.full_like(pt, m)
    # -----------------------------
    # object-level selection
    # -----------------------------
    mask = ak.ones_like(pt, dtype=bool)
    if pt_cut is not None:
        mask = mask & (pt > pt_cut)
    if eta_cut is not None:
        mask = mask & (abs(eta) < eta_cut)

    pt = pt[mask]
    eta = eta[mask] if pre != "MissingET" else ak.zeros_like(pt)
    phi = phi[mask]
    mass = mass[mask]
    data = dict(pt=pt, eta=eta, phi=phi, mass=mass)

    if q is not None:
        data["charge"] = ak.full_like(pt, q)
    elif f"{pre}.Charge" in arr.fields:
        data["charge"] = arr[f"{pre}.Charge"][mask]

    if f"{pre}.PDG" in arr.fields or pdg is not None:
        data["pdg"] = arr[f"{pre}.PDG"][mask] if f"{pre}.PDG" in arr.fields else ak.full_like(pt, pdg)

    if pre == "Electron":
        data["pdg"] = arr[f"{pre}.Charge"][mask] * (-11)
    if pre == "Muon":
        data["pdg"] = arr[f"{pre}.Charge"][mask] * (-13)

    for k in extra:
        key = f"{pre}.{k}"
        if key in arr.fields:
            data[k] = arr[key][mask]

    vecs = ak.zip(data, with_name="Momentum4D")

    return vecs


def build_objects(arr):
    TAU_MASS = 1.77686  # GeV
    ELE_MASS = 0.000510998  # GeV
    MU_MASS = 0.105658375  # GeV

    jet_cut = dict(pt_cut=20, eta_cut=2.4)
    lep_cut = dict(pt_cut=25, eta_cut=2.4)
    tau_cut = dict(pt_cut=30, eta_cut=2.4)

    jets = vec(arr, "Jet", q=0, pdg=0, extra=("BTag", "TauTag", "NCharged", "NNeutrals"), **jet_cut)
    els = vec(arr, "Electron", extra=("D0", "DZ", "ErrorD0", "ErrorDZ"), m=ELE_MASS, **lep_cut)
    mus = vec(arr, "Muon", extra=("D0", "DZ", "ErrorD0", "ErrorDZ"), m=MU_MASS, **lep_cut)
    met = vec(arr, "MissingET", mom_names=("MET", "Eta", "Phi"), m=0.0, q=0, pdg=0)

    taus = vec(arr, "Jet", pdg=0, extra=("BTag", "TauTag", "NCharged", "NNeutrals"), **tau_cut)
    taus = taus[taus.TauTag == 1]
    taus = ak.with_field(taus, ak.full_like(taus.pt, TAU_MASS), "mass")
    taus = ak.with_field(taus, ak.full_like(taus.pt, 15) * taus.charge, "pdg")

    return {
        "jets": jets,
        "electrons": els,
        "muons": mus,
        "leptons": ak.concatenate([els, mus], axis=1),
        "taus": taus,
        "bjets": jets[(jets.BTag == 1) & (jets.TauTag != 1)],
        "ljets": jets[(jets.BTag == 0) & (jets.TauTag != 1)],
        "met": met,
    }


CUTFLOW_STEPS = OrderedDict([
    ("all", None),
    ("1_lepton", lambda o: ak.num(o["leptons"]) == 1),
    ("1_tau", lambda o: ak.num(o["taus"]) == 1),
    (">=1_bjet", lambda o: ak.num(o["bjets"]) >= 1),
])


def apply_cutflow(objects):
    mask = ak.ones_like(ak.num(objects["leptons"]), dtype=bool)
    cutflow = OrderedDict()

    for name, func in CUTFLOW_STEPS.items():
        if func is not None:
            mask = mask & func(objects)
        cutflow[name] = int(ak.sum(mask))

    return mask, cutflow


def build_event_tensor(objs):
    MAX_PART = 18
    N_FEAT = 7

    def write_particles(x, mask, offset, parts, *, tag=0, lep=0):
        """
        Write particles into x[:, :, :] starting at column offset.

        parts: awkward array of Momentum4D with fields
        tag  : 0/1/2 (none/b/tau)
        lep  : 0/1/2 (none/e/mu)

        Returns new offset.
        """
        n_evt = len(parts)

        for i in range(n_evt):
            objs = parts[i]
            n = min(len(objs), MAX_PART - offset[i])

            if n <= 0:
                continue

            sl = slice(offset[i], offset[i] + n)

            x[i, sl, 0] = objs.energy[:n]
            x[i, sl, 1] = objs.pt[:n]
            x[i, sl, 2] = objs.eta[:n]
            x[i, sl, 3] = objs.phi[:n]
            x[i, sl, 4] = tag
            x[i, sl, 5] = lep
            x[i, sl, 6] = objs.charge[:n]

            mask[i, sl] = True
            offset[i] += n

        return offset

    def build_x_and_mask(objs):
        n_evt = len(objs["bjets"])

        x = np.zeros((n_evt, MAX_PART, N_FEAT), dtype=np.float32)
        x_mask = np.zeros((n_evt, MAX_PART), dtype=bool)
        offset = np.zeros(n_evt, dtype=np.int32)

        # ---- sort once in awkward ----
        leptons = objs["leptons"][ak.argsort(objs["leptons"].pt, axis=1, ascending=False)]
        taus = objs["taus"][ak.argsort(objs["taus"].pt, axis=1)]
        bjets = objs["bjets"][ak.argsort(objs["bjets"].pt, axis=1, ascending=False)]
        jets = objs["ljets"][ak.argsort(objs["ljets"].pt, axis=1, ascending=False)]

        lep_e = leptons[abs(leptons.pdg) == 11]
        lep_m = leptons[abs(leptons.pdg) == 13]

        # ---- fill numpy buffers ----
        offset = write_particles(x, x_mask, offset, lep_e, lep=1)
        offset = write_particles(x, x_mask, offset, lep_m, lep=2)
        offset = write_particles(x, x_mask, offset, bjets, tag=1)
        offset = write_particles(x, x_mask, offset, jets)
        write_particles(x, x_mask, offset, taus, tag=2)

        return x, x_mask

    def build_globals(objs):
        jets = objs["jets"]
        leptons = objs["leptons"]
        bjets = objs["bjets"]

        def inv_mass(x):
            return ak.fill_none((ak.sum(x, axis=1)).mass, 0.0)

        globals = np.stack([
            ak.to_numpy(objs["met"].pt).flatten(),
            ak.to_numpy(objs["met"].phi).flatten(),
            ak.to_numpy(ak.num(leptons)),
            ak.to_numpy(ak.num(bjets)),
            ak.to_numpy(ak.num(jets)),
            ak.to_numpy(ak.sum(jets.pt, axis=1)),
            ak.to_numpy(ak.sum(leptons.pt, axis=1)),
            ak.to_numpy(inv_mass(jets)),
            ak.to_numpy(inv_mass(leptons)),
            ak.to_numpy(inv_mass(bjets)),
        ], axis=1)

        return globals

    x, x_mask = build_x_and_mask(objs)
    globals = build_globals(objs)

    return {
        "x": torch.tensor(x, dtype=torch.float32),
        "x_mask": torch.tensor(x_mask, dtype=torch.bool),
        "globals": torch.tensor(globals, dtype=torch.float32),
        "weights": torch.ones(len(x), dtype=torch.float32)
    }


def build_event_tabular(objs):
    l1 = ak.firsts(objs['leptons'])
    tau1 = ak.firsts(objs['taus'])
    met = ak.firsts(objs["met"])

    bjets = objs["bjets"]
    ljets = objs["ljets"]
    # pT ordering (descending)
    bjets = bjets[ak.argsort(bjets.pt, axis=1, ascending=False)]
    ljets = ljets[ak.argsort(ljets.pt, axis=1, ascending=False)]
    nb = ak.num(bjets)
    nl = ak.num(ljets)
    zero_jet = ak.zip(
        {field: ak.zeros_like(ak.firsts(bjets).pt) for field in bjets.fields},
        with_name="Momentum4D",
    )
    b1 = ak.where(nb >= 1, ak.firsts(bjets), zero_jet, )
    b2 = ak.where(
        nb >= 2,
        ak.pad_none(bjets, 2)[:, 1],  # Case: ≥2 b-jets
        ak.where(
            (nb == 1) & (nl >= 1),
            ak.firsts(ak.pad_none(ljets, 1)),  # Case: 1 b-jet + ≥1 light jet
            zero_jet,  # Otherwise
        ),
    )

    # 4. Feature Calculation
    # Wrap selected objects
    def get_p4(obj):
        return ak.zip(
            {"pt": obj.pt, "eta": obj.eta, "phi": obj.phi, "mass": obj.mass},
            with_name="Momentum4D"
        )

    l1_p4 = get_p4(l1)
    t1_p4 = get_p4(tau1)
    b1_p4 = get_p4(b1)
    b2_p4 = get_p4(b2)
    met_p4 = get_p4(met)

    # Variables
    mbb = (b1_p4 + b2_p4).mass
    # mHH proxy: l + t + b + b + met
    mHH = (l1_p4 + t1_p4 + b1_p4 + b2_p4 + met_p4).mass
    m_tautau_visible = (l1_p4 + t1_p4).mass

    dr_ltau = l1_p4.deltaR(t1_p4)
    dr_bb = b1_p4.deltaR(b2_p4)
    dpt_ltau = abs(l1_p4.pt - t1_p4.pt)
    b2_pt = b2.pt
    b1_pt = b1.pt
    l1_pt = l1.pt
    t1_pt = tau1.pt

    dphi_l_met = abs(l1_p4.deltaphi(met_p4))
    mT_W = np.sqrt(2 * l1.pt * met.pt * (1 - np.cos(dphi_l_met)))

    ltau_sys = l1_p4 + t1_p4
    bb_sys = b1_p4 + b2_p4
    dphi_ltau_bb = abs(ltau_sys.deltaphi(bb_sys))

    mb1l = (b1_p4 + l1_p4).mass
    mb1t = (b1_p4 + t1_p4).mass
    mb2l = (b2_p4 + l1_p4).mass
    mb2t = (b2_p4 + t1_p4).mass
    dr_b1l = b1_p4.deltaR(l1_p4)
    dr_b1t = b1_p4.deltaR(t1_p4)
    dr_b2l = b2_p4.deltaR(l1_p4)
    dr_b2t = b2_p4.deltaR(t1_p4)

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

    return data


def save_shard(event_tensor, outdir, shard_id, folder_name, method):
    n = len(next(iter(event_tensor.values())))
    splits = {
        "train": np.arange(n)[::2],
        "valid": np.arange(n)[1::2],
    }

    for split, idx in splits.items():
        out = {}
        for k, v in event_tensor.items():
            if isinstance(v, list):
                out[k] = v
            else:
                out[k] = v[idx]
        if method == "evenet":
            d = outdir / method / split
            d.mkdir(parents=True, exist_ok=True)
            torch.save(out, d / f"shard_{folder_name}_{shard_id:04d}.pt")
        if method == "xgb":
            d = outdir / method / split
            d.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(d / f"shard_{folder_name}_{shard_id:04d}.npz", **out)


def discover_processes(grid_dir: Path):
    """
    Returns:
      dict: process_key -> list[Path]
    process_key includes mass-point info if present
    """
    processes = defaultdict(list)

    for f in grid_dir.glob("*.root"):
        meta = parse_filename(f)

        if meta["is_signal"]:
            key = (meta["proc"], meta["m1"], meta["m2"],)
        else:
            key = (meta["proc"],)

        processes[key].append(f)

    return processes


def process_one_process(task):
    """
    task = (process_key, files, outdir, chunk_size)
    """
    process_key, files, outdir, chunk_size, folder_name = task

    cutflow_total = defaultdict(int)
    shard_id = 0

    FILTER_BRANCHES = [
        # Jets
        "Jet/Jet.PT",
        "Jet/Jet.Eta",
        "Jet/Jet.Phi",
        "Jet/Jet.Mass",
        "Jet/Jet.BTag",
        "Jet/Jet.TauTag",
        "Jet/Jet.Charge",
        "Jet/Jet.NCharged",
        "Jet/Jet.NNeutrals",

        # Electrons
        "Electron/Electron.PT",
        "Electron/Electron.Eta",
        "Electron/Electron.Phi",
        "Electron/Electron.Charge",
        "Electron/Electron.Mass",
        "Electron/Electron.PDG",
        "Electron/Electron.D0",
        "Electron/Electron.DZ",
        "Electron/Electron.ErrorD0",
        "Electron/Electron.ErrorDZ",

        # Muons
        "Muon/Muon.PT",
        "Muon/Muon.Eta",
        "Muon/Muon.Phi",
        "Muon/Muon.Charge",
        "Muon/Muon.Mass",
        "Muon/Muon.PDG",
        "Muon/Muon.D0",
        "Muon/Muon.DZ",
        "Muon/Muon.ErrorD0",
        "Muon/Muon.ErrorDZ",

        # MET
        "MissingET/MissingET.MET",
        "MissingET/MissingET.Eta",
        "MissingET/MissingET.Phi",
    ]

    all_inputs_evenet = []
    all_input_tabular = []

    valid_files = []
    for f in files:
        try:
            with uproot.open(f) as root:
                if "Delphes" in root:
                    valid_files.append(f)
                else:
                    print(f"[skip] no Delphes tree: {f}")
        except Exception as e:
            print(f"[skip] broken file: {f} ({e})")

    for arrays, report in uproot.iterate(
            [f"{f}:Delphes" for f in valid_files],
            filter_name=FILTER_BRANCHES,
            step_size=chunk_size,
            library="ak",
            report=True,
    ):
        objects = build_objects(arrays)
        mask, cutflow = apply_cutflow(objects)

        for k, v in cutflow.items():
            cutflow_total[k] += v

        if ak.sum(mask) == 0:
            continue

        meta = {
            "proc": process_key[0],
            "is_signal": len(process_key) == 3,
            "m1": process_key[1] if len(process_key) == 3 else -1,
            "m2": process_key[2] if len(process_key) == 3 else -1,
        }

        event_tensor = {
            # objects
            "leptons": objects["leptons"][mask],
            "taus": objects["taus"][mask],
            "bjets": objects["bjets"][mask],
            "ljets": objects["ljets"][mask],
            "met": objects["met"][mask],
            "jets": objects["jets"][mask],
        }

        input_evenet = build_event_tensor(event_tensor)
        all_inputs_evenet.append(input_evenet)

        input_tabular = build_event_tabular(event_tensor)
        all_input_tabular.append(input_tabular)

    input_tensor_evenet = {
        k: torch.concat([d[k] for d in all_inputs_evenet], dim=0)
        for k in all_inputs_evenet[0].keys()
    }
    input_tabular = {
        k: np.concat([d[k] for d in all_input_tabular], axis=0)
        for k in all_input_tabular[0].keys()
    }
    input_tensor_tabular = {
        "X": np.stack(list(input_tabular.values()), axis=1),
        "features": list(input_tabular.keys()),
    }
    input_tensor_tabular["weights"] = np.ones(len(input_tensor_tabular["X"]), dtype=float)

    save_shard(input_tensor_evenet, outdir, shard_id, folder_name, method="evenet")
    save_shard(input_tensor_tabular, outdir, shard_id, folder_name, method="xgb")

    return process_key, cutflow_total, outdir


def build_all_tasks(
        input_root: Path,
        output_root: Path,
        folder_structure: str,
        chunk_size,
):
    tasks = []

    for grid_dir in sorted(input_root.glob(folder_structure)):
        processes = discover_processes(grid_dir)
        rel = grid_dir.relative_to(input_root)
        grid_name = "_".join(rel.parts)

        for key, files in processes.items():
            if len(key) == 3:
                proc, m1, m2 = key
                outdir = output_root / f"MX-{m1}_MY-{m2}"
            else:
                proc = key[0]
                outdir = output_root / proc

            tasks.append(
                (key, files, outdir, chunk_size, grid_name)
            )

    return tasks


def run_all_grids(
        input_root,
        output_root,
        folder_structure="grid_*",
        n_workers=8,
        chunk_size="1000MB",
):
    input_root = Path(input_root)
    output_root = Path(output_root)

    tasks = build_all_tasks(
        input_root,
        output_root,
        folder_structure,
        chunk_size,
    )

    cutflows = defaultdict(lambda: {
        "out": None,
        "cutflow": defaultdict(int),
    })

    with Pool(n_workers) as pool:
        for process_key, cutflow, outdir in tqdm(
                pool.imap_unordered(process_one_process, tasks),
                total=len(tasks),
                desc="Processing all grids",
        ):
            cutflows[process_key]["out"] = outdir

            for cut, count in cutflow.items():
                cutflows[process_key]["cutflow"][cut] += count

    # write outputs
    for process_key, info in cutflows.items():
        outdir = info["out"]
        outdir.mkdir(parents=True, exist_ok=True)
        with open(outdir / "cutflow.json", "w") as f:
            json.dump(info["cutflow"], f, indent=2)

    return cutflows


if __name__ == '__main__':
    import argparse


    def parse_args():
        parser = argparse.ArgumentParser(
            description="Run EveNet-Lite grid processing."
        )

        parser.add_argument(
            "--in-dir",
            required=True,
            type=Path,
            help="Input directory containing grid folders",
        )

        parser.add_argument(
            "--out-dir",
            required=True,
            type=Path,
            help="Output directory",
        )

        parser.add_argument(
            "--folder-structure",
            default="grid_*/results",
            help="Glob pattern for grid subfolders",
        )

        parser.add_argument(
            "--chunk-size",
            default="100 MB",
            help="Chunk size (e.g. '100 MB', '1 GB')",
        )

        parser.add_argument(
            "--n-workers",
            type=int,
            default=8,
            help="Number of worker processes",
        )

        return parser.parse_args()


    args = parse_args()

    run_all_grids(
        input_root=args.in_dir,
        output_root=args.out_dir,
        folder_structure=args.folder_structure,
        chunk_size=args.chunk_size,
        n_workers=args.n_workers,
    )

    # in_dir = "/Users/avencastmini/PycharmProjects/EveNet-Lite/workspace/new_grid/"
    # out_dir = "/Users/avencastmini/PycharmProjects/EveNet-Lite/workspace/new_grid.output"
    #
    # run_all_grids(in_dir, out_dir, folder_structure="test_*/results", chunk_size="100 MB", n_workers=8)
