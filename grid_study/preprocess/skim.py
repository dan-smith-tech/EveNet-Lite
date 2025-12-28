import re
from pathlib import Path
from typing import Optional, Dict, Union
import awkward as ak
import json
from collections import OrderedDict
from multiprocessing import Pool
from tqdm import tqdm
import numpy as np
import uproot
from collections import defaultdict
import vector

vector.register_awkward()

SIG_PROCESS = "NMSSM_XToYHTo2B2Tau"

SIG_PATTERN = re.compile(
    r"NMSSM_XToYHTo2B2Tau_(?P<seed>\d+)_m35-(?P<m1>\d+)_m45-(?P<m2>\d+)\.root"
)
GEN_PATTERN = re.compile(
    r"(?P<proc>[^_]+)_(?P<seed>\d+)\.root"
)


def parse_filename(path) -> Dict[str, object]:
    p = Path(path)  # normalize: works for str or Path
    name = p.name  # only the filename, no directory

    m = SIG_PATTERN.match(name)
    if m:
        out = m.groupdict()
        out["is_signal"] = True
        out["path"] = str(p)
        out["proc"] = 'NMSSM_XToYHTo2B2Tau'
        out["seed"] = int(out["seed"])
        out["m1"] = int(out["m1"])
        out["m2"] = int(out["m2"])
        return out

    m = GEN_PATTERN.match(name)
    if m:
        out = m.groupdict()
        out["is_signal"] = False
        out["path"] = str(p)
        out["proc"] = str(out["proc"])
        out["seed"] = int(out["seed"])
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

    pt   = pt[mask]
    eta  = eta[mask]
    phi  = phi[mask]
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
        "x": x.astype(np.float32),
        "x_mask": x_mask.astype(bool),
        "globals": globals.astype(np.float32),
    }


def save_shard(event_tensor, outdir, shard_id, folder_name):
    n = len(next(iter(event_tensor.values())))
    splits = {
        "train": np.arange(n)[::2],
        "valid": np.arange(n)[1::2],
    }

    for split, idx in splits.items():
        out = {k: v[idx] for k, v in event_tensor.items()}
        d = outdir / split
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

        # process key uniquely identifies physics content
        if meta["is_signal"]:
            key = (
                meta["proc"],
                meta["m1"],
                meta["m2"],
            )
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

    (outdir / "train").mkdir(parents=True, exist_ok=True)
    (outdir / "valid").mkdir(parents=True, exist_ok=True)

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

    all_inputs = []

    for arrays, report in uproot.iterate(
            [f"{f}:Delphes" for f in files],
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

            # metadata (broadcast-safe)
            "process": meta["proc"],
            "is_signal": meta["is_signal"],
            "m1": meta.get("m1", -1),
            "m2": meta.get("m2", -1),
        }

        input_tensor = build_event_tensor(event_tensor)
        all_inputs.append(input_tensor)

    keys = all_inputs[0].keys()

    input_tensor = {
        k: np.concatenate([d[k] for d in all_inputs], axis=0)
        for k in keys
    }
    save_shard(input_tensor, outdir, shard_id, folder_name)

    with open(outdir / "cutflow.json", "w") as f:
        json.dump(cutflow_total, f, indent=2)

    return process_key


def process_grid_folder(
        grid_dir: Path,
        output_root: Path,
        n_workers: int,
        chunk_size: Union[int, str],
):
    processes = discover_processes(grid_dir)

    tasks = []
    for key, files in processes.items():
        if len(key) == 3:
            proc, m1, m2 = key
            outdir = output_root / f"{proc}_m35-{m1}_m45-{m2}"
        else:
            proc = key[0]
            # outdir = output_root / grid_dir.name / proc
            outdir = output_root / proc

        tasks.append((key, files, outdir, chunk_size, grid_dir.name))

    with Pool(n_workers) as pool:
        for _ in tqdm(
                pool.imap_unordered(process_one_process, tasks),
                total=len(tasks),
                desc=f"{grid_dir.name}",
        ):
            pass


def run_all_grids(
        input_root,
        output_root,
        folder_structure="grid_*",
        n_workers=8,
        chunk_size="1000MB",
):
    input_root = Path(input_root)
    output_root = Path(output_root)

    for grid_dir in sorted(input_root.glob(folder_structure)):
        print(f"\n=== Processing {grid_dir.name} ===")
        process_grid_folder(
            grid_dir,
            output_root,
            n_workers,
            chunk_size,
        )


if __name__ == '__main__':
    in_dir = "/Users/avencastmini/PycharmProjects/EveNet-Lite/workspace/new_grid/"
    out_dir = "/Users/avencastmini/PycharmProjects/EveNet-Lite/workspace/new_grid.output"

    run_all_grids(in_dir, out_dir, folder_structure="test_*", chunk_size="100 MB", n_workers=8)
