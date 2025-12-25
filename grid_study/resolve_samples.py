#!/usr/bin/env python3
import yaml
import json
import re
import subprocess
import requests
import concurrent.futures
from tqdm import tqdm
import argparse

# --- Configuration ---
COD_API = "https://opendata.cern.ch/api/records"
MAX_WORKERS = 8


def get_cod_records(query_wildcard):
    """Searches CERN Open Data for NANOAODSIM datasets."""
    # Forced filters: NANOAODSIM, CMS, Online, Dataset
    full_query = f'{query_wildcard} AND NANOAOD AND experiment:CMS AND availability:online'

    params = {'q': full_query, 'size': 1000, 'page': 1}
    response = requests.get(COD_API, params=params)
    response.raise_for_status()

    data = response.json()
    hits = data.get('hits', {}).get('hits', [])

    records = []
    for hit in hits:
        records.append({
            'recid': hit['metadata']['recid'],
            'title': hit['metadata']['title']
        })
    return records


def extract_mass(title):
    """Parses MX and MY from title."""
    mx_match = re.search(r"MX[-_]?([0-9]+)", title, re.IGNORECASE)
    my_match = re.search(r"MY[-_]?([0-9]+)", title, re.IGNORECASE)

    mx = int(mx_match.group(1)) if mx_match else None
    my = int(my_match.group(1)) if my_match else None
    return mx, my


def resolve_file_urls(recid):
    """Resolves root:// paths for a single recid."""
    try:
        cmd = [
            "cernopendata-client", "get-file-locations",
            "--recid", str(recid),
            "--protocol", "xrootd"
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)

        urls = []
        for line in result.stdout.splitlines():
            line = line.strip()
            if line.startswith("root://"):
                urls.append(line)
        return urls
    except Exception as e:
        print(f"!! Error resolving recid {recid}: {e}")
        return []


def worker_resolve_named_dataset(name, recids):
    """Helper for thread pool: Resolves a list of recids and returns (name, urls)."""
    all_urls = []
    for rid in recids:
        all_urls.extend(resolve_file_urls(rid))
    return name, sorted(list(set(all_urls)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--yaml", required=True, help="input YAML configuration file")
    ap.add_argument("--output", default="file_locations.json", help="output JSON file")
    args = ap.parse_args()
    input_yaml = args.yaml
    output_json = args.output

    with open(input_yaml, 'r') as f:
        config = yaml.safe_load(f)

    final_output = {
        "signal": {},
        "background": {}
    }

    # Task list for parallel execution: [(key, [recid1, recid2]), ...]
    # We differentiate signal tasks and background tasks
    signal_tasks = []  # will hold: ( "MX-700_MY-400", [12345] )
    background_tasks = []  # will hold: ( "ttbar", [67890, 67891] )

    # --- 1. Prepare Signal Tasks ---
    if 'signal' in config:
        print("-> Searching Signal...")
        sig_conf = config['signal']
        records = get_cod_records(sig_conf['wildcard'])
        mx_range = sig_conf.get('mx', [0, 99999])
        my_range = sig_conf.get('my', [0, 99999])

        for r in records:
            mx, my = extract_mass(r['title'])
            if mx is None or my is None:
                continue

            if (mx_range[0] <= mx <= mx_range[1]) and (my_range[0] <= my <= my_range[1]):
                # Create a specific key for this mass point
                # e.g., "MX-700_MY-400"
                key_name = f"MX-{mx}_MY-{my}"

                # Signal datasets are unique by mass, so usually 1 recid per key
                signal_tasks.append((key_name, [r['recid']]))

        print(f"   Identified {len(signal_tasks)} unique signal mass points.")

    # --- 2. Prepare Background Tasks ---
    if 'background' in config:
        print("-> Searching Backgrounds...")
        for bkg_name, bkg_conf in config['background'].items():
            records = get_cod_records(bkg_conf['wildcard'])
            # For background, we group ALL matching records under the YAML key (e.g. 'ttbar')
            all_recids = [r['recid'] for r in records]
            if all_recids:
                background_tasks.append((bkg_name, all_recids))
        print(f"   Identified {len(background_tasks)} background processes.")

    # --- 3. Execute All Resolutions in Parallel ---
    print("\n-> Resolving File URLs...")

    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # Submit Signal Jobs
        future_to_sig = {
            executor.submit(worker_resolve_named_dataset, name, rids): name
            for name, rids in signal_tasks
        }

        # Submit Background Jobs
        future_to_bkg = {
            executor.submit(worker_resolve_named_dataset, name, rids): name
            for name, rids in background_tasks
        }

        # Process Signal Results
        for future in tqdm(concurrent.futures.as_completed(future_to_sig), total=len(signal_tasks), desc="Signals"):
            name, urls = future.result()
            if urls:
                final_output["signal"][name] = urls

        # Process Background Results
        for future in tqdm(concurrent.futures.as_completed(future_to_bkg), total=len(background_tasks),
                           desc="Backgrounds"):
            name, urls = future.result()
            if urls:
                final_output["background"][name] = urls

    # --- 4. Save Output ---
    print(f"\nWriting results to {output_json}...")
    with open(output_json, 'w') as f:
        json.dump(final_output, f, indent=2)

    # Summary
    print("\n--- Summary ---")
    print(f"Signal Groups: {len(final_output['signal'])}")
    print(f"Background Groups: {len(final_output['background'])}")


if __name__ == "__main__":
    main()