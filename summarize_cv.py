"""Summarize cross-validation results into a CSV and report the winner.

Reads ``output/<exp>/fold_*/val_results_<device_id>.txt`` (the per-config
VALIDATION MeanARE arrays) across all COMPLETE folds, computes each config's CV
validation score (mean over the 21 outputs, averaged across folds) and its std
across folds, joins the architecture/activation from ``config/<exp>/<idx>.json``,
writes a CSV sorted by CV score (best first), and prints the winner -- the config
with the lowest CV validation error.

Usage:
    python summarize_cv.py --experiment_name hg_large_model_cv
"""
import argparse
import csv
import glob
import json
import os
import re

import numpy as np


def _fold_scalars(path, n_configs):
    """Return {config_idx: mean-over-outputs ValMeanARE} for one fold's results file."""
    entries = {}
    for block in re.split(r'(?m)^(?=\d+:)', open(path).read()):
        block = block.strip()
        if not block:
            continue
        idx = int(block.split(':', 1)[0])
        m = re.search(r'ValMeanARE:\s*\[(.*?)\]', block, re.S)
        if m:
            entries[idx] = float(np.array([float(x) for x in m.group(1).split()]).mean())
    return entries


def main():
    p = argparse.ArgumentParser(description="Summarize CV results into a CSV and report the winner")
    p.add_argument('--experiment_name', required=True)
    p.add_argument('--device_id', type=int, default=0)
    p.add_argument('--out', default=None, help='CSV output path (default output/<exp>/cv_summary.csv)')
    p.add_argument('--require_complete', type=lambda x: x.lower() == 'true', default=True,
                   help='Only include folds that have scored every config (avoids mixing fold counts)')
    args = p.parse_args()

    exp = args.experiment_name
    cfg_dir = os.path.join('config', exp)
    n_configs = len([f for f in glob.glob(os.path.join(cfg_dir, '*.json'))])

    fold_files = sorted(glob.glob(os.path.join('output', exp, 'fold_*', f'val_results_{args.device_id}.txt')))
    per_config = {}   # idx -> {fold_name: scalar}
    folds_used = []
    for fp in fold_files:
        fold = os.path.basename(os.path.dirname(fp))
        entries = _fold_scalars(fp, n_configs)
        if args.require_complete and len(entries) < n_configs:
            continue
        folds_used.append(fold)
        for i, s in entries.items():
            per_config.setdefault(i, {})[fold] = s

    if not per_config:
        raise SystemExit("No complete folds found yet.")

    fold_cols = [f'fold_{f.split("_")[-1]}' for f in folds_used]
    rows = []
    for i in sorted(per_config):
        c = json.load(open(os.path.join(cfg_dir, f'{i}.json')))
        scalars = [per_config[i][f] for f in folds_used if f in per_config[i]]
        row = {
            'config': i,
            'hidden_size': c['hidden_size'],
            'num_layers': c['num_layers'],
            'activation': c['activation'],
            'layer_type': c['layer_type'],
            'batch_size': c['batch_size'],
            'n_epoch': c['n_epoch'],
            'cv_val_meanARE': round(float(np.mean(scalars)), 4),
            'cv_val_std': round(float(np.std(scalars)), 4),
            'n_folds': len(scalars),
        }
        for f in folds_used:
            row[f'fold_{f.split("_")[-1]}'] = round(per_config[i][f], 4) if f in per_config[i] else ''
        rows.append(row)

    rows.sort(key=lambda r: r['cv_val_meanARE'])
    winner = rows[0]

    out = args.out or os.path.join('output', exp, 'cv_summary.csv')
    fieldnames = (['rank', 'config', 'hidden_size', 'num_layers', 'activation', 'layer_type',
                   'batch_size', 'n_epoch', 'cv_val_meanARE', 'cv_val_std', 'n_folds']
                  + fold_cols + ['is_winner'])
    with open(out, 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        for rank, r in enumerate(rows, 1):
            r = {**r, 'rank': rank, 'is_winner': r['config'] == winner['config']}
            w.writerow(r)

    print(f"Wrote {out}  ({len(rows)} configs over folds {folds_used})")
    print(f"WINNER: config {winner['config']} = {winner['hidden_size']}/{winner['num_layers']} "
          f"{winner['activation']}  |  CV ValMeanARE {winner['cv_val_meanARE']} +/- {winner['cv_val_std']}")
    print(f"WINNER_CONFIG={winner['config']}")


if __name__ == '__main__':
    main()
