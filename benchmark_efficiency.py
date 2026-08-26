"""Benchmark: does AMP (bf16) + larger batch preserve accuracy vs the baseline?

Trains the SAME config (active config 0: TransformerConv 256/3, activation none,
standard scaling) under each setting with the same seed, on the hg_large_model
data, and reports test MeanARE (range-normalized) + final val loss + wall-clock
time. The 'baseline' (batch 8, fp32) reproduces the existing result; 'amp_b32'
(batch 32, bf16 AMP) is the efficiency candidate.

Usage:
  python benchmark_efficiency.py --n_epoch 500 --seed 42 --device_id 0
"""
import argparse
import os
import pickle
import tempfile
import time

import numpy as np
import torch

from src.scaler import standard_scaling_fit_transform, standard_scaling_transform
from src.test import predict_and_transform_new
from src.train import train_model
from src.utility_functions import seed_all, mean_maxARE_by_range
from src.variable import TRAIN_SET, VALID_SET, TEST_SET, INPUT_SCALER, TARGET_SCALER


def load_scaled(data_dir, tmp_out):
    with open(os.path.join(data_dir, TRAIN_SET), "rb") as f:
        train = pickle.load(f)
    with open(os.path.join(data_dir, VALID_SET), "rb") as f:
        valid = pickle.load(f)
    with open(os.path.join(data_dir, TEST_SET), "rb") as f:
        test = pickle.load(f)
    in_sc, train = standard_scaling_fit_transform(train, "x", os.path.join(tmp_out, INPUT_SCALER))
    valid = standard_scaling_transform(valid, "x", in_sc)
    test = standard_scaling_transform(test, "x", in_sc)
    tg_sc, train = standard_scaling_fit_transform(train, "y", os.path.join(tmp_out, TARGET_SCALER))
    valid = standard_scaling_transform(valid, "y", tg_sc)
    test = standard_scaling_transform(test, "y", tg_sc)
    return train, valid, test, tg_sc


def run_one(params, train, valid, test, tg_sc, seed, device, tmp_models):
    seed_all(seed)
    t0 = time.time()
    model, _, _, val_losses = train_model(
        params=params, train_data=train, valid_data=valid,
        output_model_dir=tmp_models, config_file="bench", target="y",
        target_dim=int(train[0].y.shape[1]), is_graph_level=True,
        device=device, is_e3nn=False, verbose=False)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed = time.time() - t0
    preds, trues = predict_and_transform_new(
        model=model, data=test, batch_size=len(test), device=device,
        target_scaling_method="standard", target_scaler=tg_sc)
    mean_are, _ = mean_maxARE_by_range(preds, trues)
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return np.asarray(mean_are, dtype=float), float(val_losses[-1]), elapsed


def main():
    ap = argparse.ArgumentParser(description="Benchmark AMP + batch size vs baseline")
    ap.add_argument("--experiment_name", default="hg_large_model")
    ap.add_argument("--n_epoch", type=int, default=500)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device_id", type=int, default=0)
    ap.add_argument("--skip_baseline", type=lambda x: x.lower() == "true", default=True,
                    help="Reuse the stored baseline (b8/fp32/lr1e-4, 500ep, seed42) instead of re-running it")
    args = ap.parse_args()

    # Stored baseline = the existing result from the identical config 0 run
    # (TransformerConv 256/3, activation none, batch 8, fp32, lr 1e-4, 500 epochs,
    # seed 42, standard scaling) captured in the edge_weight 'OFF' benchmark.
    STORED_BASELINE = dict(
        mean_are=np.array([0.31, 0.45, 0.31, 0.45, 0.50, 0.35, 2.10, 1.98, 2.04, 0.59, 2.01,
                           2.93, 2.69, 0.46, 0.60, 0.85, 0.88, 0.40, 2.74, 2.51, 0.53]),
        final_val=0.0084,
    )

    device = torch.device(f"cuda:{args.device_id}" if torch.cuda.is_available() else "cpu")
    data_dir = os.path.join("graph_data", args.experiment_name)
    base = dict(w_decay=1e-4, n_epoch=args.n_epoch,
                hidden_size=256, num_layers=3, layer_type="TransformerConv", activation="none")

    # baseline = the existing config (batch 8, fp32, lr 1e-4) -- skipped by default.
    # amp_b32_samelr = literal "AMP + batch 32" at the same lr (does 4x fewer updates/epoch).
    # amp_b32_lr4x   = AMP + batch 32 with the linear LR-scaling rule (lr x4) -> fair accuracy.
    arms = [
        ("amp_b32_samelr", dict(batch_size=32, use_amp=True, l_rate=1e-4)),
        ("amp_b32_lr4x",   dict(batch_size=32, use_amp=True, l_rate=4e-4)),
    ]
    if not args.skip_baseline:
        arms = [("baseline_b8_fp32", dict(batch_size=8, use_amp=False, l_rate=1e-4))] + arms

    results = {}
    order = []
    for name, over in arms:
        tmp = tempfile.mkdtemp(prefix="bench_eff_")
        train, valid, test, tg = load_scaled(data_dir, tmp)
        params = dict(base, **over)
        mean_are, final_val, elapsed = run_one(params, train, valid, test, tg, args.seed, device, tmp)
        results[name] = (mean_are, final_val, elapsed, over)
        order.append(name)
        print(f"{name:18} | batch={over['batch_size']:3} lr={over['l_rate']:.0e} amp={str(over['use_amp']):5} | "
              f"MeanARE={mean_are.mean():.3f} | val={final_val:.4f} | time={elapsed:.1f}s", flush=True)

    if args.skip_baseline:
        b_are, b_val, b_t = STORED_BASELINE["mean_are"], STORED_BASELINE["final_val"], None
        base_label = "baseline_b8_fp32 (stored)"
    else:
        b_are, b_val, b_t, _ = results["baseline_b8_fp32"]
        base_label = "baseline_b8_fp32 (run)"

    print("\n==================== AMP + BATCH-32 vs BASELINE ====================")
    print(f"config: TransformerConv 256/3 activation=none n_epoch={args.n_epoch} seed={args.seed}")
    print(f"{'setting':26} {'MeanARE':>9} {'Δvs base':>9} {'val':>8} {'time(s)':>9}")
    print(f"{base_label:26} {b_are.mean():9.3f} {0.0:+9.3f} {b_val:8.4f} {('n/a' if b_t is None else f'{b_t:.1f}'):>9}")
    for name in order:
        are, val, t, _ = results[name]
        print(f"{name:26} {are.mean():9.3f} {are.mean()-b_are.mean():+9.3f} {val:8.4f} {t:9.1f}")
    print("\nper-output MeanARE (baseline -> amp_b32_lr4x):")
    lr4_are = results["amp_b32_lr4x"][0]
    for i in range(len(b_are)):
        print(f"  y[{i:2}]: {b_are[i]:6.2f} -> {lr4_are[i]:6.2f}  ({lr4_are[i]-b_are[i]:+.2f})")


if __name__ == "__main__":
    main()
