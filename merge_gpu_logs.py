"""Merge per-GPU validation-result logs into a single, config-ordered file.

``run_all_gpus.sh`` runs one worker per GPU; the configs are sharded across GPUs,
and each worker writes its own ``val_results_<device_id>.txt`` (the per-config
validation selection metric). After all workers finish, this script merges those
per-GPU files into ``val_results_merged.txt`` ordered by config index,
deduplicated by config index (keeping the last occurrence).

Per-config train/val loss curves are written to their own files
(``loss_logs/config_<i>.txt``) and the final test loss to ``final_test_results.txt``,
so neither needs merging across GPUs.

Usage:
    python merge_gpu_logs.py --experiment_name <name> [--fold_name <fold>]
"""
import argparse
import glob
import os
import re


def merge_val_results(output_dir):
    # A result entry starts with "<idx>: ..." but the MeanARE/MaxARE numpy arrays
    # wrap across several physical lines, so we accumulate each entry as a block
    # until the next "<idx>:" header (otherwise the continuation lines are lost).
    header_re = re.compile(r"^(\d+):")
    merged = {}  # config_idx -> full (possibly multi-line) entry text
    files = sorted(f for f in glob.glob(os.path.join(output_dir, "val_results_*.txt"))
                   if not f.endswith("val_results_merged.txt"))
    for path in files:
        cur_idx, cur_lines = None, []
        with open(path, "r") as fh:
            for raw in fh:
                line = raw.rstrip("\n")
                m = header_re.match(line)
                if m:
                    if cur_idx is not None:
                        merged[cur_idx] = "\n".join(cur_lines).rstrip()
                    cur_idx, cur_lines = int(m.group(1)), [line]
                elif cur_idx is not None:
                    cur_lines.append(line)
        if cur_idx is not None:
            merged[cur_idx] = "\n".join(cur_lines).rstrip()
    out_path = os.path.join(output_dir, "val_results_merged.txt")
    with open(out_path, "w") as fh:
        for idx in sorted(merged):
            fh.write(merged[idx] + "\n")
    return out_path, len(merged)


def main():
    parser = argparse.ArgumentParser(description="Merge per-GPU validation-result logs")
    parser.add_argument("--experiment_name", type=str, required=True)
    parser.add_argument("--fold_name", type=str, default="")
    args = parser.parse_args()

    output_dir = os.path.join("output", args.experiment_name, args.fold_name)
    if not os.path.isdir(output_dir):
        raise SystemExit(f"Output dir not found: {output_dir}")

    val_path, n_val = merge_val_results(output_dir)
    print(f"Merged {n_val} result line(s) -> {val_path}")


if __name__ == "__main__":
    main()
