"""Generate the hyperparameter-grid config files for an experiment ONCE.

Used by ``run_all_gpus.sh`` before fanning out to per-GPU workers, so that all
workers read an identical, already-materialized config set. Doing it here (a
single process) avoids a race where each worker would otherwise clear and
rewrite the config directory concurrently.

Usage:
    python prepare_configs.py --meta_config <grid.json> --experiment_name <name>
"""
import argparse
import json
import os

from src.config_generation import generate_configs


def main():
    parser = argparse.ArgumentParser(description="Generate hyperparameter config files for an experiment")
    parser.add_argument("--meta_config", type=str, required=True, help="Path to meta config (hyperparameter grid) JSON")
    parser.add_argument("--experiment_name", type=str, required=True, help="Experiment name")
    args = parser.parse_args()

    config_dir = os.path.join("config", args.experiment_name)
    with open(args.meta_config, "r") as f:
        hyperparameter_grid = json.load(f)

    # start_index=0 makes generate_configs clear any stale configs first.
    generate_configs(config_dir=config_dir, start_index=0, hyperparameter_grid=hyperparameter_grid)
    n = len([f for f in os.listdir(config_dir) if f.endswith(".json")])
    print(f"Generated {n} config(s) in {config_dir}")


if __name__ == "__main__":
    main()
