"""Generate the activation-sweep config files for the CV experiment.

We compare a fixed SET of architectures -- explicit ``(hidden_size, num_layers)``
pairs -- crossed with a set of hidden activations, holding everything else fixed.
The standard meta-config grid takes the Cartesian product of ``hidden_size`` x
``num_layers``, which cannot express "only these specific pairs" (it would also
emit 128/3 and 256/2). So this script materializes the paired configs directly
via the repo's ``generate_configs`` -- one call per architecture, using
``start_index`` so the indices stay contiguous.

Only negative-capable / two-sided hidden activations are swept; the one-sided
non-negative-output ones (relu, softplus) are intentionally excluded because the
standardized (zero-mean, signed) targets need a hidden activation that can pass
the negative half. The output head stays linear in all cases.

Index layout (with the defaults below):
    0-6 : hidden_size=128, num_layers=2  x  [none, elu, leaky_relu, tanh, silu, gelu, shifted_softplus]
    7-13: hidden_size=256, num_layers=3  x  (same activations, same order)

Run the training with ``--skip_config_generation true`` so
``run_train_test_model.py`` uses exactly these files instead of regenerating
them from a single meta config.

Usage:
    python prepare_activation_configs.py --experiment_name hg_large_model_cv --n_epoch 1500
"""
import argparse
import os

from src.config_generation import generate_configs

# Explicit architectures to compare (hidden_size, num_layers): the best config
# from the meta_config_hg_large_model run (128/2) and the canonical baseline (256/3).
ARCH_PAIRS = [(128, 2), (256, 3)]

# Negative-capable / two-sided hidden activations only ("none" = linear baseline).
ACTIVATIONS = ["none", "elu", "leaky_relu", "tanh", "silu", "gelu", "shifted_softplus"]


def main():
    p = argparse.ArgumentParser(description="Generate paired (architecture x activation) configs")
    p.add_argument("--experiment_name", required=True)
    p.add_argument("--n_epoch", type=int, default=1500)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--l_rate", type=float, default=1e-4)
    p.add_argument("--w_decay", type=float, default=1e-4)
    p.add_argument("--layer_type", default="TransformerConv")
    p.add_argument("--use_amp", type=lambda x: x.lower() == "true", default=True)
    args = p.parse_args()

    config_dir = os.path.join("config", args.experiment_name)
    base = {
        "l_rate": [args.l_rate],
        "w_decay": [args.w_decay],
        "n_epoch": [args.n_epoch],
        "batch_size": [args.batch_size],
        "layer_type": [args.layer_type],
        "use_amp": [args.use_amp],
    }

    idx = 0
    for hidden_size, num_layers in ARCH_PAIRS:
        grid = {**base,
                "hidden_size": [hidden_size],
                "num_layers": [num_layers],
                "activation": ACTIVATIONS}
        # start_index 0 (first arch) clears stale configs first; later archs append.
        generate_configs(config_dir=config_dir, start_index=idx, hyperparameter_grid=grid)
        idx += len(ACTIVATIONS)

    print(f"Generated {idx} configs in {config_dir} "
          f"({len(ARCH_PAIRS)} archs x {len(ACTIVATIONS)} activations, "
          f"n_epoch={args.n_epoch}, batch_size={args.batch_size}, use_amp={args.use_amp})")


if __name__ == "__main__":
    main()
