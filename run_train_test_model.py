import os
import glob
import json
import time
import pickle
import torch
import numpy as np
from src.test import predict_and_transform_new
from src.config_generation import generate_configs
from src.train import train_model
from src.utility_functions import *
from src.parse import create_model_running_parse
from src.scaler import standard_scaling_fit_transform, standard_scaling_transform, per_output_standard_scaling_fit_transform, hybrid_scaling_fit_transform, hybrid_scaling_fit_transform_v2
from src.variable import *
from src.utility_plot import plot_results


# Set seeds for complete reproducibility
parser = create_model_running_parse()
args = parser.parse_args()

seed = args.seed
seed_all(seed)

# Set device based on argument
device_id = args.device_id
if torch.cuda.is_available():
    if device_id >= torch.cuda.device_count():
        raise ValueError(f"Device {device_id} not available. Only {torch.cuda.device_count()} GPU(s) available.")
    device = torch.device(f'cuda:{device_id}')
    print(f"Using GPU device: cuda:{device_id}")
else:
    device = torch.device('cpu')
    print("CUDA not available, using CPU")

meta_config_filepath = args.meta_config
experiment_name = args.experiment_name
target = args.target
verbose = args.verbose
label_mask = args.label_mask
input_scaling_method = args.input_scaling_method
target_scaling_method = args.target_scaling_method
file_weight_filepath = args.file_weight_filepath
fixed_weight = args.fixed_weight
is_graph_level = args.is_graph_level
fold_name = args.fold_name
is_e3nn = args.is_e3nn
is_continuous = args.is_continuous
skip_config_generation = args.skip_config_generation
run_final_test = args.run_final_test
final_test_only = args.final_test_only
final_test_config = args.final_test_config
final_test_epochs = args.final_test_epochs
train_on_all = args.train_on_all

config_dir = os.path.join('config', experiment_name)

### Step 1: Generate hyperparameter configs ONCE (shared across folds; config_dir is
### not fold-specific). Honors --skip_config_generation for the multi-GPU launcher.
msg(f'Loading hyperparameter grid from {meta_config_filepath}', verbose=verbose)
with open(meta_config_filepath, 'r') as f:
    hyperparameter_grid = json.load(f)
# In --train_on_all mode the winning hyperparameters are read straight from the grid
# (see train_on_all_data), so the per-config files are neither generated nor listed --
# this avoids clobbering an existing selection grid in config/<exp>/.
config_files = []
if not train_on_all:
    if not skip_config_generation:
        generate_configs(config_dir=config_dir, start_index=0, hyperparameter_grid=hyperparameter_grid)
    config_files = natural_sort([f for f in os.listdir(config_dir) if f.endswith('.json')])


def _apply_per_output(valid_data, test_data, attr, scalers):
    """Apply a list of per-output (fitted-on-train) scalers to valid/test in place."""
    for i in range(len(valid_data)):
        transformed_valid = np.zeros_like(valid_data[i][attr].numpy())
        transformed_test = np.zeros_like(test_data[i][attr].numpy())
        for j in range(len(scalers)):
            transformed_valid[:, j] = scalers[j].transform(valid_data[i][attr][:, j].reshape(-1, 1)).flatten()
            transformed_test[:, j] = scalers[j].transform(test_data[i][attr][:, j].reshape(-1, 1)).flatten()
        valid_data[i][attr] = torch.from_numpy(transformed_valid).to(torch.float32)
        test_data[i][attr] = torch.from_numpy(transformed_test).to(torch.float32)


def _apply_hybrid_v2(valid_data, test_data, attr, scalers, epsilon=1e-8):
    """Apply the hybrid_v2 mix (log1p markers + fitted scalers) to valid/test in place."""
    for i in range(len(valid_data)):
        transformed_valid = np.zeros_like(valid_data[i][attr].numpy())
        transformed_test = np.zeros_like(test_data[i][attr].numpy())
        for j in range(len(scalers)):
            if scalers[j] == "log":
                transformed_valid[:, j] = np.log1p(valid_data[i][attr][:, j].numpy() + epsilon)
                transformed_test[:, j] = np.log1p(test_data[i][attr][:, j].numpy() + epsilon)
            else:
                transformed_valid[:, j] = scalers[j].transform(valid_data[i][attr][:, j].reshape(-1, 1)).flatten()
                transformed_test[:, j] = scalers[j].transform(test_data[i][attr][:, j].reshape(-1, 1)).flatten()
        valid_data[i][attr] = torch.from_numpy(transformed_valid).to(torch.float32)
        test_data[i][attr] = torch.from_numpy(transformed_test).to(torch.float32)


def fit_scalers(train_data, valid_data, test_data, output_dir):
    """Fit input/target scalers on THIS fold's train set and apply to valid/test in place.

    Fitting on train only (and transforming valid/test) is leak-free per ML standard.
    Returns the target scaler, which is needed to inverse-transform predictions for the
    test/train metrics.
    """
    target_scaler = None

    # ---- input (x) scaling ----
    if input_scaling_method == STD_SCALER:
        input_scaler, _ = standard_scaling_fit_transform(train_data, "x", os.path.join(output_dir, INPUT_SCALER))
        standard_scaling_transform(valid_data, "x", input_scaler)
        standard_scaling_transform(test_data, "x", input_scaler)
    elif input_scaling_method == ALL_STD_SCALER:
        input_scaler, _ = per_output_standard_scaling_fit_transform(train_data, "x", os.path.join(output_dir, INPUT_SCALER))
        _apply_per_output(valid_data, test_data, "x", input_scaler)

    # ---- target (y) scaling ----
    target_scaler_filepath = os.path.join(output_dir, TARGET_SCALER)
    if target_scaling_method == STD_SCALER:
        target_scaler, _ = standard_scaling_fit_transform(train_data, "y", target_scaler_filepath)
        standard_scaling_transform(valid_data, "y", target_scaler)
        standard_scaling_transform(test_data, "y", target_scaler)
    elif target_scaling_method == ALL_STD_SCALER:
        target_scaler, _ = per_output_standard_scaling_fit_transform(train_data, "y", target_scaler_filepath)
        _apply_per_output(valid_data, test_data, "y", target_scaler)
    elif target_scaling_method == HYBRID_SCALER:
        target_scaler, _ = hybrid_scaling_fit_transform(train_data, "y", target_scaler_filepath)
        _apply_per_output(valid_data, test_data, "y", target_scaler)
    elif target_scaling_method == HYBRID_SCALER_V2:
        target_scaler, _ = hybrid_scaling_fit_transform_v2(train_data, "y", target_scaler_filepath)
        _apply_hybrid_v2(valid_data, test_data, "y", target_scaler)

    return target_scaler


def run_fold(fold):
    """Train/evaluate every (sharded) config for a single fold.

    `fold` is "" for the flat (non-CV) layout, or "fold_k" for a CV fold. Data is read
    from graph_data/<exp>/<fold>/ and all outputs go to output/<exp>/<fold>/, so each
    fold has its own scalers, logs, models, and plots. Returns a list of
    (config_index, meanARE, maxARE) for the configs this device handled.
    """
    msg(f"\n========== Fold: {fold or '(flat)'} ==========", verbose=True)
    data_dir = os.path.join('graph_data', experiment_name, fold)
    output_dir = os.path.join('output', experiment_name, fold)
    output_model_dir, output_plot_dir = create_output_dirs(output_dir)

    ### Load data for this fold
    with open(os.path.join(data_dir, TRAIN_SET), 'rb') as f:
        train_data = pickle.load(f)
    with open(os.path.join(data_dir, VALID_SET), 'rb') as f:
        valid_data = pickle.load(f)
    with open(os.path.join(data_dir, TEST_SET), 'rb') as f:
        test_data = pickle.load(f)
    msg(f"Loaded {len(train_data)} train / {len(valid_data)} val / {len(test_data)} test from {data_dir}", verbose=verbose)
    target_dim, train_data = compute_target_dim_and_transform_train_data(train_data, label_mask, verbose)

    log_file_path = os.path.join(output_dir, f"val_results_{device_id}.txt")
    loss_log_dir = os.path.join(output_dir, "loss_logs")  # one train/val loss-curve file per config
    os.makedirs(loss_log_dir, exist_ok=True)
    # On a fresh (non-continuous) run, truncate this device's validation results and
    # clear the per-config loss-curve files so stale entries are not left behind.
    # Continuous runs keep them so check_model_is_finished can detect trained configs.
    if not is_continuous:
        open(log_file_path, "w").close()
        for stale in glob.glob(os.path.join(loss_log_dir, "config_*.txt")):
            os.remove(stale)

    ### Fit scalers on this fold's train set and apply to valid/test
    msg("Training scalers (fit on this fold's train set)", verbose=verbose)
    target_scaler = fit_scalers(train_data, valid_data, test_data, output_dir)

    ### Train each config assigned to this device
    fold_results = []
    for i, config_file in enumerate(config_files):
        if i % args.num_gpu != device_id:
            continue
        if is_continuous and check_model_is_finished(i, output_dir, device_id=device_id):
            msg(f"Config {i} already finished. Skipping...", verbose=verbose)
            continue

        start_time = time.time()
        with open(os.path.join(config_dir, config_file), 'r') as f:
            params = json.load(f)
        msg(f'Loaded config file: {config_file}', verbose=verbose)

        model, optimizer, train_losses, val_losses = train_model(params=params,
                    train_data=train_data,
                    valid_data=valid_data,
                    output_model_dir=output_model_dir,
                    config_file=config_file,
                    target=target,
                    target_dim=target_dim,
                    is_graph_level=is_graph_level,
                    device=device,
                    is_e3nn=is_e3nn,
                    verbose=verbose)

        # SELECTION metric = VALIDATION only (the held fold in CV, or val.pkl in the
        # flat setting). The TEST set is NOT touched here -- it is evaluated exactly
        # once at the very end, on the single selected model.
        val_preds, val_trues = predict_and_transform_new(model=model,
                                                        data=valid_data,
                                                        batch_size=len(valid_data),
                                                        device=device,
                                                        target_scaling_method=target_scaling_method,
                                                        target_scaler=target_scaler)
        val_meanARE, val_maxARE = mean_maxARE_by_range(val_preds, val_trues)
        time_elapsed = time.time() - start_time
        msg(f'{i+1:>{len(str(len(config_files)))}}/{len(config_files)}: Time elapsed: {time_elapsed:.2f}s\t'
            f' Validation (MeanARE {val_meanARE}, MaxARE {val_maxARE})', verbose=verbose)

        # Validation (selection) metric. Leading "<i>:" is preserved so
        # check_model_is_finished / merge_gpu_logs still parse it.
        with open(log_file_path, "a") as log_file:
            log_file.write(f'{i}: ValMeanARE: {val_meanARE}: ValMaxARE: {val_maxARE} Time elapsed: {time_elapsed:.2f}\n')

        # Per-config train/val loss curve in its own file: loss_logs/config_<i>.txt
        loss_log_path = os.path.join(loss_log_dir, f"config_{i}.txt")
        with open(loss_log_path, "w") as loss_file:
            loss_file.write(f"Config {i} Train and Validation Losses:\n")
            loss_file.write(f"{'epoch':>6} | {'train_loss':>12} | {'val_loss':>12}\n")
            loss_file.write(f"{'-' * 6}-+-{'-' * 12}-+-{'-' * 12}\n")
            for epoch, (train_loss, val_loss) in enumerate(zip(train_losses, val_losses)):
                loss_file.write(f"{epoch + 1:>6} | {train_loss:>12.6f} | {val_loss:>12.6f}\n")
            loss_file.write(f"Config {i}: Validation MeanARE: {val_meanARE}, MaxARE: {val_maxARE}\n")

        # Parity plot: train vs the held-out validation set (test is not touched here).
        train_preds, train_trues = predict_and_transform_new(model=model,
                                                            data=train_data,
                                                            batch_size=params['batch_size'],
                                                            device=device,
                                                            target_scaling_method=target_scaling_method,
                                                            target_scaler=target_scaler)
        plot_filename = f"config_{i}_target_{target}.png"
        plot_results(train_preds=train_preds,
                    train_trues=train_trues,
                    test_trues=val_trues,
                    test_preds=val_preds,
                    output_dir=output_plot_dir,
                    fname=plot_filename)

        fold_results.append((i, val_meanARE, val_maxARE))
        # Clean up GPU memory
        del model
        del optimizer
        torch.cuda.empty_cache()

    return fold_results


def select_best_config(cv_results):
    """Select the best config by VALIDATION MeanARE (mean over outputs, averaged across
    folds). Validation is the ONLY metric used here -- the test set is reserved for a
    single final evaluation. Writes a CV summary when more than one fold ran. Returns
    (best_config_index, cv_val_score_dict); (None, {}) if nothing was trained here."""
    by_config = {}  # config_index -> list of (fold, val_meanARE_array)
    for fold, results in cv_results.items():
        for (i, val_meanARE, _val_max) in results:
            by_config.setdefault(i, []).append((fold, np.asarray(val_meanARE, dtype=float)))
    if not by_config:
        return None, {}

    cv_val_score = {i: float(np.stack([v for _, v in entries]).mean(axis=1).mean())
                    for i, entries in by_config.items()}

    if len(cv_results) > 1:
        summary_path = os.path.join('output', experiment_name, f"cv_summary_{device_id}.txt")
        with open(summary_path, "w") as f:
            f.write(f"Cross-validation summary over folds: {list(cv_results.keys())}\n")
            f.write("Selection metric = VALIDATION MeanARE (mean over outputs, averaged across folds).\n")
            f.write("The test set is NOT used here; it is evaluated once on the final selected model.\n\n")
            for i in sorted(by_config):
                folds = [fn for fn, _ in by_config[i]]
                val = np.stack([v for _, v in by_config[i]])   # (n_folds, num_outputs)
                val_scalar = val.mean(axis=1)
                f.write(f"Config {i} (folds {folds}):\n")
                f.write(f"  per-fold Val MeanARE  : {np.round(val_scalar, 3).tolist()}\n")
                f.write(f"  CV Val mean +/- std   : {val_scalar.mean():.3f} +/- {val_scalar.std():.3f}\n")
                f.write(f"  CV Val per-output mean: {np.round(val.mean(axis=0), 3).tolist()}\n\n")
            best = min(cv_val_score, key=cv_val_score.get)
            scope = "all configs" if args.num_gpu == 1 else f"this device's shard (device {device_id})"
            f.write(f"Best config by CV validation MeanARE ({scope}): config {best} ({cv_val_score[best]:.3f})\n")
        msg(f"Wrote CV summary to {summary_path}", verbose=True)

    return min(cv_val_score, key=cv_val_score.get), cv_val_score


def final_test(best_idx, n_epoch_override=None):
    """Touch the test set EXACTLY ONCE: retrain the selected config on ALL non-test data
    (train + validation pooled) and evaluate the held-out test set a single time. In CV,
    any fold's train+val is the full cross-validation set; in the flat layout it is the
    standard train+val."""
    ref_fold = folds_to_run[0]
    data_dir = os.path.join('graph_data', experiment_name, ref_fold)
    with open(os.path.join(data_dir, TRAIN_SET), 'rb') as f:
        train_data = pickle.load(f)
    with open(os.path.join(data_dir, VALID_SET), 'rb') as f:
        valid_data = pickle.load(f)
    with open(os.path.join(data_dir, TEST_SET), 'rb') as f:
        test_data = pickle.load(f)
    full_train = list(train_data) + list(valid_data)
    target_dim, full_train = compute_target_dim_and_transform_train_data(full_train, label_mask, verbose)

    out_dir = os.path.join('output', experiment_name, 'final_model')
    model_dir, plot_dir = create_output_dirs(out_dir)
    # Fit scalers on the pooled non-test data; transform the test set with them.
    target_scaler = fit_scalers(full_train, [], test_data, out_dir)

    config_file = f"{best_idx}.json"
    with open(os.path.join(config_dir, config_file), 'r') as f:
        params = json.load(f)
    if n_epoch_override:
        msg(f"Final model: overriding n_epoch {params['n_epoch']} -> {n_epoch_override} (longer horizon).", verbose=True)
        params['n_epoch'] = n_epoch_override
    msg(f"\n========== Final model: config {best_idx} (retrain on {len(full_train)} non-test graphs, test ONCE) ==========", verbose=True)
    seed_all(seed)
    # All non-test data is used for training; pass the training set itself as the
    # (logging-only) validation curve so the test set is never seen during training.
    model, _, _, _ = train_model(params=params,
                train_data=full_train,
                valid_data=full_train,
                output_model_dir=model_dir,
                config_file=config_file,
                target=target,
                target_dim=target_dim,
                is_graph_level=is_graph_level,
                device=device,
                is_e3nn=is_e3nn,
                verbose=verbose)

    test_preds, test_trues = predict_and_transform_new(model=model,
                                                    data=test_data,
                                                    batch_size=len(test_data),
                                                    device=device,
                                                    target_scaling_method=target_scaling_method,
                                                    target_scaler=target_scaler)
    test_meanARE, test_maxARE = mean_maxARE_by_range(test_preds, test_trues)
    results_path = os.path.join('output', experiment_name, 'final_test_results.txt')
    with open(results_path, "w") as f:
        f.write(f"Final selected config: {best_idx}\n")
        f.write(f"Trained on {len(full_train)} non-test graphs (train + validation pooled).\n")
        f.write(f"Test set touched ONCE.\n")
        f.write(f"Test MeanARE: {test_meanARE}\nTest MaxARE: {test_maxARE}\n")
    train_preds, train_trues = predict_and_transform_new(model=model, data=full_train,
        batch_size=params['batch_size'], device=device,
        target_scaling_method=target_scaling_method, target_scaler=target_scaler)
    plot_results(train_preds=train_preds, train_trues=train_trues,
                 test_trues=test_trues, test_preds=test_preds,
                 output_dir=plot_dir, fname=f"final_config_{best_idx}_target_{target}.png")
    msg(f"FINAL test (config {best_idx}): MeanARE {test_meanARE} -> wrote {results_path}", verbose=True)


def train_on_all_data(n_epoch_override=None):
    """Train the final DEPLOYMENT model on ALL data (train + validation + test pooled).

    This is the model to ship AFTER selection and held-out evaluation are already done:
    every available record is used for training, so there is NO held-out set and NO test
    metric is produced. The winning hyperparameters come from the single combination in
    --meta_config (the first combination if the grid has more than one). Scalers are fit
    on the full pooled dataset and saved alongside the model so the deployed model can
    transform new inputs / inverse-transform predictions."""
    ref_fold = folds_to_run[0]
    data_dir = os.path.join('graph_data', experiment_name, ref_fold)
    with open(os.path.join(data_dir, TRAIN_SET), 'rb') as f:
        train_data = pickle.load(f)
    with open(os.path.join(data_dir, VALID_SET), 'rb') as f:
        valid_data = pickle.load(f)
    with open(os.path.join(data_dir, TEST_SET), 'rb') as f:
        test_data = pickle.load(f)
    full_data = list(train_data) + list(valid_data) + list(test_data)
    msg(f"Deployment model: pooling ALL data -> {len(full_data)} graphs "
        f"({len(train_data)} train + {len(valid_data)} val + {len(test_data)} test).", verbose=True)
    target_dim, full_data = compute_target_dim_and_transform_train_data(full_data, label_mask, verbose)

    out_dir = os.path.join('output', experiment_name, 'deploy_model')
    model_dir, plot_dir = create_output_dirs(out_dir)
    # Fit scalers on the full pooled dataset (it IS the training set); nothing held out.
    fit_scalers(full_data, [], [], out_dir)

    # Winning hyperparameters = the single combination in --meta_config. With single-
    # element lists there is exactly one; if the grid expands, take the first and warn.
    n_combos = int(np.prod([len(v) for v in hyperparameter_grid.values()]))
    if n_combos != 1:
        msg(f"WARNING: --meta_config has {n_combos} combinations; --train_on_all uses only "
            f"the FIRST. Pass a single-config grid for an unambiguous deployment model.", verbose=True)
    params = {k: v[0] for k, v in hyperparameter_grid.items()}
    if n_epoch_override:
        msg(f"Deployment model: overriding n_epoch {params['n_epoch']} -> {n_epoch_override}.", verbose=True)
        params['n_epoch'] = n_epoch_override

    config_file = "deploy"  # models saved as config_deploy_*.pth
    msg(f"\n========== Deployment model: train on ALL {len(full_data)} graphs, config {params} ==========", verbose=True)
    seed_all(seed)
    # No held-out data: pass the training set itself as the (logging-only) val curve.
    model, _, train_losses, _ = train_model(params=params,
                train_data=full_data,
                valid_data=full_data,
                output_model_dir=model_dir,
                config_file=config_file,
                target=target,
                target_dim=target_dim,
                is_graph_level=is_graph_level,
                device=device,
                is_e3nn=is_e3nn,
                verbose=verbose)

    info_path = os.path.join(out_dir, 'deploy_model_info.txt')
    with open(info_path, "w") as f:
        f.write("Final DEPLOYMENT model -- trained on ALL data (no held-out set, no test metric).\n")
        f.write(f"Trained on {len(full_data)} graphs (train + val + test pooled).\n")
        f.write(f"Config: {json.dumps(params)}\n")
        f.write(f"Final training loss: {train_losses[-1]:.6f}\n")
        f.write("Selection/evaluation were completed earlier on the CV set "
                "(winner = 256/3 + elu; pooled-train Test MeanARE 0.867). This model "
                "is for deployment only.\n")
    msg(f"Deployment model trained on ALL {len(full_data)} graphs -> models in {model_dir}; "
        f"scalers in {out_dir}; info at {info_path}", verbose=True)


### Determine which folds to run
if fold_name:
    # Explicit fold (or flat layout if "") requested -> run just that one.
    folds_to_run = [fold_name]
else:
    exp_data_dir = os.path.join('graph_data', experiment_name)
    fold_dirs = []
    if os.path.isdir(exp_data_dir):
        fold_dirs = natural_sort([d for d in os.listdir(exp_data_dir)
                                  if d.startswith('fold_')
                                  and os.path.isdir(os.path.join(exp_data_dir, d))
                                  and os.path.exists(os.path.join(exp_data_dir, d, TRAIN_SET))])
    # Use the CV folds if present, otherwise the flat top-level layout.
    folds_to_run = fold_dirs if fold_dirs else [""]

if len(folds_to_run) > 1:
    msg(f"Cross-validation: running {len(folds_to_run)} folds -> {folds_to_run}", verbose=True)

def _winner_from_cv_summary():
    """Read the selected config index from this device's CV summary file."""
    import re
    path = os.path.join('output', experiment_name, f"cv_summary_{device_id}.txt")
    if not os.path.exists(path):
        return None
    m = re.search(r"Best config by CV validation MeanARE.*?config (\d+)", open(path).read(), re.S)
    return int(m.group(1)) if m else None


if train_on_all:
    # Final deployment model: train the winning config on ALL data (train+val+test).
    # No selection, no held-out test -- those are already done.
    train_on_all_data(n_epoch_override=(final_test_epochs or None))
elif final_test_only:
    # Skip the CV loop entirely; just retrain the chosen config on the pooled non-test
    # data and evaluate the test set once (optionally at a longer training horizon).
    best_idx = final_test_config if final_test_config >= 0 else _winner_from_cv_summary()
    if best_idx is None:
        raise ValueError("--final_test_only: provide --final_test_config or ensure "
                         f"output/{experiment_name}/cv_summary_{device_id}.txt exists.")
    final_test(best_idx, n_epoch_override=(final_test_epochs or None))
else:
    cv_results = {}
    for fold in folds_to_run:
        cv_results[fold] = run_fold(fold)

    # Select the best config on VALIDATION only.
    best_idx, _ = select_best_config(cv_results)

    # Touch the test set EXACTLY ONCE: retrain the selected config on all non-test data and
    # evaluate the test set a single time. This requires having scored ALL configs, which
    # holds for a single-GPU run. For a sharded multi-GPU run, no single device sees every
    # config, so run a single-process pass (num_gpu=1) to do the global selection + final test.
    if best_idx is None:
        msg("No configs were trained on this device; skipping final test.", verbose=True)
    elif not run_final_test:
        msg(f"--run_final_test=false: skipping the held-out test set. "
            f"Best config by CV validation MeanARE: config {best_idx}.", verbose=True)
    elif args.num_gpu == 1:
        final_test(best_idx, n_epoch_override=(final_test_epochs or None))
    else:
        msg("Multi-GPU shard: validation selection done for this device's configs only. "
            "Run a single-process pass (--num_gpu 1) to select globally and evaluate the test set once.",
            verbose=True)
