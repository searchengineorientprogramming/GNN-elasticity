import torch
import numpy as np
import random
import os
import re
import glob
import pickle


def seed_all(seed):
    '''
    Set random seeds for reproducability
    '''
    if not seed:
        seed = 42
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    msg(f'Set random seeds to {seed}')

def mean_maxARE_by_range(y_pred, y_true, axis=0, epsilon=1e-8):
    '''Mean and Max Absolute Relative Error, normalized per output by the TRUE
    range (max - min) of the targets.

    The actual minimum is used even when it is negative; previously the minimum
    was clipped up to 0, which shrank the denominator and inflated the reported
    error for every output that takes negative values. A small epsilon guards
    against a zero range for constant outputs.
    '''
    min_val = np.min(y_true, axis=axis, keepdims=True)
    max_val = np.max(y_true, axis=axis, keepdims=True)
    range_val = (max_val - min_val) + epsilon
    meanARE = np.around(100*np.mean(np.abs((y_pred-y_true)/range_val), axis=axis), decimals=2)
    maxARE = np.around(100*np.max(np.abs((y_pred-y_true)/range_val), axis=axis), decimals=2)
    return meanARE, maxARE

def natural_sort(l):
    def convert(text): return int(text) if text.isdigit() else text.lower()
    def alphanum_key(key): return [convert(c)
                                   for c in re.split('([0-9]+)', key)]
    return sorted(l, key=alphanum_key)

def create_output_dirs(output_dir):
    output_model_dir = os.path.join(output_dir, 'models')
    if not os.path.exists(output_model_dir):
        os.makedirs(output_model_dir, exist_ok=True)
    output_plot_dir = os.path.join(output_dir, 'plots')
    if not os.path.exists(output_plot_dir):
        os.makedirs(output_plot_dir, exist_ok=True)
    
    return output_model_dir, output_plot_dir

def msg(msg, verbose=True):
    if verbose:
        print(msg)

def check_feature_dimension(data):
    top = data[0]["x"].shape[-1]
    for i in range(len(data)):
        if top != data[i]["x"].shape[-1]:
            raise ValueError("All samples do not have the same feature dimension.")
    return top

def save_scaler(scaler, filepath):
    with open(filepath, 'wb') as f:
        pickle.dump(scaler, f, protocol=pickle.HIGHEST_PROTOCOL)
    msg(f"Saved scaler to {filepath}")

def numerical_label_mask(label_mask):
    return [int(l) for l in label_mask]

def compute_target_dim_and_transform_train_data(train_data, label_mask, verbose):
    if label_mask != []:
        label_mask = numerical_label_mask(label_mask)
        msg(f"Label mask applied: {label_mask}", verbose=verbose)
        target_dim = len(label_mask)
        for i, record in enumerate(train_data):
            record.y = record.y[:, label_mask]
    else:
        target_dim = int(train_data[0].y.shape[1])
        msg(f"No label mask provided, using default mask with all y elements, target dimension is set to {target_dim}", verbose=verbose)
    return target_dim, train_data

def check_model_is_finished(config_idx, output_dir, device_id=None):
    """Return True if ``config_idx`` already has a recorded result.

    Scans every per-GPU ``val_results_*.txt`` under ``output_dir`` (not only this
    device's file) so that resume (``--is_continuous``) keeps working even when the
    GPU count / sharding changes between runs. Each result line is written as
    ``"<config_idx>: MeanARE: ..."``, so we compare the leading integer EXACTLY
    rather than by substring -- this avoids config 1 being mistaken as finished
    because config 11's line contains "1:". Missing files are treated as "not
    finished" (e.g. the very first continuous run before any log exists).
    """
    for log_file_path in glob.glob(os.path.join(output_dir, "val_results_*.txt")):
        try:
            with open(log_file_path, "r") as log_file:
                for line in log_file:
                    head = line.split(":", 1)[0].strip()
                    if head.isdigit() and int(head) == config_idx:
                        return True
        except FileNotFoundError:
            continue
    return False