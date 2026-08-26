import os
import numpy as np
from src.utility_functions import *
from src.parse import create_parser
from src.io_functions import *

# Argument parser setup
parser = create_parser()
args = parser.parse_args()

input_dir = os.path.expanduser(args.input_dir)
input_dir_prefix = args.input_dir_prefix
input_file_postfix = args.input_file_postfix
input_graph_postfix = args.input_graph_postfix
target_file_postfix = args.target_file_postfix
textures = args.textures
experiment_name = args.experiment_name

train_size = args.train_size
val_size = args.val_size
test_size = args.test_size

n_folds = args.n_folds # e.g., 5 for 5-fold CV
seed = args.seed

# file_weight_filepath = args.file_weight_filepath
# constant_weight = args.constant_weight
# standard_weight_dim = args.standard_weight_dim

output_dir = os.path.join("./graph_data", experiment_name)

# Step 1: Read input data
msg("Step 1: Reading input data...")
all_input_data = []
all_graph_filepaths = []
all_graph_data = []

for texture in textures:
    print("processing texture:", texture)
    texture_input_data = []
    texture_graph_data = []

    texture_input_dir = get_input_dir(input_dir, texture, input_dir_prefix, "input")
    texture_graph_filepaths = get_filepaths(texture_input_dir, input_graph_postfix)
    texture_graph_data = load_graph_data(texture_graph_filepaths)

    # Step 1: Read input data
    for i, graph_filename in enumerate(texture_graph_filepaths):
        input = None
        graph_data = texture_graph_data[i]
        if (i + 1) % 10 == 0:
            print(f"Processing input file {i+1}/{len(texture_graph_filepaths)}")
        for post_fix in input_file_postfix:
            input_filename = graph_filename.replace(input_graph_postfix, post_fix)
            input_tmp = read_file(input_filename, post_fix, graph_data, input_graph_postfix)
            input = np.concatenate((input, input_tmp), axis=1) if input is not None else input_tmp

        texture_input_data.append(input)
    all_input_data.append(texture_input_data)
    all_graph_data.append(texture_graph_data)
    all_graph_filepaths.append(texture_graph_filepaths)

num_experiments_per_texture = len(texture_graph_filepaths)

# Step 2: Read target data
msg("Step 2: Reading target data...")
all_target_data = []

for texture in textures:
    print("processing texture:", texture)
    texture_target_data = []

    target_dir = get_input_dir(input_dir, texture, input_dir_prefix, "target")
    target_filepaths = get_filepaths(target_dir, f"{target_file_postfix}")
    for i, target_filepath in enumerate(target_filepaths):
        target = read_target_file(target_filepath)
        if len(target.shape) <= 1:
            target = target.reshape(1, -1)
        if len(target.shape) > 2:
            target = target.reshape(target.shape[0], -1)
        texture_target_data.append(target)

    all_target_data.append(texture_target_data)

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Step 3: Split the data and save the base data
msg("Step 3: Splitting the data and saving the base data...")

if n_folds is None:
    msg("No cross-validation is performed. Saving train, test, and validation data...")
    train_indices = np.random.choice(num_experiments_per_texture, size=int(train_size * num_experiments_per_texture), replace=False)
    tmp_indices = np.setdiff1d(np.arange(num_experiments_per_texture), train_indices)
    test_indices = np.random.choice(tmp_indices, size=int(test_size * num_experiments_per_texture), replace=False)
    val_indices = np.setdiff1d(tmp_indices, test_indices)
    save_train_test_valid_data(textures, 
                           all_graph_data, 
                           all_input_data, 
                           all_target_data, 
                           train_indices, 
                           test_indices, 
                           val_indices,
                           output_dir)
else:
    msg(f"Cross-validation is performed with {n_folds} folds (seed={seed})...")
    # Seeded shuffle so the partition is reproducible.
    rng = np.random.default_rng(seed)
    shuffled_indices = rng.permutation(num_experiments_per_texture)

    # Split into a held-out test set and a cross-validation set.
    test_count = int(test_size * num_experiments_per_texture)
    test_indices = shuffled_indices[:test_count]
    cv_indices = shuffled_indices[test_count:]
    if n_folds > len(cv_indices):
        raise ValueError(f"n_folds ({n_folds}) exceeds the cross-validation set size ({len(cv_indices)})")

    # Split the (already shuffled) CV set into k folds. Each fold is the validation
    # set once and the other folds are training; the held-out test set is shared
    # across all folds and written into every fold directory so each fold is
    # self-contained for the training pipeline.
    folds = np.array_split(cv_indices, n_folds)
    for fold in range(n_folds):
        fold_dir = f"fold_{fold}"
        os.makedirs(os.path.join(output_dir, fold_dir), exist_ok=True)
        val_indices = folds[fold]
        train_indices = np.concatenate([folds[i] for i in range(n_folds) if i != fold])
        msg(f"  fold {fold}: train={len(train_indices)} val={len(val_indices)} test={len(test_indices)}")
        save_train_test_valid_data(textures, all_graph_data, all_input_data, all_target_data,
                                   train_indices, test_indices, val_indices, output_dir, fold_dir=fold_dir)
