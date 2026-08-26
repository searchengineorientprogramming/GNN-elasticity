import argparse

def create_parser():
    parser = argparse.ArgumentParser(description='Generate training data for graph neural networks.')
    parser.add_argument('--input_dir', type=str, default=None, help='Directory containing input files')
    parser.add_argument('--input_dir_prefix', type=str, default=None, help='Prefix for input directories')
    parser.add_argument('--input_file_postfix', nargs='+', default=["_out.npy"], help='Postfix for input files')
    parser.add_argument('--input_graph_postfix', type=str, default=".S", help='Postfix for input graph files')
    parser.add_argument('--target_file_postfix', type=str, default="_sigma_out.npy", help='Postfix for target files')
    parser.add_argument('--textures', nargs='+', default=[], help='Textures to process')

    parser.add_argument('--experiment_name', type=str, default='default_experiment', help='Experiment name')

    parser.add_argument('--train_size', type=float, default=0.8, help='Proportion of data to use for training')
    parser.add_argument('--val_size', type=float, default=0.1, help='Proportion of data to use for validation')
    parser.add_argument('--test_size', type=float, default=0.1, help='Proportion of data to use for testing')

    parser.add_argument('--n_folds', type=int, default=None, help='Number of folds for cross-validation')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducible data splitting')
    parser.add_argument('--num_gpu', type=int, default=1, help='Number of GPUs to use')
    # parser.add_argument('--file_weight_filepath', type=str, default=None, help='Filename for the file weight')
    # parser.add_argument('--constant_weight', type=float, default=None, help='Scalar weight to apply to input file weights')
    # parser.add_argument('--standard_weight_dim', nargs='+', default=None, type=int, help='Dimension of the standard weight')

    return parser

def create_model_running_parse():
    parser = argparse.ArgumentParser(description='Train GNN model')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--meta_config', type=str, default=None, help='Path to meta config file')
    parser.add_argument('--experiment_name', type=str, default=None, help='Experiment name')
    parser.add_argument('--target', type=str, default='y', help='Property to predict')
    parser.add_argument('--verbose', type=lambda x: x.lower() == 'true', default=False, help='Enable verbose output')
    parser.add_argument('--label_mask', nargs='+', default=[],help='Label mask to process')
    parser.add_argument('--input_scaling_method', type=str, choices=['standard', 'file_weight', 'fixed', 'all_standard'], default='standard', help='Scaling method to use for input data')
    parser.add_argument('--target_scaling_method', type=str, choices=['standard', 'file_weight', 'fixed', 'all_standard', 'hybrid', 'hybrid_v2'], default='standard', help='Scaling method to use for target data')
    parser.add_argument('--file_weight_filepath', type=str, default=None, help='Filename for the output file weight')
    parser.add_argument('--fixed_weight', type=float, default=None, help='Scalar weight to apply to input file weights')
    parser.add_argument('--is_graph_level', type=lambda x: x.lower() == 'true', default=False, help='Is graph level prediction')
    parser.add_argument('--fold_name', type=str, default='', help='Fold name for K-fold cross-validation')
    parser.add_argument('--is_e3nn', type=lambda x: x.lower() == 'true', default=False, help='Is e3nn')
    parser.add_argument('--is_continuous', type=lambda x: x.lower() == 'true', default=False, help='Continuous training from last interception')
    parser.add_argument('--skip_config_generation', type=lambda x: x.lower() == 'true', default=False, help='Skip (re)generating config files; assume they already exist. Used by run_all_gpus.sh so concurrent workers do not race on the config directory.')
    parser.add_argument('--run_final_test', type=lambda x: x.lower() == 'true', default=True, help='After CV selection, retrain the best config on the pooled non-test data and evaluate the held-out test set ONCE. Set false to skip the test set entirely (selection by CV validation only).')
    parser.add_argument('--final_test_only', type=lambda x: x.lower() == 'true', default=False, help='Skip the CV loop; run ONLY the final retrain-on-pooled-data + single test evaluation, on the config given by --final_test_config (or the CV-summary winner if unset).')
    parser.add_argument('--final_test_config', type=int, default=-1, help='Config index to use for the final test (with --final_test_only). If <0, read the winner from output/<exp>/cv_summary_<device_id>.txt.')
    parser.add_argument('--final_test_epochs', type=int, default=0, help='If >0, override the config n_epoch for the final-model training (e.g., a longer horizon than CV used).')
    parser.add_argument('--train_on_all', type=lambda x: x.lower() == 'true', default=False, help='Train a final DEPLOYMENT model on ALL data (train + val + test pooled), using the single winning config in --meta_config. No held-out set, so NO test metric is produced; use only after selection/evaluation are complete.')
    parser.add_argument('--device_id', type=int, default=0, help='GPU device ID to use (default: 0)')
    parser.add_argument('--num_gpu', type=int, default=1, help='Number of GPUs to use')

    if parser.parse_args().experiment_name is None:
        raise ValueError("experiment_name is required")
    if parser.parse_args().meta_config is None:
        raise ValueError("meta_config is required")

    return parser