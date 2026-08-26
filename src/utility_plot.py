import matplotlib.pyplot as plt
import numpy as np
import os
import seaborn as sns

def plot_results(train_preds, train_trues, test_trues, test_preds, output_dir, fname):
    '''Plot evaluation results
    '''
    sns.set_style("ticks")
    fig, ax = plt.subplots(figsize=(8.5, 5.5), dpi=100)

    # Determine the maximum value for setting the limits
    max_value = max(np.max(train_trues), np.max(train_preds), np.max(test_trues), np.max(test_preds))
    min_value = min(np.min(train_trues), np.min(train_preds), np.min(test_trues), np.min(test_preds))
    plt.plot([min_value, max_value], [min_value, max_value], 'gray', linewidth=1, zorder=1)
    plt.scatter(train_trues, train_preds, s=1, color='gray', label='Train')
    plt.scatter(test_trues, test_preds, color='deepskyblue', s=1, label='Test')

    plt.xlabel('True')
    plt.ylabel('Predicted')
    plt.xlim([min_value, max_value])
    plt.ylim([min_value, max_value])

    ax.set_aspect(1.0/ax.get_data_ratio(), adjustable='box')
    ax.legend(loc="lower right", fontsize=10)

    if output_dir is not None:
        fname = os.path.join(output_dir, fname)
        plt.savefig(fname, dpi=300, bbox_inches="tight")
        plt.close(fig)
