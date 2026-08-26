import os
import json
import glob
from itertools import product

def generate_configs(config_dir, start_index, hyperparameter_grid:dict):

    if not os.path.exists(config_dir):
        os.makedirs(config_dir, exist_ok=True)
    elif not start_index:
        # Fresh generation (start_index 0/None): remove stale config_*.json so a
        # shrunk hyperparameter grid does not leave orphan configs behind that the
        # training loop (which lists the whole directory) would otherwise train.
        for stale in glob.glob(os.path.join(config_dir, "*.json")):
            os.remove(stale)

    keys = hyperparameter_grid.keys()
    values = hyperparameter_grid.values()
    
    index = start_index or 0
    for combination in product(*values):
        config = dict(zip(keys, combination))
        
        config_filepath = os.path.join(config_dir, f"{index}.json")
        with open(config_filepath, 'w') as h:
            json.dump(config, h, indent=4)
        index += 1