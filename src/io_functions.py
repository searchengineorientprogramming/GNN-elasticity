import os
import pickle
import torch
import numpy as np
import torch_geometric
import re
from src.utility_functions import natural_sort


def get_filepaths(input_dir, postfix):
    graph_filepaths = []
    input_dir_file_list = os.listdir(input_dir)
    if len(input_dir_file_list) == 0:
        raise ValueError(f"No files found in {input_dir}")
    if len(input_dir_file_list) == 1:
        filename =  input_dir_file_list[0]
        if postfix in filename:
            graph_filepath = os.path.join(input_dir, filename)
            graph_filepaths.append(graph_filepath)
    else:
        for filename in os.listdir(input_dir):
            # Split filename into parts by underscore
            parts = re.split('([0-9]+)', filename)
            # Check if this is a direct match with postfix (no extra text before postfix)
            if parts[-1] == postfix:
                graph_filepath = os.path.join(input_dir, filename)
                graph_filepaths.append(graph_filepath)
    return natural_sort(graph_filepaths)

def get_input_dir(entry_dir, texture, input_dir_prefix, subfolder):
    texture_folder = f'{input_dir_prefix}_{texture}' if input_dir_prefix is not None else f'{texture}'
    input_dir = os.path.join(entry_dir, texture_folder, subfolder)
    if not os.path.exists(input_dir):
        raise ValueError(f"Directory {input_dir} does not exist.")
    return input_dir

def load_graph_data(graph_filepaths):
    if type(graph_filepaths) == list:
        graph_filepaths = natural_sort(graph_filepaths)
        graph_data_list = []
        for i, graph_filepath in enumerate(graph_filepaths):
            if (i + 1) % 10 == 0:
                print(f"Processing graph file {i+1}/{len(graph_filepaths)}")
            
            with open(graph_filepath, 'rb') as f:
                graph_data = pickle.load(f)
            graph_data = torch_geometric.utils.from_networkx(graph_data)
            graph_data_list.append(graph_data)
        return graph_data_list

    if type(graph_filepaths) == str:
        with open(graph_filepaths, 'rb') as f:
            graph_data = pickle.load(f)
        graph_data = torch_geometric.utils.from_networkx(graph_data)
        return graph_data
    
    raise ValueError("Invalid input type for graph_filepaths")

def read_file(input_filename, post_fix, graph_data=None, input_graph_postfix=None):
    if post_fix.endswith(".npy"):
        input_tmp = np.load(input_filename)
        input_tmp = input_tmp.reshape(input_tmp.shape[0], -1) if len(input_tmp.shape) >= 2 else input_tmp
    elif post_fix in [".C", ".S"]:
        if post_fix == input_graph_postfix:
            input_tmp = graph_data.x.numpy() # Load from pre-loaded graph data
        else:
            graph_data = load_graph_data(input_filename) # Load from file
            input_tmp = graph_data.x.numpy()
    else:
        raise ValueError(f"Unsupported file format for {input_filename}")
    return input_tmp

def read_target_file(target_filepath):
    if target_filepath.endswith('.txt'):
        with open(target_filepath, 'r') as file:
            target = np.genfromtxt(file, skip_header=0, dtype=None, encoding=None)
    elif target_filepath.endswith('.npy'):
        target = np.load(target_filepath)
    else:
        raise ValueError(f"Unsupported file format for {target_filepath}")
    return target

def save_train_test_valid_data(textures, 
                               all_graph_data, 
                               all_input_data, 
                               all_target_data, 
                               train_indices, 
                               test_indices, 
                               val_indices,
                               output_dir,
                               fold_dir=''):
    
    train_data = []
    test_data = []
    val_data = []

    num_experiments_per_texture = len(all_graph_data[0])
    for i, texture in enumerate(textures):
        for j in range(num_experiments_per_texture):
            record = all_graph_data[i][j]
            record.x = torch.from_numpy(all_input_data[i][j]).to(torch.float32)
            record.y = torch.from_numpy(all_target_data[i][j]).to(torch.float32)
            
            if j in train_indices:
                train_data.append(record)
            elif j in test_indices:
                test_data.append(record)
            elif j in val_indices:
                val_data.append(record)

    with open(os.path.join(output_dir, fold_dir, "train.pkl"), 'wb') as train_file:
        pickle.dump(train_data, train_file, protocol=pickle.HIGHEST_PROTOCOL)
    
    with open(os.path.join(output_dir, fold_dir, "test.pkl"), 'wb') as test_file:
        pickle.dump(test_data, test_file, protocol=pickle.HIGHEST_PROTOCOL)
    
    with open(os.path.join(output_dir, fold_dir, "val.pkl"), 'wb') as val_file:
        pickle.dump(val_data, val_file, protocol=pickle.HIGHEST_PROTOCOL)
