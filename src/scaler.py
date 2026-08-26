import numpy as np
import torch
from sklearn.preprocessing import StandardScaler, QuantileTransformer
from .utility_functions import save_scaler
import pickle

def standard_scaling_fit_transform(geo_data_list, attr, save_filepath):
    # Fit and transform the data, can take list of records as input
    scaler = StandardScaler()
    data_np = np.concatenate([data[attr] for data in geo_data_list], axis=0)
    scaler.fit(data_np)
    save_scaler(scaler, save_filepath)
    for i in range(len(geo_data_list)):
        geo_data_list[i][attr] =  torch.from_numpy(scaler.transform(geo_data_list[i][attr])).to(torch.float32)
    return scaler, geo_data_list

def hybrid_scaling_fit_transform(geo_data_list, attr, save_filepath):
    """
    Applies QuantileTransformer to selected indices and StandardScaler to others.
    Saves the transformers for inverse transformation.
    """
    log_transform_outputs = {1, 2, 6, 7, 18}  # Log-transformed features
    quantile_transform_outputs = {0, 11, 15, 20}  # Multi-modal features
    num_features = geo_data_list[0][attr].shape[1]  # Number of output dimensions

    scalers = {}  # Dictionary to store transformers per feature
    data_np = np.concatenate([data[attr].numpy() for data in geo_data_list], axis=0)

    # Apply appropriate transformation to each output dimension
    for i in range(num_features):
        if i in log_transform_outputs or i in quantile_transform_outputs:
            scalers[i] = QuantileTransformer(output_distribution='normal', n_quantiles=100)
        else:
            scalers[i] = StandardScaler()
        
        scalers[i].fit(data_np[:, i].reshape(-1, 1))

    # Save the scalers
    with open(save_filepath, 'wb') as f:
        pickle.dump(scalers, f)

    # Apply transformations to training data
    for i in range(len(geo_data_list)):
        transformed_data = np.zeros_like(geo_data_list[i][attr].numpy())
        for j in range(num_features):
            transformed_data[:, j] = scalers[j].transform(geo_data_list[i][attr][:, j].reshape(-1, 1)).flatten()
        geo_data_list[i][attr] = torch.from_numpy(transformed_data).to(torch.float32)

    return scalers, geo_data_list

def per_output_standard_scaling_fit_transform(geo_data_list, attr, save_filepath):
    """
    Applies per-output StandardScaler independently for each input/output feature.
    Saves the per-output scalers for later transformation and inverse transformation.
    """
    num_features = geo_data_list[0][attr].shape[1]  # Get number of features in x or y
    scalers = [StandardScaler() for _ in range(num_features)]  # One scaler per feature

    # Convert list of tensors to numpy array
    data_np = np.concatenate([data[attr].numpy() for data in geo_data_list], axis=0)

    # Apply independent scaling to each feature dimension
    for i in range(num_features):
        scalers[i].fit(data_np[:, i].reshape(-1, 1))

    # Save scalers
    with open(save_filepath, 'wb') as f:
        pickle.dump(scalers, f)

    # Transform dataset
    for i in range(len(geo_data_list)):
        transformed_data = np.zeros_like(geo_data_list[i][attr].numpy())
        for j in range(num_features):
            transformed_data[:, j] = scalers[j].transform(geo_data_list[i][attr][:, j].reshape(-1, 1)).flatten()
        geo_data_list[i][attr] = torch.from_numpy(transformed_data).to(torch.float32)

    return scalers, geo_data_list

def standard_scaling_transform(geo_data_list, attr, scaler):
    # Transform the data, can take list of records as input
    for i in range(len(geo_data_list)):
        geo_data_list[i][attr] =  torch.from_numpy(scaler.transform(geo_data_list[i][attr])).to(torch.float32)
    return geo_data_list


def hybrid_scaling_fit_transform_v2(geo_data_list, attr, save_filepath):
    """
    Applies Log Transformation to specific outputs, QuantileTransformer to others, 
    and StandardScaler to the remaining features. Saves the scalers for inverse transformation.
    """
    log_transform_outputs = {1, 2, 6, 7, 18}  # Log-transformed features
    quantile_transform_outputs = {0, 11, 15, 20}  # Multi-modal features
    num_features = geo_data_list[0][attr].shape[1]  # Number of output features

    scalers = {}  # Dictionary to store transformers per feature
    epsilon = 1e-8  # Small value to prevent log(0)
    
    # Convert dataset to a single NumPy array for fitting scalers
    data_np = np.concatenate([data[attr].numpy() for data in geo_data_list], axis=0)

    # Apply appropriate transformation per feature
    for i in range(num_features):
        if i in log_transform_outputs:
            scalers[i] = "log"  # Mark as log-transformed
            data_np[:, i] = np.log1p(data_np[:, i] + epsilon)  # Apply log1p (log(x + 1))
        elif i in quantile_transform_outputs:
            scalers[i] = QuantileTransformer(output_distribution='normal', n_quantiles=100)
            scalers[i].fit(data_np[:, i].reshape(-1, 1))
        else:
            scalers[i] = StandardScaler()
            scalers[i].fit(data_np[:, i].reshape(-1, 1))

    # Save the scalers
    with open(save_filepath, 'wb') as f:
        pickle.dump(scalers, f)

    # Apply transformations to training data
    for i in range(len(geo_data_list)):
        transformed_data = np.zeros_like(geo_data_list[i][attr].numpy())
        for j in range(num_features):
            if j in log_transform_outputs:
                transformed_data[:, j] = np.log1p(geo_data_list[i][attr][:, j].numpy() + epsilon)
            else:
                transformed_data[:, j] = scalers[j].transform(geo_data_list[i][attr][:, j].reshape(-1, 1)).flatten()
        geo_data_list[i][attr] = torch.from_numpy(transformed_data).to(torch.float32)

    return scalers, geo_data_list