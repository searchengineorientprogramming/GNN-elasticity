import torch
from torch_geometric.loader import DataLoader
from src.variable import *
import numpy as np

def predict_and_transform_new(model, data, batch_size, device, target_scaling_method, target_scaler=None, transform:bool=True):
    loader = DataLoader(data, batch_size=batch_size, shuffle=False)
    all_preds = []
    all_trues = []
    for batch in loader:
        preds, trues = test(model, batch, TARGET, device)
        all_preds.append(preds)
        all_trues.append(trues)
    preds = torch.cat(all_preds).cpu().numpy()
    trues = torch.cat(all_trues).cpu().numpy()
    if preds.shape != trues.shape:
        trues = trues.reshape(preds.shape)
    
    if transform is True:
        if target_scaling_method == STD_SCALER:
            preds = target_scaler.inverse_transform(preds)
            trues = target_scaler.inverse_transform(trues)
        elif target_scaling_method == ALL_STD_SCALER:
            for i in range(len(target_scaler)):  # Loop through each output scaler
                preds[:, i] = target_scaler[i].inverse_transform(preds[:, i].reshape(-1, 1)).flatten()
                trues[:, i] = target_scaler[i].inverse_transform(trues[:, i].reshape(-1, 1)).flatten()
        elif target_scaling_method == HYBRID_SCALER:
            for i in range(preds.shape[1]):  
                preds[:, i] = target_scaler[i].inverse_transform(preds[:, i].reshape(-1, 1)).flatten()
                trues[:, i] = target_scaler[i].inverse_transform(trues[:, i].reshape(-1, 1)).flatten()
        elif target_scaling_method == HYBRID_SCALER_V2:
            epsilon = 1e-8
            for i in range(preds.shape[1]):  
                if target_scaler[i] == "log":
                    preds[:, i] = np.expm1(preds[:, i]) - epsilon  # Reverse log1p
                    trues[:, i] = np.expm1(trues[:, i]) - epsilon
                else:
                    preds[:, i] = target_scaler[i].inverse_transform(preds[:, i].reshape(-1, 1)).flatten()
                    trues[:, i] = target_scaler[i].inverse_transform(trues[:, i].reshape(-1, 1)).flatten()
    return preds, trues

def test(model, data, target, device):  # Fixed parameter name from data_loader to data
    model.eval()
    data = data.to(device)  # Fixed: directly use the batch data instead of trying to iterate
    with torch.no_grad():  # Added to prevent gradient computation during inference
        pred = model(data)
        true = getattr(data, target)
        if true is None:
            raise AttributeError(f"'data' object has no attribute '{target}'")
    return pred, true
