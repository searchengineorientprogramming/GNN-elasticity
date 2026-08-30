import torch
from torch_geometric.loader import DataLoader
from src.model import init_model
from src.utility_functions import msg, check_feature_dimension
import numpy as np

def normalized_mae_loss(pred, target, return_per_output=False, epsilon=1e-8, range_val=None):
    """
    Computes MAE normalized by the range of each output.

    - If `return_per_output=True`, returns a tensor of shape (num_outputs,).
    - If `return_per_output=False`, returns a single scalar loss value.
    - `range_val` (per-output tensor): the FIXED normalizer to divide by. Pass the
      full-training-set per-output range so the denominator does not change from
      one mini-batch to the next. If None, the range is estimated from the current
      batch (legacy behavior, very noisy for small batches).
    """
    if range_val is None:
        true_max = target.max(dim=0)[0]  # Shape: (num_outputs,)
        true_min = target.min(dim=0)[0]  # Shape: (num_outputs,)
        range_val = true_max - true_min  # Shape: (num_outputs,)

    mae_per_output = torch.mean(torch.abs(pred - target), dim=0) / (range_val + epsilon)  # Shape: (num_outputs,)

    if return_per_output:
        return mae_per_output  # Keep per-output loss for training
    else:
        return mae_per_output.mean()  # Return scalar for validation


def train_model(params, 
                train_data, 
                valid_data,  # Added validation data
                output_model_dir, 
                config_file, 
                target,
                target_dim,
                is_graph_level,
                device, 
                is_e3nn,
                verbose=False):

    learning_rate = params['l_rate']
    weight_decay = params['w_decay']
    n_epoch = params['n_epoch']
    batch_size = params['batch_size']
    hidden_layer_size = params['hidden_size']
    num_hidden_layers = params['num_layers']
    layer_type = params['layer_type']
    activation = params['activation']
    use_edge_weight = params.get('use_edge_weight', False)
    use_amp = params.get('use_amp', False)  # bf16 mixed precision (no GradScaler needed)

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_data, batch_size=batch_size, shuffle=False)

    top_layer_input_size = check_feature_dimension(train_data)

    # Model training
    msg('\n====== Training / Testing ======', verbose=verbose)
    
    model, optimizer = init_model(hidden_layer_size=hidden_layer_size, 
                                  num_hidden_layers=num_hidden_layers, 
                                  top_layer_input_size=top_layer_input_size,
                                  output_size=target_dim,
                                  is_graph_level=is_graph_level,
                                  learning_rate=learning_rate, 
                                  weight_decay=weight_decay, 
                                  layer_type=layer_type,
                                  activation=activation,
                                  device=device,
                                  is_e3nn=is_e3nn,
                                  use_edge_weight=use_edge_weight,
                                  verbose=verbose)

    train_losses = []
    val_losses = []

    weights = torch.ones(target_dim).to(device)  # initial value; reassigned per batch below

    # Fixed per-output target range over the WHOLE training set. Using this as the
    # normalized-MAE denominator (instead of recomputing max-min from each
    # 8-sample mini-batch) keeps the loss/gradient scale stable across batches and
    # makes the train and validation losses directly comparable.
    train_targets = torch.cat(
        [getattr(record, target).reshape(1, -1) for record in train_data], dim=0
    ).to(device)
    target_range = (train_targets.max(dim=0)[0] - train_targets.min(dim=0)[0]).detach()

    weight_history = []

    # Print a clean, aligned, periodically-sampled loss line instead of one per epoch
    # (the full per-epoch curve is still written to loss_log). ~20 evenly spaced rows
    # plus the first and last epoch.
    log_every = max(1, n_epoch // 20)
    msg(f"{'epoch':>11} | {'train_loss':>11} | {'val_loss':>11}", verbose=verbose)

    for epoch in range(n_epoch):
        model.train()
        epoch_train_loss = 0.0

        for train_batch in train_loader:
            train_batch = train_batch.to(device)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_amp):
                train_pred = model(train_batch)
            train_true = getattr(train_batch, target, None)
            if train_true is None:
                raise AttributeError(f"'train_batch' object has no attribute '{target}'")
            # Compute the loss in fp32 even under AMP (the model ran in bfloat16).
            train_pred = train_pred.reshape(train_true.shape).float()
            per_output_mae = normalized_mae_loss(train_pred, train_true, return_per_output=True, range_val=target_range)

            weights = per_output_mae / (per_output_mae.sum() + 1e-8)
            weights = weights.detach()  # Prevent gradient tracking

            weighted_loss = torch.sum(weights * per_output_mae)

            weighted_loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            epoch_train_loss += weighted_loss.item()

        weight_history.append(weights.cpu().numpy())
        # Validation loss calculation
        model.eval()
        epoch_val_loss = 0.0
        with torch.no_grad():
            for valid_batch in valid_loader:
                valid_batch = valid_batch.to(device)
                with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_amp):
                    valid_pred = model(valid_batch)
                valid_true = getattr(valid_batch, target, None)
                valid_pred = valid_pred.reshape(valid_true.shape).float()
                val_loss = normalized_mae_loss(valid_pred, valid_true, return_per_output=False, range_val=target_range)
                epoch_val_loss += val_loss.item()

        epoch_train_loss /= len(train_loader)
        epoch_val_loss /= len(valid_loader)

        train_losses.append(epoch_train_loss)
        val_losses.append(epoch_val_loss)

        if epoch == 0 or (epoch + 1) % log_every == 0 or epoch == n_epoch - 1:
            msg(f"{epoch+1:>5}/{n_epoch:<5} | {epoch_train_loss:>11.6f} | {epoch_val_loss:>11.6f}", verbose=verbose)

        if epoch % 100 == 0:
            torch.save(model, f"{output_model_dir}/config_{config_file}_{epoch}.pth")
            np.save(f"{output_model_dir}/weight_history_{config_file}_{epoch}.npy", np.array(weight_history))

    torch.save(model, f"{output_model_dir}/config_{config_file}_final.pth")

    np.save(f"{output_model_dir}/weight_history_{config_file}_final.npy", np.array(weight_history))

    return model, optimizer, train_losses, val_losses
