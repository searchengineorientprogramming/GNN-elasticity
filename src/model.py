import math
import torch
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, GCNConv, GraphConv, TransformerConv, GMMConv, global_mean_pool
from .gatconv import GATConv
from .utility_functions import msg


def get_conv_layer(layer_type):
    if layer_type == "SAGEConv":
        return SAGEConv
    if layer_type == "GCNConv":
        return GCNConv
    if layer_type == "GraphConv":
        return GraphConv
    if layer_type == "TransformerConv":
        return TransformerConv
    if layer_type == "GMMConv":
        return GMMConv
    if layer_type == "GATConv":
        return GATConv
    else:
        raise ValueError(f"Invalid layer type: {layer_type}")

def shifted_softplus(x):
    """SchNet's shifted softplus: ssp(x) = ln(0.5*e^x + 0.5) = softplus(x) - ln(2).

    Smooth and infinitely differentiable, negative-capable, and ssp(0) = 0 so it is
    zero-centered at the origin (good for standardized features without norm layers).
    Defined at module level (not a lambda) so a saved model can be unpickled.
    """
    return F.softplus(x) - math.log(2.0)


def get_activation_function(name):
    if name == "elu":
        return F.elu
    elif name == "relu":
        return F.relu
    elif name == "leaky_relu":
        return F.leaky_relu
    elif name == "tanh":
        return F.tanh
    elif name == "silu":  # a.k.a. Swish; DimeNet default
        return F.silu
    elif name == "gelu":
        return F.gelu
    elif name == "shifted_softplus":  # SchNet default
        return shifted_softplus
    elif name == "none":
        return None
    elif name == "softplus":
        return F.softplus
    else:
        raise ValueError(f"Invalid activation function: {name}")

from torch.nn import Linear as TorchLinear


class GNN(torch.nn.Module):
    def __init__(self, 
                 hidden_layer_size, 
                 num_hidden_layers, 
                 top_layer_input_size, 
                 output_size, 
                 is_graph_level,
                 layer_type,
                 activation,
                 device,
                 use_edge_weight=False):
        super(GNN, self).__init__()

        self.convs = torch.nn.ModuleList()
        self.conv_layer = get_conv_layer(layer_type)
        self.activation = get_activation_function(activation)
        self.device = device
        self.is_graph_level = is_graph_level
        self.use_edge_weight = use_edge_weight

        conv_args = {}

        if layer_type == "GMMConv":
            conv_args["dim"] = 1
            conv_args["kernel_size"] = 16

        # When using the scalar edge_weight as a 1-D edge feature, attention-style
        # convs need edge_dim=1 so they build their edge-feature projection
        # (lin_edge). GCNConv/GraphConv take edge_weight directly in forward().
        if use_edge_weight and layer_type in ("TransformerConv", "GATConv"):
            conv_args["edge_dim"] = 1

        additional_args = {key: arg for key, arg in conv_args.items() if arg is not None}

        self.top_layer_conv_args_input = {"in_channels": top_layer_input_size, 
                                          "out_channels": hidden_layer_size, 
                                          **additional_args}
        self.inter_layer_conv_args_input = {"in_channels": hidden_layer_size, 
                                            "out_channels": hidden_layer_size, 
                                            **additional_args}
        
        # Top layer
        self.convs.append(self.conv_layer(**self.top_layer_conv_args_input))

        for _ in range(num_hidden_layers):
            # Inter layer
            self.convs.append(self.conv_layer(**self.inter_layer_conv_args_input))
        
        # Output layer
        self.out = TorchLinear(hidden_layer_size, output_size)

    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch

        # Optional scalar edge weights (grain-boundary contact counts). They are
        # integer counts spanning a wide range (~1-640), so log1p-normalize them
        # before use. edge_feat (E, 1) feeds attention/transformer/GMM convs as an
        # edge feature; edge_scalar (E,) feeds GCN/GraphConv as edge_weight.
        edge_feat = None
        edge_scalar = None
        if self.use_edge_weight:
            ew = getattr(data, "edge_weight", None)
            if ew is not None:
                edge_scalar = torch.log1p(ew.to(x.dtype))
                edge_feat = edge_scalar.view(-1, 1)

        if isinstance(self.convs[0], GATConv):
            self.attention_weights = []
            self.edge_index = []
        for conv in self.convs:
            if isinstance(conv, GMMConv):
                dim = self.inter_layer_conv_args_input["dim"]
                if edge_feat is not None:
                    pseudo = edge_feat
                elif edge_attr is not None:
                    pseudo = edge_attr
                else:
                    pseudo = torch.ones(edge_index.shape[1], dim, device=self.device)
                x = conv(x, edge_index, pseudo)
            elif isinstance(conv, GATConv):
                # Do NOT overwrite edge_index/edge_feat: keep them consistent across
                # layers (the conv adds self-loops internally per call).
                x, (ei_out, attention_weights) = conv(
                    x, edge_index, edge_attr=edge_feat, return_attention_weights=True)
                self.attention_weights.append(attention_weights)
                self.edge_index.append(ei_out)
            elif isinstance(conv, TransformerConv):
                x = conv(x, edge_index, edge_attr=edge_feat)
            elif isinstance(conv, (GCNConv, GraphConv)):
                x = conv(x, edge_index, edge_weight=edge_scalar)
            else:
                x = conv(x, edge_index)
            if self.activation is not None:
                x = self.activation(x)
        if self.is_graph_level:
            x = global_mean_pool(x, batch)
        x = self.out(x)

        return x

def init_model(hidden_layer_size, 
               num_hidden_layers, 
               top_layer_input_size,
               output_size,
               is_graph_level,
               learning_rate, 
               weight_decay,
               layer_type,
               activation,
               device,
               is_e3nn,
               use_edge_weight=False,
               verbose=True):
    model = GNN(hidden_layer_size=hidden_layer_size,
                num_hidden_layers=num_hidden_layers,
                top_layer_input_size=top_layer_input_size,
                output_size=output_size,
                is_graph_level=is_graph_level,
                layer_type=layer_type,
                activation=activation,
                device=device,
                use_edge_weight=use_edge_weight).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    msg("Initialized model:", verbose=verbose)
    msg(model, verbose=verbose)
    msg("\n", verbose=verbose)
    
    return model, optimizer
