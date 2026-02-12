import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, SAGEConv, GATConv, global_mean_pool, global_add_pool, global_max_pool
from typing import Dict, Any, Optional, List, Union
import numpy as np

class GraphEncoder(nn.Module):
    """
    A simple GNN encoder for graph embedding.
    """
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int = 3,
        gnn_type: str = 'gcn',
        pooling: str = 'mean',
        dropout: float = 0.5
    ):
        super(GraphEncoder, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.gnn_type = gnn_type.lower()
        self.pooling = pooling.lower()
        self.dropout = dropout

        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()

        # Input layer
        self.convs.append(self._build_conv(input_dim, hidden_dim))
        self.bns.append(nn.BatchNorm1d(hidden_dim))

        # Hidden layers
        for _ in range(num_layers - 2):
            self.convs.append(self._build_conv(hidden_dim, hidden_dim))
            self.bns.append(nn.BatchNorm1d(hidden_dim))

        # Output layer (if more than 1 layer)
        if num_layers > 1:
            self.convs.append(self._build_conv(hidden_dim, hidden_dim)) # Keep hidden dim before pooling
            self.bns.append(nn.BatchNorm1d(hidden_dim))

        # Readout (global pooling) -> Final linear
        self.project = nn.Linear(hidden_dim, output_dim)


    def _build_conv(self, in_channels, out_channels):
        if self.gnn_type == 'gcn':
            return GCNConv(in_channels, out_channels)
        elif self.gnn_type == 'sage':
            return SAGEConv(in_channels, out_channels)
        elif self.gnn_type == 'gat':
            return GATConv(in_channels, out_channels, heads=1) # simplified single head
        else:
            raise ValueError(f"Unknown GNN type: {self.gnn_type}")

    def forward(self, x, edge_index, batch=None):
        if batch is None:
            batch = torch.zeros(x.shape[0], dtype=torch.long, device=x.device)

        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        # Global Pooling
        if self.pooling == 'mean':
            x = global_mean_pool(x, batch)
        elif self.pooling == 'add':
            x = global_add_pool(x, batch)
        elif self.pooling == 'max':
            x = global_max_pool(x, batch)
        else:
            raise ValueError(f"Unknown pooling type: {self.pooling}")
        
        # Final projection
        x = self.project(x)
        return x

def extract_gnn_embeddings(
    data: Data,
    model: Optional[nn.Module] = None,
    device: torch.device = None
) -> np.ndarray:
    """
    Extracts graph embeddings using a GNN model.
    
    Args:
        data (Data): PyG Data object.
        model (nn.Module, optional): Pre-loaded GNN model. 
                                     If None, a default random initialized model is used (for testing/baseline).
        device (torch.device, optional): Device to run on.
        
    Returns:
        np.ndarray: Graph embedding vector (1D or 2D if batched, but here usually 1 graph).
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if model is None:
        # Default fallback: Random GCN
        # Assume input features exist
        input_dim = data.x.shape[1] if data.x is not None else 1
        model = GraphEncoder(input_dim=input_dim, hidden_dim=64, output_dim=128)
        model.eval()

    model = model.to(device)
    data = data.to(device)
    
    with torch.no_grad():
        if data.x is None:
             # Create dummy features if missing (though they should exist)
             data.x = torch.ones((data.num_nodes, 1), device=device)

        embedding = model(data.x, data.edge_index, data.batch)
        
    return embedding.cpu().numpy()
