
import logging
import torch
import numpy as np
import pandas as pd
from torch_geometric.data import Data
from torch_geometric.nn import knn_graph, radius_graph
from torch_geometric.utils import remove_isolated_nodes, to_undirected
from typing import Optional, Dict, Any, Union, List

def build_nuclei_graph(
    nuclei_table: pd.DataFrame,
    method: str,
    params: Dict[str, Any],
    feature_cols: Optional[List[str]] = None,
    coordinate_cols: Optional[List[str]] = None,
) -> Data:
    """
    Builds a PyTorch Geometric Graph from a nuclei table (DataFrame).

    Args:
        nuclei_table (pd.DataFrame): DataFrame containing nuclei information.
        method (str): Graph construction method ('knn' or 'radius').
        params (Dict[str, Any]): Dictionary of parameters for the chosen method.
            - k (int): Number of neighbors for kNN.
            - r (float): Radius for radius graph.
            - max_edge_length (float, optional): Maximum edge length for pruning.
            - remove_isolated_nodes (bool, optional): Whether to remove isolated nodes.
            - mpp (float, optional): Microns per pixel.
            - coord_space (str, optional): Coordinate space name (default: "level-0-pixels").
        feature_cols (List[str], optional): List of column names to use as node features.
            If None, attempts to use all columns except coordinates and IDs.
        coordinate_cols (List[str], optional): List of column names for x and y coordinates.
            Defaults to ['centroid_x', 'centroid_y'].

    Returns:
        torch_geometric.data.Data: Constructed graph with:
            - x: Node features matrix.
            - pos: Node positions (x, y).
            - edge_index: Graph connectivity.
            - edge_attr: Edge attributes (distance).
            - kwargs: Metadata (mpp, coord_space).
    """
    
    # 1. Extract Coordinates
    if coordinate_cols is None:
        # Try to infer coordinate columns
        if 'centroid_x' in nuclei_table.columns and 'centroid_y' in nuclei_table.columns:
            coordinate_cols = ['centroid_x', 'centroid_y']
        elif 'x' in nuclei_table.columns and 'y' in nuclei_table.columns:
            coordinate_cols = ['x', 'y']
        else:
            raise ValueError("Could not infer coordinate columns. Please specify `coordinate_cols`.")
    
    pos = torch.tensor(nuclei_table[coordinate_cols].values, dtype=torch.float)

    # 2. Extract Features
    if feature_cols is None:
        # Use all columns except coordinates
        feature_cols = [c for c in nuclei_table.columns if c not in coordinate_cols]
    
    # Filter only numeric columns for features
    feature_cols = [c for c in feature_cols if pd.api.types.is_numeric_dtype(nuclei_table[c])]
    
    if not feature_cols:
         # If no features, use a dummy feature (e.g., all ones or just skip x)
         # PyG usually expects x to be present.
         x = torch.ones((pos.shape[0], 1), dtype=torch.float)
    else:
        x = torch.tensor(nuclei_table[feature_cols].values, dtype=torch.float)

    # 3. Build Graph
    edge_index = None
    
    # Use sklearn for graph construction to avoid complex dependencies (torch-cluster, etc.)
    from sklearn.neighbors import NearestNeighbors
    
    pos_np = pos.numpy()
    
    if method.lower() == 'knn':
        k = params.get('k', 5)
        # sklearn includes the point itself as the first neighbor, so we ask for k+1
        nbrs = NearestNeighbors(n_neighbors=k+1, algorithm='auto').fit(pos_np)
        distances, indices = nbrs.kneighbors(pos_np)
        
        # Create edge list
        # We skip the first column because it's the node itself (dist 0)
        # Flatten excluding the first column
        sources = np.repeat(np.arange(pos_np.shape[0]), k)
        targets = indices[:, 1:].flatten()
        
        edge_index = torch.from_numpy(np.stack([sources, targets])).long()
        
    elif method.lower() == 'radius':
        r = params.get('r', 50.0)
        nbrs = NearestNeighbors(radius=r, algorithm='auto').fit(pos_np)
        distances, indices = nbrs.radius_neighbors(pos_np)
        
        sources = []
        targets = []
        for i, neighbors in enumerate(indices):
            # neighbors includes the point itself if distance is 0, usually depending on implementation
            # radius_neighbors returns points within radius. It might include itself.
            # We should exclude self-loops explicitly
            valid_neighbors = neighbors[neighbors != i]
            if len(valid_neighbors) > 0:
                sources.append(np.full(len(valid_neighbors), i))
                targets.append(valid_neighbors)
        
        if sources:
            sources = np.concatenate(sources)
            targets = np.concatenate(targets)
            edge_index = torch.from_numpy(np.stack([sources, targets])).long()
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long)
        
    elif method.lower() == 'delaunay':
        # Optional: Delaunay
        try:
            from scipy.spatial import Delaunay
        except ImportError:
            raise ImportError("scipy is required for Delaunay triangulation.")
        
        tri = Delaunay(pos_np)
        edges = []
        for simplex in tri.simplices:
            edges.append([simplex[0], simplex[1]])
            edges.append([simplex[1], simplex[2]])
            edges.append([simplex[2], simplex[0]])
            
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        edge_index = to_undirected(edge_index)

    else:
        raise ValueError(f"Unknown method: {method}")

    # 4. Compute Edge Attributes (Distance)
    if edge_index.size(1) > 0:
        row, col = edge_index
        # Use torch for distance computation to stay in tensor land
        dist = (pos[row] - pos[col]).norm(dim=-1)
        edge_attr = dist.view(-1, 1)
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, 1), dtype=torch.float)

    # 5. Pruning
    
    # Max Edge Length Pruning
    max_edge_length = params.get('max_edge_length')
    if max_edge_length is not None and edge_index.size(1) > 0:
        mask = edge_attr.squeeze() <= max_edge_length
        edge_index = edge_index[:, mask]
        edge_attr = edge_attr[mask]

    # Remove Isolated Nodes
    if params.get('remove_isolated_nodes', False):
        num_nodes = x.size(0)
        edge_index, edge_attr, mask = remove_isolated_nodes(edge_index, edge_attr, num_nodes=num_nodes)
        
    # 6. Metadata
    coord_space = params.get('coord_space', 'level-0-pixels')
    mpp = params.get('mpp', None)

    data = Data(x=x, pos=pos, edge_index=edge_index, edge_attr=edge_attr)
    
    # Assign custom attributes
    data.coord_space = coord_space
    if mpp is not None:
        data.mpp = mpp

    return data

def save_nuclei_graph(data: Data, out_path: str):
    """
    Saves the PyG Data object to a file.
    
    Args:
        data (Data): PyG Data object.
        out_path (str): Path to save the file. Supported extensions: .pt, .pth, .npz
    """
    if out_path.endswith('.pt') or out_path.endswith('.pth'):
        torch.save(data, out_path)
    elif out_path.endswith('.npz'):
        # Save as numpy arrays
        np.savez(
            out_path,
            x=data.x.numpy(),
            pos=data.pos.numpy(),
            edge_index=data.edge_index.numpy(),
            edge_attr=data.edge_attr.numpy() if data.edge_attr is not None else None,
            coord_space=getattr(data, 'coord_space', 'level-0-pixels'),
            mpp=getattr(data, 'mpp', None)
        )
    else:
         # Default to torch save
         torch.save(data, out_path)

