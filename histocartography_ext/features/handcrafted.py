import numpy as np
import networkx as nx
from torch_geometric.data import Data
from torch_geometric.utils import to_networkx, degree
import torch
from typing import Dict, Any

def extract_graph_stats(data: Data) -> Dict[str, float]:
    """
    Extracts handcrafted statistics from a PyTorch Geometric Data object.
    
    Args:
        data (Data): PyG Data object containing the graph.
        
    Returns:
        Dict[str, float]: Dictionary of extracted statistics.
    """
    stats = {}
    
    # helper for safe scalar extraction
    def safe_item(val):
        if torch.is_tensor(val):
            return val.item()
        return val

    num_nodes = data.num_nodes
    num_edges = data.num_edges
    
    stats['num_nodes'] = num_nodes
    stats['num_edges'] = num_edges
    
    # Degree statistics
    if data.edge_index is not None and data.edge_index.numel() > 0:
        d = degree(data.edge_index[0], num_nodes=num_nodes)
        stats['degree_mean'] = safe_item(d.mean())
        stats['degree_std'] = safe_item(d.std())
        
        # Degree histogram bins (normalized)
        # Bins: 0-2, 3-5, 6-8, 9+ (customizable)
        bins = [0, 3, 6, 9, float('inf')]
        hist = torch.histc(d, bins=len(bins)-1, min=0, max=12) # Approximation
        # A more manual approach for specific bins
        stats['degree_bin_0_2'] = safe_item(((d >= 0) & (d < 3)).sum() / num_nodes)
        stats['degree_bin_3_5'] = safe_item(((d >= 3) & (d < 6)).sum() / num_nodes)
        stats['degree_bin_6_8'] = safe_item(((d >= 6) & (d < 9)).sum() / num_nodes)
        stats['degree_bin_9_plus'] = safe_item((d >= 9).sum() / num_nodes)

    else:
        stats['degree_mean'] = 0.0
        stats['degree_std'] = 0.0
        stats['degree_bin_0_2'] = 1.0 # All isolated
        stats['degree_bin_3_5'] = 0.0
        stats['degree_bin_6_8'] = 0.0
        stats['degree_bin_9_plus'] = 0.0

    # Convert to NetworkX for structural properties
    # to_networkx might be slow for very large graphs, but WSI graphs are usually reasonable (<100k nodes? maybe large)
    # If graph is too large, we might want to skip some complex stats or approximate them.
    # For now, we assume it's feasible or we use a limit.
    
    # Check graph feature size before conversion if needed.
    
    try:
        G = to_networkx(data, to_undirected=True)
        
        # Clustering Coefficient (Average)
        # Average clustering can be slow on large graphs.
        if num_nodes < 50000: # Heuristic limit
            stats['avg_clustering_coeff'] = nx.average_clustering(G)
        else:
             stats['avg_clustering_coeff'] = -1.0 # Skipped
             
        # Assortativity (Degree)
        if num_edges > 0:
             try:
                stats['degree_assortativity'] = nx.degree_assortativity_coefficient(G)
             except Exception:
                stats['degree_assortativity'] = 0.0
        else:
             stats['degree_assortativity'] = 0.0
             
        # Connected Components
        if num_nodes > 0:
            # nx.connected_components returns a generator
            components = sorted(nx.connected_components(G), key=len, reverse=True)
            stats['num_connected_components'] = len(components)
            stats['largest_cc_ratio'] = len(components[0]) / num_nodes if components else 0.0
        else:
            stats['num_connected_components'] = 0
            stats['largest_cc_ratio'] = 0.0
            
    except Exception as e:
        # Fallback if NetworkX conversion or calc fails
        print(f"Warning: NetworkX stats failed: {e}")
        stats['avg_clustering_coeff'] = 0.0
        stats['degree_assortativity'] = 0.0
        stats['num_connected_components'] = 0
        stats['largest_cc_ratio'] = 0.0

    # Edge Length Statistics
    if data.edge_attr is not None and data.edge_attr.numel() > 0:
        # Assuming edge_attr contains distance as the first column
        # If edge_attr is [E, D], we take [:, 0]
        if data.edge_attr.dim() > 1:
            edge_lengths = data.edge_attr[:, 0]
        else:
            edge_lengths = data.edge_attr
            
        stats['edge_length_mean'] = safe_item(edge_lengths.mean())
        stats['edge_length_std'] = safe_item(edge_lengths.std())
        
        q = torch.quantile(edge_lengths, torch.tensor([0.25, 0.5, 0.75]).to(edge_lengths.device))
        stats['edge_length_q25'] = safe_item(q[0])
        stats['edge_length_median'] = safe_item(q[1])
        stats['edge_length_q75'] = safe_item(q[2])
    else:
        stats['edge_length_mean'] = 0.0
        stats['edge_length_std'] = 0.0
        stats['edge_length_q25'] = 0.0
        stats['edge_length_median'] = 0.0
        stats['edge_length_q75'] = 0.0

    # Spatial Dispersion (kNN distance quantiles)
    # We can infer this from edge lengths if the graph is a kNN graph.
    # Otherwise, we might need to compute it from `pos`.
    # If the graph was built with kNN, edge lengths *are* kNN distances (mostly).
    # If not, let's skip explicit kNN re-computation for speed unless requested.
    # For now, we reuse edge length stats as a proxy for spatial dispersion if it's a spatial graph.
    
    return stats
