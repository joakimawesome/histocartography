import os
import logging
import json
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
from torch_geometric.data import Data

def generate_qc_thumbnail(
    thumb_image: Image.Image,
    nuclei_df: pd.DataFrame,
    graph_data: Data,
    output_path: str,
    downsample_factor: float = 1.0
):
    """
    Draws nuclei centroids and graph edges on a downsampled thumbnail.
    
    Args:
        thumb_image: PIL Image of the slide thumbnail.
        nuclei_df: DataFrame with nuclei info (needs centroid_x, centroid_y).
        graph_data: PyG Data object with edge_index.
        output_path: Path to save the QC image.
        downsample_factor: Factor by which coordinates in nuclei_df/graph need to be divided 
                           to match the thumbnail size. 
                           (e.g. if slide is 100k wide and thumb is 2k, factor is 50).
    """
    # Create a copy to draw on
    canvas = thumb_image.copy().convert("RGB")
    draw = ImageDraw.Draw(canvas)
    
    # 1. Draw Edges
    # Edge indices are typically based on the order of nodes in graph_data.x or nuclei_df
    # We assume nuclei_df matches the graph nodes order if graph was built from it.
    
    if graph_data is not None and graph_data.edge_index is not None:
        edge_index = graph_data.edge_index.cpu().numpy()
        # Get coordinates
        # Assumes nuclei_df index corresponds to node index
        coords = nuclei_df[['centroid_x', 'centroid_y']].values
        
        # Scale coordinates
        scaled_coords = coords / downsample_factor
        
        # Draw a subset of edges if too many? For thumbnail, we might want to see density.
        # But drawing millions of edges on a 1000x1000 image is messy.
        # Let's draw them semi-transparently or just a sample if needed.
        # For now, draw all but maybe with high transparency.
        # PIL doesn't support alpha separation for lines easily without a separate layer.
        
        # Create an overlay layer for edges
        edge_layer = Image.new("RGBA", canvas.size, (255, 255, 255, 0))
        edge_draw = ImageDraw.Draw(edge_layer)
        
        # Vectorized segment drawing isn't easy in PIL, loop is slow.
        # If graph is huge, this loop will be slow.
        # Optimization: only draw edges that are long enough to be visible? 
        # Or use matplotlib to save an image? Matplotlib is often slower for many lines.
        # Let's try direct PIL drawing. limits: 50k edges is fine, 1M is slow.
        # If edges > 100k, maybe subsample?
        
        num_edges = edge_index.shape[1]
        step = 1
        if num_edges > 50000:
            step = num_edges // 50000 # Cap at ~50k edges for visualization speed
            
        for i in range(0, num_edges, step):
            idx_src = edge_index[0, i]
            idx_dst = edge_index[1, i]
            
            x1, y1 = scaled_coords[idx_src]
            x2, y2 = scaled_coords[idx_dst]
            
            edge_draw.line([(x1, y1), (x2, y2)], fill=(0, 255, 255, 100), width=1)
            
        canvas = Image.alpha_composite(canvas.convert("RGBA"), edge_layer).convert("RGB")
        draw = ImageDraw.Draw(canvas) # Re-get draw for RGB

    # 2. Draw Nuclei Centroids
    # Draw as small dots
    if not nuclei_df.empty:
        # Scale
        # We can likely iterate faster or use point drawing
        # For >100k nuclei, drawing individual circles is slow. 
        # `point` is faster.
        
        coords = nuclei_df[['centroid_x', 'centroid_y']].values
        scaled_coords = coords / downsample_factor
        
        # Filter those that are within image bounds (should be all, but safe check)
        w, h = canvas.size
        valid_mask = (scaled_coords[:, 0] >= 0) & (scaled_coords[:, 0] < w) & \
                     (scaled_coords[:, 1] >= 0) & (scaled_coords[:, 1] < h)
        
        valid_coords = scaled_coords[valid_mask]
        
        # Draw points
        # PIL point takes list of tuples
        points = list(map(tuple, valid_coords))
        draw.point(points, fill=(255, 255, 0)) # Yellow dots

    canvas.save(output_path)


def compute_qc_metrics(
    nuclei_df: pd.DataFrame,
    graph_data: Data,
    slide_level_info: dict = None
) -> dict:
    """
    Computes quantitative QC metrics.
    
    Args:
        nuclei_df: DataFrame of nuclei.
        graph_data: PyG Data object.
        slide_level_info: Dict with 'mpp', 'width', 'height', 'tissue_area_px', etc.
    
    Returns:
        dict of metrics.
    """
    metrics = {}
    
    # Nuclei Metrics
    num_nuclei = len(nuclei_df)
    metrics['nuclei_count'] = num_nuclei
    
    if slide_level_info:
        # Density (nuclei / mm^2)
        # Need MPP to convert pixels to mm
        mpp = slide_level_info.get('mpp', 0.5) # Default assumption if missing?
        if mpp:
            # area in microns^2 = pixels * mpp^2
            # area in mm^2 = microns^2 / 1e6
            
            # If we utilize tissue area:
            features_area_px = slide_level_info.get('tissue_area_px', None)
            if features_area_px:
                area_mm2 = (features_area_px * (mpp ** 2)) / 1e6
                metrics['tissue_area_mm2'] = area_mm2
                metrics['nuclei_density_per_mm2'] = num_nuclei / area_mm2 if area_mm2 > 0 else 0
            
    # Graph Metrics
    if graph_data:
        num_nodes = graph_data.num_nodes
        num_edges = graph_data.num_edges
        
        metrics['graph_num_nodes'] = num_nodes
        metrics['graph_num_edges'] = num_edges
        
        if num_nodes > 0:
            metrics['graph_avg_degree'] = num_edges / num_nodes # Directed? PyG usually is (2 edges per connection)
            # If PyG edge_index is undirected (contains (i,j) and (j,i)), degree is edges/nodes.
            # Usually we treat it as directed edge list sum.
        
        # Connected components (requires scipy or networkx, might be heavy for QC but useful)
        # components = connected_components(to_scipy_sparse_matrix(graph_data.edge_index))
        # metrics['num_connected_components'] = components[0]
        
    return metrics

def save_qc_metrics(metrics: dict, output_path: str):
    with open(output_path, 'w') as f:
        json.dump(metrics, f, indent=4)

def plot_qc_distributions(graph_data: Data, nuclei_df: pd.DataFrame, output_path: str):
    """
    Plots distributions of edge lengths and node degrees.
    """
    if graph_data is None:
        return

    edge_index = graph_data.edge_index
    num_nodes = graph_data.num_nodes
    
    # 1. Degree Distribution
    # Calculate degrees
    # degree = index count
    # use torch_geometric.utils.degree if available, or simpler bincount
    import torch_geometric.utils as pyg_utils
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Degree
    d = pyg_utils.degree(edge_index[0], num_nodes=num_nodes).cpu().numpy()
    axes[0].hist(d, bins=30, color='skyblue', edgecolor='black')
    axes[0].set_title("Node Degree Distribution")
    axes[0].set_xlabel("Degree")
    axes[0].set_ylabel("Count")
    axes[0].set_yscale('log')
    
    # 2. Edge Length Distribution (if pos provided or inferred from nuclei_df)
    # If graph_data has pos
    if hasattr(graph_data, 'pos') and graph_data.pos is not None:
        pos = graph_data.pos
    elif not nuclei_df.empty:
        pos = torch.tensor(nuclei_df[['centroid_x', 'centroid_y']].values, dtype=torch.float)
    else:
        pos = None
        
    if pos is not None:
        src, dst = edge_index
        # Calculate euclidean distance
        # Filter if on GPU
        if pos.device != src.device:
            pos = pos.to(src.device)
            
        dist = (pos[src] - pos[dst]).norm(dim=1).cpu().numpy()
        
        axes[1].hist(dist, bins=30, color='salmon', edgecolor='black')
        axes[1].set_title("Edge Length Distribution (px)")
        axes[1].set_xlabel("Length (pixels)")
        axes[1].set_ylabel("Count")
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close(fig)
