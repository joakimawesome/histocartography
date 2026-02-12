import os
import logging
import argparse
import time
from pathlib import Path
from typing import Optional, Dict, Any
import numpy as np
import pandas as pd
import torch

# Imports from within the package
from .preprocessing.tissue_mask import get_tissue_mask
# We assume segment_nuclei is available. If not, we might need to adjust imports.
try:
    from .nuclei.segmentation import segment_nuclei
except ImportError:
    # Fallback or placeholder if dependencies are missing during dev
    def segment_nuclei(*args, **kwargs) -> pd.DataFrame:
        raise NotImplementedError("segment_nuclei not imported")

from .preprocessing.graph_builder_pyg import build_nuclei_graph, save_nuclei_graph
# Feature extraction imports
from .features.gnn import extract_gnn_embeddings, GraphEncoder
from .features.handcrafted import extract_graph_stats
from .features.extract import load_graph # Reuse loading logic if available, or just torch.load
from .utils.reproducibility import set_seeds, capture_environment, save_metadata

def setup_logger(log_file: str) -> logging.Logger:
    """Sets up a logger that writes to a file and console."""
    logger = logging.getLogger(log_file)
    logger.setLevel(logging.INFO)
    
    # Clear existing handlers to avoid duplicates if re-running in same process
    if logger.hasHandlers():
        logger.handlers.clear()

    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    # File handler
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    return logger

def run_pipeline(
    slide_path: str,
    output_dir: str,
    model_path: str,
    config: Optional[Dict[str, Any]] = None,
    force_rerun: bool = False
):
    """
    Runs the full pipeline: Tissue Mask -> Nuclei Segmentation -> Graph -> Features.
    
    Args:
        slide_path: Path to the input WSI.
        output_dir: Base directory for output. A subdir for the slide will be created.
        model_path: Path to the HoVer-Net model checkpoint.
        config: Dictionary of configuration parameters.
        force_rerun: If True, ignore cache and rerun all steps.
    """
    if config is None:
        config = {}

    # --- Reproducibility ---
    seed = config.get('reproducibility', {}).get('seed', 42)
    set_seeds(seed)
    
    slide_id = Path(slide_path).stem
    slide_out_dir = Path(output_dir) / slide_id
    slide_out_dir.mkdir(parents=True, exist_ok=True)
    
    # Save metadata (config + environment)
    if config.get('reproducibility', {}).get('save_metadata', True):
        save_metadata(slide_out_dir, config)
    
    log_file = slide_out_dir / "pipeline.log"
    logger = setup_logger(str(log_file))
    
    logger.info(f"Starting pipeline for slide: {slide_id}")
    logger.info(f"Output directory: {slide_out_dir}")
    
    # --- Step 1: Tissue Mask ---
    mask_out_path = slide_out_dir / "tissue_mask.png" # Or .npy if we want raw
    tissue_mask = None
    
    # For now, we don't save the tissue mask as a file to check against for skipping,
    # because it's fast and needed in memory for segmentation. 
    # But if we wanted to cache it, we could save/load it.
    # The user request says "if nuclei_table.parquet exists, reuse".
    # Tissue mask is a prerequisite for nuclei segmentation.
    
    # Let's check if we can skip segmentation wholly.
    nuclei_out_path = slide_out_dir / "nuclei.parquet"
    
    if start_step_segmentation := (force_rerun or not nuclei_out_path.exists()):
        logger.info("Step 1: Generating Tissue Mask...")
        try:
            import openslide
            slide = openslide.OpenSlide(slide_path)
            # Get thumbnail for mask
            # Adjust downsample as needed
            w, h = slide.dimensions
            downsample = max(w // 2048, 1) # Target ~2k width
            thumb = slide.get_thumbnail((w // downsample, h // downsample))
            thumb_np = np.array(thumb)
            _, tissue_mask = get_tissue_mask(thumb_np)
            if tissue_mask is not None:
                tissue_mask = np.asarray(tissue_mask).astype(bool)
            logger.info("Tissue mask generated.")
        except Exception as e:
            logger.error(f"Failed to generate tissue mask: {e}")
            raise e
    else:
        logger.info("Step 1 & 2: Nuclei table exists. Skipping tissue mask and segmentation.")

    # --- Step 2: Nuclei Segmentation ---
    if start_step_segmentation:
        logger.info("Step 2: Segmenting Nuclei...")
        try:
            # We assume the segmentation function handles the full WSI logic
            # And accepts the thumbnail-sized tissue mask (it might need to scale it up)
            # Based on inspection, segment_nuclei does scaling if provided.
            
            # Note: segment_nuclei needs to be robust for the memory usage.
            nuclei_df = segment_nuclei(
                slide_path=slide_path,
                tissue_mask=tissue_mask, # It handles scaling
                model_path=model_path,
                device=config.get('segmentation', {}).get('device', 'cuda' if torch.cuda.is_available() else 'cpu'),
                batch_size=config.get('segmentation', {}).get('batch_size', 16),
                min_nucleus_area=config.get('segmentation', {}).get('min_nucleus_area', 10)
            )
            
            # Save nuclei table
            nuclei_df.to_parquet(nuclei_out_path)
            logger.info(f"Nuclei segmentation done. Saved to {nuclei_out_path}. Count: {len(nuclei_df)}")
        except Exception as e:
            logger.error(f"Nuclei segmentation failed: {e}")
            raise e
    else:
        # Load existing for next steps
        logger.info(f"Loading existing nuclei table from {nuclei_out_path}")
        nuclei_df = pd.read_parquet(nuclei_out_path)

    # --- Step 3: Graph Construction ---
    graph_out_path = slide_out_dir / "graph.pt"
    
    if force_rerun or not graph_out_path.exists():
        logger.info("Step 3: Building Graph...")
        try:
            # Param config
            graph_config = config.get('graph', {})
            graph_method = graph_config.get('method', 'knn')
            graph_params = {}
            if graph_method == 'knn':
                graph_params['k'] = graph_config.get('k', 5)
            elif graph_method == 'radius':
                graph_params['r'] = graph_config.get('r', 50.0)
            
            graph_params['max_edge_length'] = graph_config.get('max_edge_length', None) # Optional pruning
            graph_params['remove_isolated_nodes'] = graph_config.get('remove_isolated_nodes', False)
            graph_params['coord_space'] = graph_config.get('coord_space', 'level-0-pixels')
            
            # Build
            graph_data = build_nuclei_graph(
                nuclei_df,
                method=graph_method,
                params=graph_params
            )
            
            # Save
            save_nuclei_graph(graph_data, str(graph_out_path))
            logger.info(f"Graph built and saved to {graph_out_path}. Nodes: {graph_data.num_nodes}, Edges: {graph_data.num_edges}")
        except Exception as e:
            logger.error(f"Graph construction failed: {e}")
            raise e
    else:
        logger.info(f"Step 3: checking graph... {graph_out_path} exists. Skipping.")
        # Load graph for feature extraction if needed
        # We assume if graph exists, we might still need to load it for features if features don't exist
    
    # --- Step 4: Feature Extraction ---
    # Determine feature output name based on config (stats or gnn)
    feat_config = config.get('features', {})
    feat_mode = feat_config.get('mode', 'gnn')
    feat_out_name = "features.npy" # User requested this specific name
    feat_out_path = slide_out_dir / feat_out_name
    
    if force_rerun or not feat_out_path.exists():
        logger.info(f"Step 4: Extracting Features ({feat_mode})...")
        try:
            # We need the graph now
            if not 'graph_data' in locals():
                logger.info(f"Loading graph from {graph_out_path}")
                graph_data = torch.load(graph_out_path)
                
            features = None
            if feat_mode == 'stats':
                stats = extract_graph_stats(graph_data)
                # Convert dict to simple array or keep as dict?
                # User asked for features.npy. 
                # If it's statistics, it's usually a 1D vector per slide.
                # Let's save as structured array or just values.
                # For compatibility with downstream, maybe a dictionary in npy?
                features = stats
                np.save(feat_out_path, features)
                
            elif feat_mode == 'gnn':
                # Load GNN model if provided
                gnn_model_path = feat_config.get('gnn_model_path', None)
                model = None
                if gnn_model_path:
                    # Load model logic
                    loaded = torch.load(gnn_model_path)
                    if isinstance(loaded, torch.nn.Module):
                        model = loaded
                    elif isinstance(loaded, dict):
                         # Assume default arch if state dict
                        input_dim = graph_data.x.shape[1]
                        model = GraphEncoder(input_dim=input_dim, hidden_dim=64, output_dim=128)
                        model.load_state_dict(loaded)
                    model.eval()
                else:
                    # Random init
                    input_dim = graph_data.x.shape[1]
                    model = GraphEncoder(input_dim=input_dim, hidden_dim=64, output_dim=128)
                    model.eval()
                    
                emb = extract_gnn_embeddings(graph_data, model)
                if isinstance(emb, torch.Tensor):
                    emb = emb.detach().cpu().numpy()
                
                features = emb
                np.save(feat_out_path, features)
            
            logger.info(f"Features extracted and saved to {feat_out_path}")

        except Exception as e:
            logger.error(f"Feature extraction failed: {e}")
            raise e
    else:
         logger.info(f"Step 4: checking features... {feat_out_path} exists. Skipping.")

    # --- Step 5: QC & Visualization ---
    qc_out_dir = slide_out_dir / "qc"
    qc_out_dir.mkdir(exist_ok=True)
    
    qc_thumb_path = qc_out_dir / "qc_thumbnail.png"
    qc_metrics_path = qc_out_dir / "qc_metrics.json"

    if force_rerun or not qc_thumb_path.exists() or not qc_metrics_path.exists():
        logger.info("Step 5: Running QC...")
        try:
            # We need:
            # 1. Slide thumbnail (re-read if needed)
            # 2. Nuclei DF (loaded)
            # 3. Graph Data (loaded)
            
            # Ensure prerequisites are loaded
            if 'nuclei_df' not in locals():
                logger.info(f"Loading nuclei for QC from {nuclei_out_path}")
                nuclei_df = pd.read_parquet(nuclei_out_path)
            
            if 'graph_data' not in locals():
                 logger.info(f"Loading graph for QC from {graph_out_path}")
                 graph_data = torch.load(graph_out_path)

            # Get thumbnail
            import openslide
            slide = openslide.OpenSlide(slide_path)
            w, h = slide.dimensions
            # Target ~2048 width for visibility
            downsample_target = max(w // 2048, 1)
            thumb_size = (w // downsample_target, h // downsample_target)
            thumb = slide.get_thumbnail(thumb_size)
            
            # Calculate actual downsample factor
            # thumb.size is (width, height)
            real_downsample = w / thumb.size[0]
            
            # Import QC functions
            from .visualization.qc import (
                generate_qc_thumbnail, 
                compute_qc_metrics, 
                save_qc_metrics,
                plot_qc_distributions
            )
            
            # 1. QC Thumbnail
            logger.info("Generating QC thumbnail...")
            generate_qc_thumbnail(
                thumb_image=thumb,
                nuclei_df=nuclei_df,
                graph_data=graph_data,
                output_path=str(qc_thumb_path),
                downsample_factor=real_downsample
            )
            
            # 2. QC Metrics
            logger.info("Computing QC metrics...")
            slide_info = {
                'width': w, 
                'height': h, 
                'mpp': float(slide.properties.get('openslide.mpp-x', 0.5)) # Default to 0.5 if missing
            }
            metrics = compute_qc_metrics(nuclei_df, graph_data, slide_info)
            save_qc_metrics(metrics, str(qc_metrics_path))
            
            # 3. QC Distributions
            # Plot graph statistics
            # Optional but good for MICCAI
            # We can save this to qc_out_dir
            dist_plot_path = qc_out_dir / "qc_distributions.png"
            # plot_qc_distributions(graph_data, nuclei_df, str(dist_plot_path)) # Optional
            
            logger.info(f"QC completed. Saved to {qc_out_dir}")

        except Exception as e:
            logger.error(f"QC step failed: {e}")
            # Don't raise, as pipeline core finished. Just log error.
            # Or raise if strict.
            logger.warning("Continuing despite QC failure.")
    else:
        logger.info("Step 5: QC already exists. Skipping.")

    logger.info("Pipeline completed successfully.")
    return str(slide_out_dir)

