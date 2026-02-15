import os
import logging
import time
from pathlib import Path
from typing import Optional, Dict, Any
import numpy as np
import pandas as pd
import torch

# Package imports
from .preprocessing.tissue_mask import get_tissue_mask
from .nuclei.segmentation import segment_nuclei

from .preprocessing.graph_builder_pyg import build_nuclei_graph, save_nuclei_graph
# Per-node feature extractors (aligned with original repo)
from .features.node_features import extract_handcrafted_node_features
from .features.deep_features import DeepNodeFeatureExtractor
# Graph-level feature extractors (optional, local extensions)
from .features.gnn import extract_gnn_embeddings, GraphEncoder
from .features.handcrafted import extract_graph_stats
from .utils.reproducibility import set_seeds, capture_environment, save_metadata


def setup_logger(log_file: str) -> logging.Logger:
    """Sets up a logger that writes to a file and console."""
    logger = logging.getLogger(log_file)
    logger.setLevel(logging.INFO)

    if logger.hasHandlers():
        logger.handlers.clear()

    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    logger.propagate = False
    return logger


def run_pipeline(
    slide_path: str,
    output_dir: str,
    model_path: str,
    config: Optional[Dict[str, Any]] = None,
    force_rerun: bool = False,
):
    """
    Run the full pipeline: Tissue Mask → Nuclei Segmentation → Graph → Features → QC.

    Args:
        slide_path: Path to the input WSI.
        output_dir: Base directory for output. A subdir per slide is created.
        model_path: Path to the HoVer-Net model checkpoint (full model object, state_dict, or legacy 'desc' format).
        config: Configuration dictionary.
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

    if config.get('reproducibility', {}).get('save_metadata', True):
        save_metadata(str(slide_out_dir), config)

    log_file = slide_out_dir / "pipeline.log"
    logger = setup_logger(str(log_file))

    logger.info(f"Starting pipeline for slide: {slide_id}")
    logger.info(f"Output directory: {slide_out_dir}")

    # ==================================================================
    # Step 1 & 2: Tissue Mask + Nuclei Segmentation
    # ==================================================================
    nuclei_out_path = slide_out_dir / "nuclei.parquet"
    instance_map_path = slide_out_dir / "instance_map.npy"

    instance_map = None
    nuclei_df = None

    if force_rerun or not nuclei_out_path.exists() or not instance_map_path.exists():
        logger.info("Step 1: Generating Tissue Mask...")
        try:
            import openslide
            slide = openslide.OpenSlide(slide_path)
            w, h = slide.dimensions
            downsample = max(w // 2048, 1)
            thumb = slide.get_thumbnail((w // downsample, h // downsample))
            thumb_np = np.array(thumb)
            _, tissue_mask = get_tissue_mask(thumb_np)
            if tissue_mask is not None:
                tissue_mask = np.asarray(tissue_mask).astype(bool)
            logger.info("Tissue mask generated.")
        except Exception as e:
            logger.error(f"Failed to generate tissue mask: {e}")
            raise

        logger.info("Step 2: Segmenting Nuclei...")
        try:
            seg_config = config.get('segmentation', {})
            stitch_mode = seg_config.get('stitch_mode', 'global')

            instance_map, nuclei_df = segment_nuclei(
                slide_path=slide_path,
                level=seg_config.get('level', 0),
                tile_size=seg_config.get('tile_size', 256),
                overlap=seg_config.get('overlap', 0),
                tissue_mask=tissue_mask,
                model_path=model_path,
                device=seg_config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'),
                batch_size=seg_config.get('batch_size', 16),
                min_nucleus_area=seg_config.get('min_nucleus_area', 10),
                stitch_mode=stitch_mode,
            )

            nuclei_df.to_parquet(nuclei_out_path)
            np.save(instance_map_path, instance_map)
            logger.info(
                f"Nuclei segmentation done. Count: {len(nuclei_df)}. "
                f"Instance map: {instance_map.shape}, stitch_mode={stitch_mode}"
            )
        except Exception as e:
            logger.error(f"Nuclei segmentation failed: {e}")
            raise
    else:
        logger.info("Steps 1-2 cached. Loading nuclei table and instance map.")
        nuclei_df = pd.read_parquet(nuclei_out_path)
        instance_map = np.load(instance_map_path)

    # ==================================================================
    # Step 3: Graph Construction
    # ==================================================================
    graph_out_path = slide_out_dir / "graph.pt"

    graph_data = None
    if force_rerun or not graph_out_path.exists():
        logger.info("Step 3: Building Graph...")
        try:
            graph_config = config.get('graph', {})
            graph_method = graph_config.get('method', 'knn')
            graph_params = {}
            if graph_method == 'knn':
                graph_params['k'] = graph_config.get('k', 5)
            elif graph_method == 'radius':
                graph_params['r'] = graph_config.get('r', 50.0)

            graph_params['max_edge_length'] = graph_config.get('max_edge_length', None)
            graph_params['remove_isolated_nodes'] = graph_config.get('remove_isolated_nodes', False)
            graph_params['coord_space'] = graph_config.get('coord_space', 'level-0-pixels')

            graph_data = build_nuclei_graph(
                nuclei_df,
                method=graph_method,
                params=graph_params,
            )

            save_nuclei_graph(graph_data, str(graph_out_path))
            logger.info(
                f"Graph built. Nodes: {graph_data.num_nodes}, Edges: {graph_data.num_edges}"
            )
        except Exception as e:
            logger.error(f"Graph construction failed: {e}")
            raise
    else:
        logger.info("Step 3: Graph cached. Skipping.")

    # ==================================================================
    # Step 4: Feature Extraction
    # ==================================================================
    feat_config = config.get('features', {})
    feat_mode = feat_config.get('mode', 'handcrafted')
    feat_out_path = slide_out_dir / "features.pt"

    if force_rerun or not feat_out_path.exists():
        logger.info(f"Step 4: Extracting Features (mode={feat_mode})...")
        try:
            # Ensure graph is loaded
            if graph_data is None:
                graph_data = torch.load(graph_out_path, weights_only=False)

            if feat_mode == 'handcrafted':
                # Per-node: 24 handcrafted features (faithful to original)
                # Need the slide image at the same resolution as the instance map
                import openslide
                slide = openslide.OpenSlide(slide_path)
                seg_level = config.get('segmentation', {}).get('level', 0)
                w, h = slide.level_dimensions[seg_level]
                # Read the full image at the segmentation level
                full_image = np.array(
                    slide.read_region((0, 0), seg_level, (w, h)).convert("RGB")
                )
                slide.close()

                node_features = extract_handcrafted_node_features(
                    full_image, instance_map
                )
                graph_data.x = node_features
                # Re-save graph with updated features
                save_nuclei_graph(graph_data, str(graph_out_path))
                torch.save(node_features, feat_out_path)
                logger.info(f"Handcrafted features: shape {node_features.shape}")

            elif feat_mode == 'deep':
                # Per-node: CNN embeddings (faithful to original)
                import openslide
                slide = openslide.OpenSlide(slide_path)
                seg_level = config.get('segmentation', {}).get('level', 0)
                w, h = slide.level_dimensions[seg_level]
                full_image = np.array(
                    slide.read_region((0, 0), seg_level, (w, h)).convert("RGB")
                )
                slide.close()

                extractor = DeepNodeFeatureExtractor(
                    architecture=feat_config.get('architecture', 'resnet50'),
                    patch_size=feat_config.get('patch_size', 72),
                    resize_size=feat_config.get('resize_size', None),
                    stride=feat_config.get('stride', None),
                    downsample_factor=feat_config.get('downsample_factor', 1),
                    batch_size=feat_config.get('batch_size', 32),
                    with_instance_masking=feat_config.get('with_instance_masking', False),
                    verbose=True,
                )
                node_features = extractor.extract(full_image, instance_map)
                graph_data.x = node_features
                save_nuclei_graph(graph_data, str(graph_out_path))
                torch.save(node_features, feat_out_path)
                logger.info(f"Deep features: shape {node_features.shape}")

            elif feat_mode == 'graph_stats':
                # Per-graph: graph-level statistics (local extension)
                stats = extract_graph_stats(graph_data)
                np.save(str(feat_out_path).replace('.pt', '.npy'), stats)
                logger.info(f"Graph stats saved ({len(stats)} metrics).")

            elif feat_mode == 'gnn':
                # Per-graph: GNN embedding (local extension)
                gnn_model_path = feat_config.get('gnn_model_path', None)
                model = None
                if gnn_model_path:
                    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                    model = torch.load(gnn_model_path, map_location=device)
                    if not isinstance(model, torch.nn.Module):
                        raise TypeError("GNN model must be a full model object")
                    model.eval()

                emb = extract_gnn_embeddings(graph_data, model)
                if isinstance(emb, torch.Tensor):
                    emb = emb.detach().cpu().numpy()
                np.save(str(feat_out_path).replace('.pt', '.npy'), emb)
                logger.info(f"GNN embedding: shape {emb.shape}")

            else:
                raise ValueError(f"Unknown feature mode: {feat_mode!r}")

            logger.info(f"Features saved to {feat_out_path}")

        except Exception as e:
            logger.error(f"Feature extraction failed: {e}")
            raise
    else:
        logger.info("Step 4: Features cached. Skipping.")

    # ==================================================================
    # Step 5: QC & Visualization
    # ==================================================================
    qc_out_dir = slide_out_dir / "qc"
    qc_out_dir.mkdir(exist_ok=True)

    qc_thumb_path = qc_out_dir / "qc_thumbnail.png"
    qc_metrics_path = qc_out_dir / "qc_metrics.json"

    if force_rerun or not qc_thumb_path.exists() or not qc_metrics_path.exists():
        logger.info("Step 5: Running QC...")
        try:
            if nuclei_df is None:
                nuclei_df = pd.read_parquet(nuclei_out_path)
            if graph_data is None:
                graph_data = torch.load(graph_out_path, weights_only=False)

            import openslide
            slide = openslide.OpenSlide(slide_path)
            w, h = slide.dimensions
            downsample_target = max(w // 2048, 1)
            thumb_size = (w // downsample_target, h // downsample_target)
            thumb = slide.get_thumbnail(thumb_size)
            real_downsample = w / thumb.size[0]

            from .visualization.qc import (
                generate_qc_thumbnail,
                compute_qc_metrics,
                save_qc_metrics,
            )

            generate_qc_thumbnail(
                thumb_image=thumb,
                nuclei_df=nuclei_df,
                graph_data=graph_data,
                output_path=str(qc_thumb_path),
                downsample_factor=real_downsample,
            )

            slide_info = {
                'width': w,
                'height': h,
                'mpp': float(slide.properties.get('openslide.mpp-x', 0.5)),
            }
            metrics = compute_qc_metrics(nuclei_df, graph_data, slide_info)
            save_qc_metrics(metrics, str(qc_metrics_path))
            slide.close()

            logger.info(f"QC completed. Saved to {qc_out_dir}")

        except Exception as e:
            logger.error(f"QC step failed: {e}")
            logger.warning("Continuing despite QC failure.")
    else:
        logger.info("Step 5: QC cached. Skipping.")

    logger.info("Pipeline completed successfully.")
    return str(slide_out_dir)
