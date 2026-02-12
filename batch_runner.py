import os
import argparse
import pandas as pd
import sys
import logging
from histocartography_ext.pipeline_runner import run_pipeline

def main():
    parser = argparse.ArgumentParser(description="Batch runner for histology pipeline.")
    parser.add_argument("--manifest", type=str, required=True, help="CSV file with list of slides.")
    parser.add_argument("--out_dir", type=str, required=True, help="Output directory.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to Nuclei Segmentation Model Checkpoint.")
    parser.add_argument("--gnn_model_path", type=str, default=None, help="Path to GNN Model Checkpoint (optional).")
    parser.add_argument("--slide_col", type=str, default="slide_path", help="Column name for slide path in manifest.")
    parser.add_argument("--skip_errors", action="store_true", help="Continue processing even if a slide fails.")
    parser.add_argument("--slurm_array_idx", type=int, default=None, help="SLURM array index (0-based). Overrides SLURM_ARRAY_TASK_ID.")
    parser.add_argument("--force_rerun", action="store_true", help="Force rerun of all steps.")
    
    # Pipeline configs
    parser.add_argument("--graph_method", type=str, default="knn", choices=["knn", "radius"], help="Graph construction method.")
    parser.add_argument("--k", type=int, default=5, help="k for kNN.")
    parser.add_argument("--r", type=float, default=50.0, help="r for radius graph.")
    parser.add_argument("--feat_mode", type=str, default="gnn", choices=["stats", "gnn"], help="Feature extraction mode.")
    
    args = parser.parse_args()
    
    # Setup global logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger("BatchRunner")
    
    # Read Manifest
    try:
        df = pd.read_csv(args.manifest)
    except Exception as e:
        logger.error(f"Failed to read manifest: {e}")
        sys.exit(1)
        
    if args.slide_col not in df.columns:
        logger.error(f"Column '{args.slide_col}' not found in manifest. Available: {list(df.columns)}")
        sys.exit(1)
        
    slides = df[args.slide_col].tolist()
    
    # Handle SLURM Array
    task_id = args.slurm_array_idx
    if task_id is None:
        env_task_id = os.environ.get("SLURM_ARRAY_TASK_ID")
        if env_task_id is not None:
            task_id = int(env_task_id)
            
    if task_id is not None:
        if task_id < 0 or task_id >= len(slides):
            logger.error(f"Invalid task ID {task_id} for manifest with {len(slides)} slides.")
            sys.exit(1)
        
        logger.info(f"Running in SLURM Array Mode. Task ID: {task_id}")
        slides_to_process = [slides[task_id]]
    else:
        logger.info(f"Running in Serial Mode. Processing {len(slides)} slides.")
        slides_to_process = slides
        
    # Process
    success_count = 0
    fail_count = 0
    
    config = {
        'graph_method': args.graph_method,
        'k': args.k,
        'r': args.r,
        'feat_mode': args.feat_mode,
        'gnn_model_path': args.gnn_model_path,
        # Add other potential configs here
    }
    
    for slide_path in slides_to_process:
        try:
            run_pipeline(
                slide_path=slide_path,
                output_dir=args.out_dir,
                model_path=args.model_path,
                config=config,
                force_rerun=args.force_rerun
            )
            success_count += 1
        except Exception as e:
            logger.error(f"Failed to process {slide_path}: {e}")
            fail_count += 1
            if not args.skip_errors:
                sys.exit(1)
                
    logger.info(f"Batch processing complete. Success: {success_count}, Failed: {fail_count}")

if __name__ == "__main__":
    main()
