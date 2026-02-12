import os
import argparse
import pandas as pd
import sys
import logging
from pathlib import Path
from typing import Optional



def _normalize_manifest_path(p: str) -> str:
    """Normalize manifest-provided paths across OSes.

    The project manifest may contain Windows-style separators (e.g. "data\\wsi_raw\\WSI_000032.svs").
    On Linux this will be treated as a literal filename (backslash is not a separator),
    which makes OpenSlide fail with "Unsupported or missing image file".
    """
    p = (p or "").strip()
    if os.name != "nt" and "\\" in p:
        p = p.replace("\\", "/")
    return p


def _guess_project_root_from_manifest(manifest_path: str) -> Optional[Path]:
    """Best-effort guess of the dataset/project root from a manifest location.

    For example:
      <root>/data/manifests/<version>/manifest.csv  -> returns <root>
    """
    try:
        mp = Path(manifest_path).resolve()
    except Exception:
        return None

    for parent in mp.parents:
        data_dir = parent / "data"
        manifests_dir = data_dir / "manifests"
        if data_dir.is_dir() and manifests_dir.is_dir():
            return parent
    return None


def _sanitize_run_name(name: str) -> str:
    # Keep it filesystem-friendly across platforms.
    name = (name or "").strip()
    if not name:
        return "run"
    allowed = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_.=")
    return "".join(c if c in allowed else "_" for c in name)


def _infer_manifest_tag(manifest_path: str) -> Optional[str]:
    """Extract a short informative tag from manifest path (e.g., version folder)."""
    try:
        mp = Path(manifest_path).resolve()
    except Exception:
        return None

    parts = [p.name for p in mp.parents]
    # Common layout: <root>/data/manifests/<tag>/manifest.csv
    # Grab the folder right above manifest.csv if it isn't a generic name.
    if len(parts) >= 1:
        tag = mp.parent.name
        if tag and tag.lower() not in {"manifests", "data"}:
            return tag
    return None


def _default_run_name(args: argparse.Namespace) -> str:
    tag = _infer_manifest_tag(args.manifest)
    parts = []
    if tag:
        parts.append(tag)

    parts.append(f"graph-{args.graph_method}")
    if args.graph_method == "knn":
        parts.append(f"k{args.k}")
    else:
        # radius
        parts.append(f"r{args.r}")

    parts.append(f"feat-{args.feat_mode}")

    return _sanitize_run_name("__".join(parts))


def _resolve_slide_path(slide_path: str, manifest_path: str, slides_root: Optional[str]) -> str:
    """Resolve slide_path to an existing path when possible."""
    slide_path = _normalize_manifest_path(slide_path)

    # Already absolute?
    try:
        sp = Path(slide_path)
        if sp.is_absolute():
            return str(sp)
        if sp.exists():
            return str(sp)
    except Exception:
        # If pathlib can't parse, just return as-is.
        return slide_path

    # Resolve relative to explicit slides_root if provided.
    if slides_root:
        candidate = Path(slides_root) / slide_path
        if candidate.exists():
            return str(candidate)

    # Best-effort: infer root from manifest location.
    root = _guess_project_root_from_manifest(manifest_path)
    if root is not None:
        candidate = root / slide_path
        if candidate.exists():
            return str(candidate)

    # Give up; caller will surface the error from downstream.
    return slide_path


def main():
    parser = argparse.ArgumentParser(description="Batch runner for histology pipeline.")
    parser.add_argument("--manifest", type=str, required=True, help="CSV file with list of slides.")
    parser.add_argument("--out_dir", type=str, required=True, help="Output directory.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to Nuclei Segmentation Model Checkpoint.")
    parser.add_argument("--gnn_model_path", type=str, default=None, help="Path to GNN Model Checkpoint (optional).")
    parser.add_argument("--slide_col", type=str, default="slide_path", help="Column name for slide path in manifest.")
    parser.add_argument("--slides_root", type=str, default=None, help="Optional root directory to resolve relative slide paths.")
    parser.add_argument(
        "--run_name",
        type=str,
        default=None,
        help=(
            "Optional subdirectory name under --out_dir to keep outputs grouped (e.g. v1.0.1__graph-knn__k5__feat-stats). "
            "If omitted, a name is derived from manifest location + graph/feature args."
        ),
    )
    parser.add_argument("--skip_errors", action="store_true", help="Continue processing even if a slide fails.")
    parser.add_argument("--slurm_array_idx", type=int, default=None, help="SLURM array index (0-based). Overrides SLURM_ARRAY_TASK_ID.")
    parser.add_argument("--force_rerun", action="store_true", help="Force rerun of all steps.")
    
    # Pipeline configs
    parser.add_argument("--graph_method", type=str, default="knn", choices=["knn", "radius"], help="Graph construction method.")
    parser.add_argument("--k", type=int, default=5, help="k for kNN.")
    parser.add_argument("--r", type=float, default=50.0, help="r for radius graph.")
    parser.add_argument("--feat_mode", type=str, default="gnn", choices=["stats", "gnn"], help="Feature extraction mode.")
    
    args = parser.parse_args()
    
    # Import here so `--help` (and other argparse errors) doesn't require full pipeline deps.
    from histocartography_ext.pipeline_runner import run_pipeline
    
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
        
    slides_raw = df[args.slide_col].tolist()
    slides = [
        _resolve_slide_path(p, manifest_path=args.manifest, slides_root=args.slides_root)
        for p in slides_raw
    ]
    
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
    
    # Config structure expected by histocartography_ext.pipeline_runner
    config = {
        'graph': {
            'method': args.graph_method,
            'k': args.k,
            'r': args.r,
        },
        'features': {
            'mode': args.feat_mode,
            'gnn_model_path': args.gnn_model_path,
        },
        # Add other potential configs here
    }

    # Group all outputs under a run-specific subdirectory to avoid cluttering --out_dir
    run_name = _sanitize_run_name(args.run_name) if args.run_name else _default_run_name(args)
    out_dir = Path(args.out_dir) / run_name
    logger.info(f"Run output root: {out_dir}")
    
    for slide_path in slides_to_process:
        # Helpful diagnostic when the manifest contains Windows separators or relative paths.
        logger.info(f"Processing slide path: {slide_path}")
        try:
            run_pipeline(
                slide_path=slide_path,
                output_dir=str(out_dir),
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
