"""
Example: Process a folder of SVS whole-slide images (WSIs) with HistoCartography.

This script:
1) reads SVS slides with OpenSlide,
2) extracts non-overlapping (or strided) tiles,
3) builds one cell graph per tile,
4) computes per-tile graph embeddings and a pooled embedding per WSI,
5) optionally computes 256x256 patch embeddings and a pooled embedding per WSI.

Outputs are saved as .npy/.npz/.csv files under --output-dir.
"""

import argparse
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from dgl.data.utils import save_graphs
from tqdm import tqdm

from histocartography.preprocessing import (
    DeepFeatureExtractor,
    GridDeepFeatureExtractor,
    KNNGraphBuilder,
    NucleiExtractor,
)


DEFAULT_INPUT_DIR = Path("data") / "wsi_raw"
DEFAULT_OUTPUT_ROOT = Path("data") / "wsi_processed"
DEFAULT_OUTPUT_SUBDIR = "histocartography_graph_embeddings"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Batch process SVS slides and export graph/patch embeddings."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help="Folder containing .svs files.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Root output folder (processed WSI root).",
    )
    parser.add_argument(
        "--output-subdir",
        type=str,
        default=DEFAULT_OUTPUT_SUBDIR,
        help="Informative subdirectory created under output-root for this pipeline run.",
    )
    parser.add_argument(
        "--glob-pattern",
        type=str,
        default="*.svs",
        help="File pattern used to find slides recursively under input-dir.",
    )
    parser.add_argument(
        "--tile-size",
        type=int,
        default=2048,
        help="Tile width/height in level pixels.",
    )
    parser.add_argument(
        "--tile-stride",
        type=int,
        default=2048,
        help="Stride between adjacent tile origins in level pixels.",
    )
    parser.add_argument(
        "--level",
        type=int,
        default=0,
        help="OpenSlide pyramid level used for reading tiles.",
    )
    parser.add_argument(
        "--max-tiles-per-slide",
        type=int,
        default=0,
        help="Maximum number of accepted tiles per slide (0 = no limit).",
    )
    parser.add_argument(
        "--min-tissue-fraction",
        type=float,
        default=0.05,
        help="Skip tiles below this tissue fraction estimate.",
    )
    parser.add_argument(
        "--white-threshold",
        type=int,
        default=225,
        help="Pixel intensity threshold used in tissue estimation.",
    )
    parser.add_argument(
        "--nuclei-pretrained-data",
        type=str,
        default="pannuke",
        choices=["pannuke", "monusac"],
        help="HoverNet checkpoint used by NucleiExtractor.",
    )
    parser.add_argument(
        "--nuclei-model-path",
        type=Path,
        default=None,
        help="Optional explicit path to a nuclei checkpoint. When provided, this overrides automatic checkpoint download logic.",
    )
    parser.add_argument(
        "--nuclei-batch-size",
        type=int,
        default=8,
        help="Batch size for nuclei extraction (HoverNet).",
    )
    parser.add_argument(
        "--cell-feature-arch",
        type=str,
        default="resnet34",
        help="Backbone for nuclei-centered feature extraction.",
    )
    parser.add_argument(
        "--cell-patch-size",
        type=int,
        default=72,
        help="Patch size around nuclei centroids for node features.",
    )
    parser.add_argument(
        "--cell-resize-size",
        type=int,
        default=224,
        help="Resize used for node feature extraction.",
    )
    parser.add_argument(
        "--cell-feature-batch-size",
        type=int,
        default=32,
        help="Batch size for node feature extractor.",
    )
    parser.add_argument(
        "--knn-k",
        type=int,
        default=5,
        help="k for KNN graph builder.",
    )
    parser.add_argument(
        "--knn-thresh",
        type=float,
        default=50.0,
        help="Distance threshold for KNN graph edges.",
    )
    parser.add_argument(
        "--graph-pool",
        type=str,
        default="mean",
        choices=["mean", "max"],
        help="Pooling op used for graph-to-vector embedding.",
    )
    parser.add_argument(
        "--save-graphs",
        action="store_true",
        help="Save one DGL graph (.bin) per accepted tile.",
    )
    parser.add_argument(
        "--export-patch-embeddings",
        action="store_true",
        help="Compute patch-level embeddings on each accepted tile.",
    )
    parser.add_argument(
        "--patch-arch",
        type=str,
        default="resnet34",
        help="Backbone used for patch-level embeddings.",
    )
    parser.add_argument(
        "--patch-size",
        type=int,
        default=256,
        help="Patch size for patch-level embeddings (default: 256).",
    )
    parser.add_argument(
        "--patch-resize-size",
        type=int,
        default=224,
        help="Resize used for patch-level embedding extraction.",
    )
    parser.add_argument(
        "--patch-stride",
        type=int,
        default=256,
        help="Stride for patch-level extraction.",
    )
    parser.add_argument(
        "--patch-feature-batch-size",
        type=int,
        default=64,
        help="Batch size for patch-level feature extraction.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="DataLoader workers used by feature extractors.",
    )
    parser.add_argument(
        "--save-patch-cubes",
        action="store_true",
        help="Save per-tile patch embedding cubes (can use substantial disk space).",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose extraction progress in feature extractors.",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue processing remaining slides if one slide fails.",
    )
    return parser.parse_args()


def _estimate_tissue_fraction(tile: np.ndarray, white_threshold: int) -> float:
    non_white = np.any(tile < white_threshold, axis=2)
    return float(non_white.mean())


def _iter_tile_origins(
    width: int,
    height: int,
    tile_size: int,
    stride: int,
) -> Iterable[Tuple[int, int]]:
    max_x = max(0, width - tile_size)
    max_y = max(0, height - tile_size)
    y = 0
    while y <= max_y:
        x = 0
        while x <= max_x:
            yield x, y
            x += stride
        y += stride


def _pool_rows(vectors: Sequence[np.ndarray], pool: str) -> Optional[np.ndarray]:
    if len(vectors) == 0:
        return None
    stacked = np.stack(vectors, axis=0)
    if pool == "mean":
        return stacked.mean(axis=0)
    if pool == "max":
        return stacked.max(axis=0)
    raise ValueError(f"Unsupported pooling method: {pool}")


def _graph_embedding_from_node_features(node_features: np.ndarray, pool: str) -> np.ndarray:
    if pool == "mean":
        return node_features.mean(axis=0)
    if pool == "max":
        return node_features.max(axis=0)
    raise ValueError(f"Unsupported pooling method: {pool}")


def _build_extractors(args: argparse.Namespace):
    nuclei_extractor = NucleiExtractor(
        pretrained_data=args.nuclei_pretrained_data,
        model_path=None if args.nuclei_model_path is None else str(args.nuclei_model_path),
        batch_size=args.nuclei_batch_size,
    )
    node_feature_extractor = DeepFeatureExtractor(
        architecture=args.cell_feature_arch,
        patch_size=args.cell_patch_size,
        resize_size=args.cell_resize_size,
        batch_size=args.cell_feature_batch_size,
        num_workers=args.num_workers,
        verbose=args.verbose,
    )
    graph_builder = KNNGraphBuilder(
        k=args.knn_k,
        thresh=args.knn_thresh,
        add_loc_feats=True,
    )
    patch_feature_extractor = None
    if args.export_patch_embeddings:
        patch_feature_extractor = GridDeepFeatureExtractor(
            architecture=args.patch_arch,
            patch_size=args.patch_size,
            resize_size=args.patch_resize_size,
            stride=args.patch_stride,
            batch_size=args.patch_feature_batch_size,
            num_workers=args.num_workers,
            verbose=args.verbose,
        )
    return nuclei_extractor, node_feature_extractor, graph_builder, patch_feature_extractor


def _process_one_slide(
    slide_path: Path,
    args: argparse.Namespace,
    nuclei_extractor: NucleiExtractor,
    node_feature_extractor: DeepFeatureExtractor,
    graph_builder: KNNGraphBuilder,
    patch_feature_extractor: Optional[GridDeepFeatureExtractor],
    out_dirs: Dict[str, Path],
) -> Dict[str, object]:
    try:
        import openslide
    except ImportError as exc:
        raise ImportError(
            "OpenSlide is required for SVS support. "
            "Install the Python package (pip install openslide-python) and the OpenSlide system library."
        ) from exc

    slide = openslide.OpenSlide(str(slide_path))
    if args.level < 0 or args.level >= slide.level_count:
        slide.close()
        raise ValueError(
            f"Requested level {args.level} is out of range for {slide_path.name} "
            f"(available levels: 0..{slide.level_count - 1})"
        )

    width, height = slide.level_dimensions[args.level]
    tile_origins = list(_iter_tile_origins(width, height, args.tile_size, args.tile_stride))

    graph_vectors: List[np.ndarray] = []
    patch_vectors: List[np.ndarray] = []
    accepted_tiles = 0
    skipped_background = 0
    skipped_no_nuclei = 0

    slide_graph_dir = out_dirs["graphs"] / slide_path.stem
    slide_patch_cube_dir = out_dirs["patch_cubes"] / slide_path.stem
    if args.save_graphs:
        slide_graph_dir.mkdir(parents=True, exist_ok=True)
    if args.save_patch_cubes and args.export_patch_embeddings:
        slide_patch_cube_dir.mkdir(parents=True, exist_ok=True)

    for tile_index, (x, y) in enumerate(tile_origins):
        if args.max_tiles_per_slide > 0 and accepted_tiles >= args.max_tiles_per_slide:
            break

        rgba = slide.read_region((x, y), args.level, (args.tile_size, args.tile_size))
        tile = np.array(rgba.convert("RGB"), dtype=np.uint8)

        tissue_fraction = _estimate_tissue_fraction(tile, args.white_threshold)
        if tissue_fraction < args.min_tissue_fraction:
            skipped_background += 1
            continue

        nuclei_map, _ = nuclei_extractor.process(tile)
        if nuclei_map.max() == 0:
            skipped_no_nuclei += 1
            continue

        node_features = node_feature_extractor.process(tile, nuclei_map)
        graph = graph_builder.process(nuclei_map, node_features)
        node_feats_np = graph.ndata["feat"].cpu().detach().numpy()
        graph_vector = _graph_embedding_from_node_features(node_feats_np, args.graph_pool)
        graph_vectors.append(graph_vector)

        tile_tag = f"tile_{tile_index:06d}_x{x}_y{y}"
        if args.save_graphs:
            save_graphs(str(slide_graph_dir / f"{tile_tag}.bin"), [graph])

        if patch_feature_extractor is not None:
            patch_cube = patch_feature_extractor.process(tile).cpu().detach().numpy()
            patch_matrix = patch_cube.reshape(-1, patch_cube.shape[-1])
            patch_vector = _graph_embedding_from_node_features(patch_matrix, args.graph_pool)
            patch_vectors.append(patch_vector)
            if args.save_patch_cubes:
                np.save(slide_patch_cube_dir / f"{tile_tag}.npy", patch_cube)

        accepted_tiles += 1

    slide.close()

    wsi_graph_embedding = _pool_rows(graph_vectors, args.graph_pool)
    wsi_patch_embedding = _pool_rows(patch_vectors, args.graph_pool)

    slide_out_base = out_dirs["slide_embeddings"] / slide_path.stem
    slide_out_base.mkdir(parents=True, exist_ok=True)
    if wsi_graph_embedding is not None:
        np.save(slide_out_base / "graph_embedding.npy", wsi_graph_embedding)
    if wsi_patch_embedding is not None:
        np.save(slide_out_base / "patch_embedding.npy", wsi_patch_embedding)
    if len(graph_vectors) > 0:
        np.save(slide_out_base / "tile_graph_embeddings.npy", np.stack(graph_vectors, axis=0))
    if len(patch_vectors) > 0:
        np.save(slide_out_base / "tile_patch_embeddings.npy", np.stack(patch_vectors, axis=0))

    return {
        "slide": slide_path.name,
        "level": args.level,
        "level_width": width,
        "level_height": height,
        "candidate_tiles": len(tile_origins),
        "accepted_tiles": accepted_tiles,
        "skipped_background": skipped_background,
        "skipped_no_nuclei": skipped_no_nuclei,
        "graph_embedding_saved": wsi_graph_embedding is not None,
        "patch_embedding_saved": wsi_patch_embedding is not None,
    }


def _save_summary(
    summary_rows: List[Dict[str, object]],
    output_dir: Path,
) -> None:
    import pandas as pd

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(output_dir / "processing_summary.csv", index=False)


def main() -> None:
    args = _parse_args()
    args.output_dir = args.output_root / args.output_subdir

    if not args.input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {args.input_dir}")

    slide_paths = sorted(args.input_dir.rglob(args.glob_pattern))
    if len(slide_paths) == 0:
        raise FileNotFoundError(
            f"No files matching pattern '{args.glob_pattern}' were found in {args.input_dir}"
        )

    out_dirs = {
        "base": args.output_dir,
        "graphs": args.output_dir / "graphs",
        "slide_embeddings": args.output_dir / "slide_embeddings",
        "patch_cubes": args.output_dir / "patch_cubes",
    }
    for directory in out_dirs.values():
        directory.mkdir(parents=True, exist_ok=True)

    (
        nuclei_extractor,
        node_feature_extractor,
        graph_builder,
        patch_feature_extractor,
    ) = _build_extractors(args)

    summary_rows: List[Dict[str, object]] = []
    failed_rows: List[Dict[str, str]] = []
    pooled_graph_embeddings: List[np.ndarray] = []
    pooled_patch_embeddings: List[np.ndarray] = []
    pooled_graph_slide_names: List[str] = []
    pooled_patch_slide_names: List[str] = []

    for slide_path in tqdm(slide_paths, desc="Slides"):
        try:
            slide_result = _process_one_slide(
                slide_path=slide_path,
                args=args,
                nuclei_extractor=nuclei_extractor,
                node_feature_extractor=node_feature_extractor,
                graph_builder=graph_builder,
                patch_feature_extractor=patch_feature_extractor,
                out_dirs=out_dirs,
            )
            summary_rows.append(slide_result)
        except Exception as exc:
            if not args.continue_on_error:
                raise
            failed_rows.append({"slide": slide_path.name, "error": str(exc)})
            continue

        slide_embed_dir = out_dirs["slide_embeddings"] / slide_path.stem
        graph_embed_path = slide_embed_dir / "graph_embedding.npy"
        patch_embed_path = slide_embed_dir / "patch_embedding.npy"
        if graph_embed_path.exists():
            pooled_graph_embeddings.append(np.load(graph_embed_path))
            pooled_graph_slide_names.append(slide_path.name)
        if patch_embed_path.exists():
            pooled_patch_embeddings.append(np.load(patch_embed_path))
            pooled_patch_slide_names.append(slide_path.name)

    if len(pooled_graph_embeddings) > 0:
        np.savez(
            out_dirs["base"] / "all_wsi_graph_embeddings.npz",
            slide_names=np.array(pooled_graph_slide_names, dtype=object),
            embeddings=np.stack(pooled_graph_embeddings, axis=0),
        )
    if len(pooled_patch_embeddings) > 0:
        np.savez(
            out_dirs["base"] / "all_wsi_patch_embeddings.npz",
            slide_names=np.array(pooled_patch_slide_names, dtype=object),
            embeddings=np.stack(pooled_patch_embeddings, axis=0),
        )

    _save_summary(summary_rows, out_dirs["base"])
    if len(failed_rows) > 0:
        import pandas as pd

        pd.DataFrame(failed_rows).to_csv(out_dirs["base"] / "processing_failures.csv", index=False)
    print(f"Done. Results written to: {out_dirs['base']}")


if __name__ == "__main__":
    main()
