"""
Render graph overlay visualizations from previously exported SVS graph embeddings.

This script expects outputs produced by examples/svs_graph_embeddings.py with --save-graphs.
For each saved tile graph (.bin), it re-reads the corresponding tile from the source
SVS slide and renders a graph overlay image.
"""

import argparse
import csv
import re
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, TypeVar

import numpy as np
from dgl.data.utils import load_graphs

from histocartography.visualization import OverlayGraphVisualization


DEFAULT_INPUT_DIR = Path("data") / "wsi_raw"
DEFAULT_EMBEDDINGS_ROOT = Path("data") / "wsi_processed"
DEFAULT_EMBEDDINGS_SUBDIR = "histocartography_graph_embeddings"
DEFAULT_VIS_SUBDIR = "histocartography_graph_visualizations"

_TILE_COORD_PATTERN = re.compile(r"_x(?P<x>-?\d+)_y(?P<y>-?\d+)$")
T = TypeVar("T")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render per-tile graph overlays from saved SVS graph files."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help="Folder containing source .svs files.",
    )
    parser.add_argument(
        "--embeddings-root",
        type=Path,
        default=DEFAULT_EMBEDDINGS_ROOT,
        help="Root output folder used by svs_graph_embeddings.py (--output-root).",
    )
    parser.add_argument(
        "--embeddings-subdir",
        type=str,
        default=DEFAULT_EMBEDDINGS_SUBDIR,
        help="Subdirectory used by svs_graph_embeddings.py (--output-subdir).",
    )
    parser.add_argument(
        "--visualization-subdir",
        type=str,
        default=DEFAULT_VIS_SUBDIR,
        help="Subdirectory created under embeddings-root for rendered visualizations.",
    )
    parser.add_argument(
        "--glob-pattern",
        type=str,
        default="*.svs",
        help="File pattern used to find slides recursively under input-dir.",
    )
    parser.add_argument(
        "--shard-index",
        type=int,
        default=0,
        help="Shard index for parallel slide processing (0-based).",
    )
    parser.add_argument(
        "--num-shards",
        type=int,
        default=1,
        help="Total number of shards for parallel slide processing.",
    )
    parser.add_argument(
        "--tile-size",
        type=int,
        default=2048,
        help="Tile width/height used during graph extraction.",
    )
    parser.add_argument(
        "--level",
        type=int,
        default=0,
        help="OpenSlide level used to read tiles for rendering.",
    )
    parser.add_argument(
        "--max-tiles-per-slide",
        type=int,
        default=0,
        help="Maximum number of graph tiles rendered per slide (0 = no limit).",
    )
    parser.add_argument(
        "--tile-step",
        type=int,
        default=1,
        help="Render every Nth graph tile (1 = render all).",
    )
    parser.add_argument(
        "--node-style",
        type=str,
        default="outline",
        choices=["outline", "fill"],
        help="Node rendering style.",
    )
    parser.add_argument(
        "--node-color",
        type=str,
        default="yellow",
        help="Node color.",
    )
    parser.add_argument(
        "--node-radius",
        type=int,
        default=5,
        help="Node radius.",
    )
    parser.add_argument(
        "--edge-color",
        type=str,
        default="blue",
        help="Edge color.",
    )
    parser.add_argument(
        "--edge-thickness",
        type=int,
        default=2,
        help="Edge thickness.",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue rendering remaining slides if one slide fails.",
    )
    return parser.parse_args()


def _parse_tile_xy(graph_file: Path) -> Optional[Tuple[int, int]]:
    match = _TILE_COORD_PATTERN.search(graph_file.stem)
    if match is None:
        return None
    return int(match.group("x")), int(match.group("y"))


def _iter_sharded(items: Sequence[T], shard_index: int, num_shards: int) -> Iterable[T]:
    for index, item in enumerate(items):
        if index % num_shards == shard_index:
            yield item


def _resolve_slides_by_stem(slides: Sequence[Path]) -> Dict[str, Path]:
    mapping: Dict[str, Path] = {}
    for slide_path in slides:
        if slide_path.stem not in mapping:
            mapping[slide_path.stem] = slide_path
    return mapping


def _write_csv(rows: Sequence[Mapping[str, object]], path: Path) -> None:
    if len(rows) == 0:
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _process_slide(
    slide_path: Path,
    graph_dir: Path,
    visualizer: OverlayGraphVisualization,
    out_dir: Path,
    tile_size: int,
    level: int,
    tile_step: int,
    max_tiles_per_slide: int,
) -> Dict[str, object]:
    try:
        import openslide
    except ImportError as exc:
        raise ImportError(
            "OpenSlide is required for SVS support. Install openslide-python and OpenSlide libs."
        ) from exc

    graph_files = sorted(graph_dir.glob("*.bin"))
    if tile_step > 1:
        graph_files = graph_files[::tile_step]
    if max_tiles_per_slide > 0:
        graph_files = graph_files[:max_tiles_per_slide]

    slide = openslide.OpenSlide(str(slide_path))
    if level < 0 or level >= slide.level_count:
        slide.close()
        raise ValueError(
            f"Requested level {level} is out of range for {slide_path.name} "
            f"(available levels: 0..{slide.level_count - 1})"
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    rendered_tiles = 0
    skipped_tiles = 0

    for graph_file in graph_files:
        xy = _parse_tile_xy(graph_file)
        if xy is None:
            skipped_tiles += 1
            continue
        x, y = xy

        graphs, _ = load_graphs(str(graph_file))
        if len(graphs) == 0:
            skipped_tiles += 1
            continue
        graph = graphs[0]

        tile_rgba = slide.read_region((x, y), level, (tile_size, tile_size))
        tile_rgb = np.array(tile_rgba.convert("RGB"), dtype=np.uint8)

        out_img = visualizer.process(tile_rgb, graph)
        out_img.save(out_dir / f"{graph_file.stem}_overlay.png", quality=95)
        rendered_tiles += 1

    slide.close()

    return {
        "slide": slide_path.name,
        "graph_tiles_found": len(sorted(graph_dir.glob("*.bin"))),
        "rendered_tiles": rendered_tiles,
        "skipped_tiles": skipped_tiles,
    }


def main() -> None:
    args = _parse_args()

    if args.num_shards <= 0:
        raise ValueError("--num-shards must be >= 1")
    if args.shard_index < 0 or args.shard_index >= args.num_shards:
        raise ValueError("--shard-index must be in [0, --num-shards)")
    if args.tile_step <= 0:
        raise ValueError("--tile-step must be >= 1")

    if not args.input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {args.input_dir}")

    embeddings_dir = args.embeddings_root / args.embeddings_subdir
    graphs_root = embeddings_dir / "graphs"
    if not graphs_root.exists():
        raise FileNotFoundError(
            f"Graph directory not found: {graphs_root}. Run svs_graph_embeddings.py with --save-graphs first."
        )

    slide_paths = sorted(args.input_dir.rglob(args.glob_pattern))
    if len(slide_paths) == 0:
        raise FileNotFoundError(
            f"No files matching pattern '{args.glob_pattern}' were found in {args.input_dir}"
        )

    slide_by_stem = _resolve_slides_by_stem(slide_paths)
    candidate_graph_dirs = sorted(path for path in graphs_root.iterdir() if path.is_dir())
    if len(candidate_graph_dirs) == 0:
        raise FileNotFoundError(f"No slide graph directories found under: {graphs_root}")

    matched_pairs: List[Tuple[Path, Path]] = []
    unmatched_graph_dirs: List[str] = []
    for graph_dir in candidate_graph_dirs:
        slide_path = slide_by_stem.get(graph_dir.name)
        if slide_path is None:
            unmatched_graph_dirs.append(graph_dir.name)
            continue
        matched_pairs.append((slide_path, graph_dir))

    if len(matched_pairs) == 0:
        raise RuntimeError(
            "No matching slides found between input-dir and graph directories. "
            "Ensure slide stems match graph folder names."
        )

    matched_pairs = list(_iter_sharded(matched_pairs, args.shard_index, args.num_shards))
    if len(matched_pairs) == 0:
        print(f"No slides assigned to shard {args.shard_index}/{args.num_shards}. Nothing to do.")
        return

    output_root = args.embeddings_root / args.visualization_subdir
    output_root.mkdir(parents=True, exist_ok=True)

    visualizer = OverlayGraphVisualization(
        node_style=args.node_style,
        node_color=args.node_color,
        node_radius=args.node_radius,
        edge_color=args.edge_color,
        edge_thickness=args.edge_thickness,
    )

    summary_rows: List[Dict[str, object]] = []
    failure_rows: List[Dict[str, str]] = []

    for slide_path, graph_dir in matched_pairs:
        try:
            row = _process_slide(
                slide_path=slide_path,
                graph_dir=graph_dir,
                visualizer=visualizer,
                out_dir=output_root / slide_path.stem,
                tile_size=args.tile_size,
                level=args.level,
                tile_step=args.tile_step,
                max_tiles_per_slide=args.max_tiles_per_slide,
            )
            summary_rows.append(row)
        except Exception as exc:
            if not args.continue_on_error:
                raise
            failure_rows.append({"slide": slide_path.name, "error": str(exc)})

    summary_name = "visualization_summary.csv"
    failures_name = "visualization_failures.csv"
    if args.num_shards > 1:
        suffix = f".shard{args.shard_index:03d}-of-{args.num_shards:03d}"
        summary_name = f"visualization_summary{suffix}.csv"
        failures_name = f"visualization_failures{suffix}.csv"

    _write_csv(summary_rows, output_root / summary_name)
    _write_csv(failure_rows, output_root / failures_name)

    if len(unmatched_graph_dirs) > 0:
        unmatched_rows = [{"graph_dir": name} for name in unmatched_graph_dirs]
        _write_csv(unmatched_rows, output_root / "visualization_unmatched_graph_dirs.csv")

    print(f"Done. Visualizations written to: {output_root}")


if __name__ == "__main__":
    main()
