"""
Convert SVS graph/patch embedding outputs from .npy/.npz to .pt format.

This is intended to run after examples/svs_graph_embeddings.py so downstream
training pipelines can load PyTorch tensors directly.
"""

import argparse
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
from tqdm import tqdm


DEFAULT_OUTPUT_ROOT = Path("data") / "wsi_processed"
DEFAULT_OUTPUT_SUBDIR = "histocartography_graph_embeddings"
DEFAULT_MERGED_FILENAME = "all_wsi_fused_embeddings.pt"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert graph embedding outputs from .npy/.npz to .pt files."
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Root output folder used by svs_graph_embeddings.py.",
    )
    parser.add_argument(
        "--output-subdir",
        type=str,
        default=DEFAULT_OUTPUT_SUBDIR,
        help="Subdirectory under output-root containing SVS embedding outputs.",
    )
    parser.add_argument(
        "--shard-index",
        type=int,
        default=0,
        help="Shard index for parallel conversion (0-based).",
    )
    parser.add_argument(
        "--num-shards",
        type=int,
        default=1,
        help="Total shard count for parallel conversion.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing .pt files.",
    )
    parser.add_argument(
        "--export-merged",
        action="store_true",
        help="Export one merged .pt artifact aligned by slide name.",
    )
    parser.add_argument(
        "--merged-filename",
        type=str,
        default=DEFAULT_MERGED_FILENAME,
        help="Filename used for merged .pt artifact under output directory.",
    )
    return parser.parse_args()


def _load_npy(path: Path) -> torch.Tensor:
    return torch.from_numpy(np.load(path))


def _iter_slide_embedding_npy_files(slide_embeddings_dir: Path) -> Iterable[Path]:
    for path in sorted(slide_embeddings_dir.glob("*/graph_embedding.npy")):
        yield path
    for path in sorted(slide_embeddings_dir.glob("*/patch_embedding.npy")):
        yield path
    for path in sorted(slide_embeddings_dir.glob("*/tile_graph_embeddings.npy")):
        yield path
    for path in sorted(slide_embeddings_dir.glob("*/tile_patch_embeddings.npy")):
        yield path


def _iter_aggregate_npz_files(base_dir: Path) -> Iterable[Path]:
    for path in sorted(base_dir.glob("all_wsi_graph_embeddings*.npz")):
        yield path
    for path in sorted(base_dir.glob("all_wsi_patch_embeddings*.npz")):
        yield path


def _apply_sharding(paths: List[Path], shard_index: int, num_shards: int) -> List[Path]:
    if num_shards <= 0:
        raise ValueError("--num-shards must be >= 1")
    if shard_index < 0 or shard_index >= num_shards:
        raise ValueError("--shard-index must be in [0, --num-shards)")
    if num_shards == 1:
        return paths
    return [path for index, path in enumerate(paths) if index % num_shards == shard_index]


def _load_aggregate_npz(path: Path) -> Tuple[List[str], np.ndarray]:
    data = np.load(path, allow_pickle=True)
    slide_names = [str(name) for name in data["slide_names"].tolist()]
    embeddings = data["embeddings"]
    return slide_names, embeddings


def _find_one_aggregate_file(
    base_dir: Path,
    pattern: str,
    shard_index: int,
    num_shards: int,
) -> Optional[Path]:
    if num_shards > 1:
        shard_pattern = f"{pattern}.shard{shard_index:03d}-of-{num_shards:03d}.npz"
        shard_path = base_dir / shard_pattern
        if shard_path.exists():
            return shard_path

    candidates = sorted(base_dir.glob(f"{pattern}*.npz"))
    if len(candidates) == 0:
        return None
    if num_shards > 1:
        shard_candidates = [candidate for candidate in candidates if ".shard" in candidate.name]
        if len(shard_candidates) > 0:
            return None
    return candidates[0]


def _build_from_slide_files(
    slide_embeddings_dir: Path,
) -> Tuple[List[str], Optional[torch.Tensor], Optional[torch.Tensor]]:
    graph_map: Dict[str, np.ndarray] = {}
    patch_map: Dict[str, np.ndarray] = {}

    for graph_path in sorted(slide_embeddings_dir.glob("*/graph_embedding.npy")):
        graph_map[graph_path.parent.name] = np.load(graph_path)
    for patch_path in sorted(slide_embeddings_dir.glob("*/patch_embedding.npy")):
        patch_map[patch_path.parent.name] = np.load(patch_path)

    all_slides: List[str] = sorted(set(graph_map.keys()) | set(patch_map.keys()))
    if len(all_slides) == 0:
        return [], None, None

    graph_dim = None
    patch_dim = None
    if len(graph_map) > 0:
        graph_dim = next(iter(graph_map.values())).shape[-1]
    if len(patch_map) > 0:
        patch_dim = next(iter(patch_map.values())).shape[-1]

    graph_rows: List[np.ndarray] = []
    patch_rows: List[np.ndarray] = []
    for slide_name in all_slides:
        if graph_dim is not None:
            graph_rows.append(graph_map.get(slide_name, np.full((graph_dim,), np.nan, dtype=np.float32)))
        if patch_dim is not None:
            patch_rows.append(patch_map.get(slide_name, np.full((patch_dim,), np.nan, dtype=np.float32)))

    graph_tensor = None
    patch_tensor = None
    if len(graph_rows) > 0:
        graph_tensor = torch.from_numpy(np.stack(graph_rows, axis=0))
    if len(patch_rows) > 0:
        patch_tensor = torch.from_numpy(np.stack(patch_rows, axis=0))
    return all_slides, graph_tensor, patch_tensor


def _export_merged_pt(
    base_dir: Path,
    slide_embeddings_dir: Path,
    merged_filename: str,
    shard_index: int,
    num_shards: int,
    overwrite: bool,
) -> bool:
    merged_path = base_dir / merged_filename
    if num_shards > 1:
        stem = merged_path.stem
        suffix = merged_path.suffix or ".pt"
        merged_path = merged_path.with_name(
            f"{stem}.shard{shard_index:03d}-of-{num_shards:03d}{suffix}"
        )

    if merged_path.exists() and not overwrite:
        return False

    graph_npz = _find_one_aggregate_file(
        base_dir=base_dir,
        pattern="all_wsi_graph_embeddings",
        shard_index=shard_index,
        num_shards=num_shards,
    )
    patch_npz = _find_one_aggregate_file(
        base_dir=base_dir,
        pattern="all_wsi_patch_embeddings",
        shard_index=shard_index,
        num_shards=num_shards,
    )

    if graph_npz is not None and patch_npz is not None:
        graph_names, graph_embeddings = _load_aggregate_npz(graph_npz)
        patch_names, patch_embeddings = _load_aggregate_npz(patch_npz)
        all_slides: List[str] = sorted(set(graph_names) | set(patch_names))

        graph_index = {name: index for index, name in enumerate(graph_names)}
        patch_index = {name: index for index, name in enumerate(patch_names)}

        graph_dim = graph_embeddings.shape[-1]
        patch_dim = patch_embeddings.shape[-1]

        graph_rows: List[np.ndarray] = []
        patch_rows: List[np.ndarray] = []
        for slide_name in all_slides:
            if slide_name in graph_index:
                graph_rows.append(graph_embeddings[graph_index[slide_name]])
            else:
                graph_rows.append(np.full((graph_dim,), np.nan, dtype=np.float32))
            if slide_name in patch_index:
                patch_rows.append(patch_embeddings[patch_index[slide_name]])
            else:
                patch_rows.append(np.full((patch_dim,), np.nan, dtype=np.float32))

        graph_tensor = torch.from_numpy(np.stack(graph_rows, axis=0))
        patch_tensor = torch.from_numpy(np.stack(patch_rows, axis=0))
    else:
        all_slides, graph_tensor, patch_tensor = _build_from_slide_files(slide_embeddings_dir)

    if len(all_slides) == 0:
        raise FileNotFoundError("No slide-level embeddings found for merged export.")

    payload = {
        "slide_names": all_slides,
        "graph_embeddings": graph_tensor,
        "patch_embeddings": patch_tensor,
        "contains_nan_for_missing": True,
    }
    torch.save(payload, merged_path)
    print(f"[INFO] Merged artifact written: {merged_path}")
    return True


def _convert_npy_to_pt(path: Path, overwrite: bool) -> bool:
    pt_path = path.with_suffix(".pt")
    if pt_path.exists() and not overwrite:
        return False
    tensor = _load_npy(path)
    torch.save(tensor, pt_path)
    return True


def _convert_npz_to_pt(path: Path, overwrite: bool) -> bool:
    pt_path = path.with_suffix(".pt")
    if pt_path.exists() and not overwrite:
        return False

    slide_names_np, embeddings_np = _load_aggregate_npz(path)
    slide_names = [str(name) for name in slide_names_np]
    embeddings = torch.from_numpy(embeddings_np)
    payload = {
        "slide_names": slide_names,
        "embeddings": embeddings,
    }
    torch.save(payload, pt_path)
    return True


def _convert_paths(
    npy_paths: List[Path],
    npz_paths: List[Path],
    overwrite: bool,
) -> Tuple[int, int]:
    converted = 0
    skipped = 0

    for path in tqdm(npy_paths, desc="Converting .npy"):
        if _convert_npy_to_pt(path, overwrite=overwrite):
            converted += 1
        else:
            skipped += 1

    for path in tqdm(npz_paths, desc="Converting .npz"):
        if _convert_npz_to_pt(path, overwrite=overwrite):
            converted += 1
        else:
            skipped += 1

    return converted, skipped


def main() -> None:
    args = _parse_args()
    base_dir = args.output_root / args.output_subdir
    slide_embeddings_dir = base_dir / "slide_embeddings"

    if not base_dir.exists():
        raise FileNotFoundError(f"Output directory not found: {base_dir}")
    if not slide_embeddings_dir.exists():
        raise FileNotFoundError(
            f"slide_embeddings directory not found: {slide_embeddings_dir}"
        )

    npy_paths = list(_iter_slide_embedding_npy_files(slide_embeddings_dir))
    npz_paths = list(_iter_aggregate_npz_files(base_dir))

    if len(npy_paths) == 0 and len(npz_paths) == 0:
        raise FileNotFoundError(
            "No embedding files found to convert. "
            f"Checked under: {slide_embeddings_dir} and {base_dir}"
        )

    all_paths = npy_paths + npz_paths
    sharded_all_paths = _apply_sharding(
        all_paths,
        shard_index=args.shard_index,
        num_shards=args.num_shards,
    )
    sharded_path_set = set(sharded_all_paths)
    npy_paths = [path for path in npy_paths if path in sharded_path_set]
    npz_paths = [path for path in npz_paths if path in sharded_path_set]

    if len(sharded_all_paths) == 0:
        print(
            f"No files assigned to shard {args.shard_index}/{args.num_shards}. Nothing to do."
        )
        return

    print(f"[INFO] Base directory: {base_dir}")
    print(f"[INFO] Shard: {args.shard_index}/{args.num_shards}")
    print(f"[INFO] .npy files to convert: {len(npy_paths)}")
    print(f"[INFO] .npz files to convert: {len(npz_paths)}")
    print(f"[INFO] Export merged artifact: {args.export_merged}")

    converted, skipped = _convert_paths(
        npy_paths=npy_paths,
        npz_paths=npz_paths,
        overwrite=args.overwrite,
    )

    merged_converted = 0
    merged_skipped = 0
    if args.export_merged:
        if _export_merged_pt(
            base_dir=base_dir,
            slide_embeddings_dir=slide_embeddings_dir,
            merged_filename=args.merged_filename,
            shard_index=args.shard_index,
            num_shards=args.num_shards,
            overwrite=args.overwrite,
        ):
            merged_converted = 1
        else:
            merged_skipped = 1

    print(f"[INFO] Converted: {converted}")
    print(f"[INFO] Skipped (already existed): {skipped}")
    if args.export_merged:
        print(f"[INFO] Merged converted: {merged_converted}")
        print(f"[INFO] Merged skipped (already existed): {merged_skipped}")
    print("[INFO] Done.")


if __name__ == "__main__":
    main()
