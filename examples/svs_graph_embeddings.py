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
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np
from dgl.data.utils import save_graphs
from skimage import measure, morphology
from tqdm import tqdm
import torch

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
        "--nuclei-mode",
        type=str,
        default="hovernet",
        choices=["hovernet", "hovernet-package", "classical"],
        help="Nuclei extraction backend. 'classical' avoids checkpoint dependencies.",
    )
    parser.add_argument(
        "--hovernet-pkg-model-mode",
        type=str,
        default="auto",
        choices=["auto", "fast", "original"],
        help="Model mode when using --nuclei-mode hovernet-package. Use auto to infer from checkpoint.",
    )
    parser.add_argument(
        "--hovernet-pkg-nr-types",
        type=int,
        default=-1,
        help="Number of nuclei types for hovernet-package backend (-1 = infer from checkpoint, 0 = segmentation only).",
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


def _extract_nuclei_classical(
    tile: np.ndarray,
    min_area: int = 20,
) -> Tuple[np.ndarray, np.ndarray]:
    """Simple nuclei-like instance extraction fallback that does not require a learned model."""
    gray = cv2.cvtColor(tile, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, binary = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    kernel = np.ones((3, 3), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
    binary = morphology.remove_small_objects(binary.astype(bool), min_size=min_area)

    labeled = measure.label(binary, connectivity=2)
    regions = measure.regionprops(labeled)
    centroids = np.empty((len(regions), 2), dtype=np.float32)
    for index, region in enumerate(regions):
        center_y, center_x = region.centroid
        centroids[index, 0] = float(center_x)
        centroids[index, 1] = float(center_y)
    return labeled.astype(np.int32), centroids


def _load_torch_checkpoint(checkpoint_path: Path) -> Any:
    try:
        return torch.load(str(checkpoint_path), map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(str(checkpoint_path), map_location="cpu")


def _infer_hovernet_pkg_model_args(
    checkpoint_desc: Dict[str, Any],
    mode_override: str,
    nr_types_override: int,
) -> Tuple[str, Optional[int]]:
    mode = mode_override
    if mode == "auto":
        decoder_key_candidates = [
            "decoder.np.u3.conva.weight",
            "decoder.hv.u3.conva.weight",
            "decoder.tp.u3.conva.weight",
            "decoder.np.u2.conva.weight",
            "decoder.hv.u2.conva.weight",
            "decoder.tp.u2.conva.weight",
            "decoder.np.u1.conva.weight",
            "decoder.hv.u1.conva.weight",
            "decoder.tp.u1.conva.weight",
        ]
        inferred_kernel = None
        for key in decoder_key_candidates:
            value = checkpoint_desc.get(key, None)
            if value is not None and hasattr(value, "shape") and len(value.shape) >= 4:
                inferred_kernel = int(value.shape[-1])
                break

        if inferred_kernel == 3:
            mode = "fast"
        elif inferred_kernel == 5:
            mode = "original"
        else:
            has_fast_pad = any(
                isinstance(key, str) and key.startswith("conv0.pad")
                for key in checkpoint_desc.keys()
            )
            mode = "fast" if has_fast_pad else "original"

    nr_types: Optional[int]
    if nr_types_override >= 0:
        nr_types = None if nr_types_override == 0 else nr_types_override
    else:
        tp_key = "decoder.tp.u0.conv.weight"
        if tp_key in checkpoint_desc and hasattr(checkpoint_desc[tp_key], "shape"):
            nr_types = int(checkpoint_desc[tp_key].shape[0])
        else:
            nr_types = None

    return mode, nr_types


class HoverNetPackageNucleiExtractor:
    """Nuclei extraction using the external hover-net package model/checkpoint conventions."""

    def __init__(
        self,
        model_path: Path,
        model_mode: str = "auto",
        nr_types: int = -1,
        batch_size: int = 4,
    ) -> None:
        try:
            from hover_net.models.hovernet.net_desc import create_model
            from hover_net.models.hovernet.post_proc import process as hover_post_process
            from hover_net.models.hovernet.run_desc import infer_step
            from hover_net.run_utils.utils import convert_pytorch_checkpoint
        except Exception as exc:
            raise ImportError(
                "hover-net package backend requested, but imports failed. "
                "Install with `pip install hover-net`."
            ) from exc

        if not model_path.is_file():
            raise FileNotFoundError(
                f"hover-net checkpoint not found: {model_path}. "
                "Provide a valid file via --nuclei-model-path."
            )

        checkpoint = _load_torch_checkpoint(model_path)
        if isinstance(checkpoint, dict) and "desc" in checkpoint and isinstance(checkpoint["desc"], dict):
            checkpoint_desc = checkpoint["desc"]
        elif isinstance(checkpoint, dict):
            checkpoint_desc = checkpoint
        else:
            raise RuntimeError(
                f"Unsupported hover-net checkpoint format in {model_path}. "
                "Expected a dict with key 'desc' or a plain state_dict-like dict."
            )

        checkpoint_desc = convert_pytorch_checkpoint(checkpoint_desc)
        resolved_mode, resolved_nr_types = _infer_hovernet_pkg_model_args(
            checkpoint_desc,
            mode_override=model_mode,
            nr_types_override=nr_types,
        )

        self.model = create_model(mode=resolved_mode, nr_types=resolved_nr_types)
        try:
            self.model.load_state_dict(checkpoint_desc, strict=True)
        except RuntimeError as exc:
            # Guardrail for ambiguous auto-detection on non-standard checkpoints.
            if model_mode == "auto":
                retry_mode = "original" if resolved_mode == "fast" else "fast"
                retry_model = create_model(mode=retry_mode, nr_types=resolved_nr_types)
                try:
                    retry_model.load_state_dict(checkpoint_desc, strict=True)
                    self.model = retry_model
                    resolved_mode = retry_mode
                except RuntimeError:
                    raise RuntimeError(
                        "Unable to load hover-net checkpoint with either model mode. "
                        "Try setting --hovernet-pkg-model-mode explicitly to 'fast' or 'original'."
                    ) from exc
            else:
                raise RuntimeError(
                    "Checkpoint/model-mode mismatch for hover-net backend. "
                    f"Selected mode='{model_mode}'. Try the other mode or use --hovernet-pkg-model-mode auto."
                ) from exc
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)
        self.model.eval()

        self.resolved_mode = resolved_mode
        self.resolved_nr_types = resolved_nr_types
        self.infer_step = infer_step
        self.post_process = hover_post_process
        self.batch_size = max(1, int(batch_size))
        if self.resolved_mode == "fast":
            self.patch_input_shape = 256
            self.patch_output_shape = 164
        else:
            self.patch_input_shape = 270
            self.patch_output_shape = 80

    @staticmethod
    def _prepare_patching(
        image: np.ndarray,
        input_size: int,
        output_size: int,
    ) -> Tuple[np.ndarray, np.ndarray, Tuple[int, int]]:
        step = output_size

        def _last_steps(length: int, mask_size: int, step_size: int) -> int:
            nr_step = int(np.ceil((length - mask_size) / step_size))
            return int((nr_step + 1) * step_size)

        height, width = image.shape[:2]
        last_h = _last_steps(height, output_size, step)
        last_w = _last_steps(width, output_size, step)

        diff = input_size - step
        pad_top = diff // 2
        pad_left = diff // 2
        pad_bottom = last_h + input_size - height
        pad_right = last_w + input_size - width

        padded = np.pad(
            image,
            ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)),
            mode="reflect",
        )

        coord_y = np.arange(0, last_h, step, dtype=np.int32)
        coord_x = np.arange(0, last_w, step, dtype=np.int32)
        row_idx = np.arange(0, coord_y.shape[0], dtype=np.int32)
        col_idx = np.arange(0, coord_x.shape[0], dtype=np.int32)

        coord_y, coord_x = np.meshgrid(coord_y, coord_x)
        row_idx, col_idx = np.meshgrid(row_idx, col_idx)

        patch_info = np.stack(
            [coord_y.flatten(), coord_x.flatten(), row_idx.flatten(), col_idx.flatten()],
            axis=-1,
        )
        return padded, patch_info, (pad_top, pad_left)

    @staticmethod
    def _reassemble_prediction(
        patch_predictions: List[np.ndarray],
        patch_info: np.ndarray,
        src_shape: Tuple[int, int],
    ) -> np.ndarray:
        if len(patch_predictions) == 0:
            raise RuntimeError("No hover-net patch predictions were produced.")

        entries = list(zip(patch_info.tolist(), patch_predictions))
        entries = sorted(entries, key=lambda item: [item[0][0], item[0][1]])
        sorted_info, sorted_data = zip(*entries)

        patch_shape = np.squeeze(sorted_data[0]).shape
        channels = 1 if len(patch_shape) == 2 else patch_shape[-1]

        nr_row = max(info[2] for info in sorted_info) + 1
        nr_col = max(info[3] for info in sorted_info) + 1

        pred_map = np.concatenate(sorted_data, axis=0)
        pred_map = np.reshape(pred_map, (nr_row, nr_col) + patch_shape)
        axes = [0, 2, 1, 3, 4] if channels != 1 else [0, 2, 1, 3]
        pred_map = np.transpose(pred_map, axes)
        pred_map = np.reshape(
            pred_map,
            (patch_shape[0] * nr_row, patch_shape[1] * nr_col, channels),
        )
        pred_map = np.squeeze(pred_map[: src_shape[0], : src_shape[1]])
        return pred_map

    def process(self, input_image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if input_image.ndim != 3 or input_image.shape[2] != 3:
            raise ValueError("Expected RGB image with shape [H, W, 3] for hover-net inference.")

        src_h, src_w = input_image.shape[:2]
        padded, patch_info, _ = self._prepare_patching(
            input_image,
            input_size=self.patch_input_shape,
            output_size=self.patch_output_shape,
        )

        patch_predictions: List[np.ndarray] = []
        total = patch_info.shape[0]
        for start in range(0, total, self.batch_size):
            end = min(start + self.batch_size, total)
            batch_info = patch_info[start:end]
            batch_patches = [
                padded[
                    int(info[0]) : int(info[0]) + self.patch_input_shape,
                    int(info[1]) : int(info[1]) + self.patch_input_shape,
                ]
                for info in batch_info
            ]
            batch_tensor = torch.from_numpy(np.stack(batch_patches, axis=0).copy())
            pred_batch = self.infer_step(batch_tensor, self.model)
            patch_predictions.extend([pred_batch[index : index + 1] for index in range(pred_batch.shape[0])])

        pred_map = self._reassemble_prediction(
            patch_predictions=patch_predictions,
            patch_info=patch_info,
            src_shape=(src_h, src_w),
        )
        inst_map, inst_info = self.post_process(
            pred_map,
            nr_types=self.resolved_nr_types,
            return_centroids=True,
        )

        target_h, target_w = input_image.shape[:2]
        map_h, map_w = inst_map.shape[:2]

        if map_h == target_h and map_w == target_w:
            aligned_map = inst_map.astype(np.int32)
            offset_x = 0
            offset_y = 0
        else:
            aligned_map = np.zeros((target_h, target_w), dtype=np.int32)
            paste_h = min(map_h, target_h)
            paste_w = min(map_w, target_w)
            src_y0 = max((map_h - paste_h) // 2, 0)
            src_x0 = max((map_w - paste_w) // 2, 0)
            dst_y0 = max((target_h - paste_h) // 2, 0)
            dst_x0 = max((target_w - paste_w) // 2, 0)
            aligned_map[dst_y0 : dst_y0 + paste_h, dst_x0 : dst_x0 + paste_w] = inst_map[
                src_y0 : src_y0 + paste_h,
                src_x0 : src_x0 + paste_w,
            ].astype(np.int32)
            offset_x = float(dst_x0 - src_x0)
            offset_y = float(dst_y0 - src_y0)

        if inst_info is None or len(inst_info) == 0:
            centroids = np.empty((0, 2), dtype=np.float32)
        else:
            centroid_rows: List[np.ndarray] = []
            for nucleus in inst_info.values():
                centroid = nucleus.get("centroid", None)
                if centroid is None:
                    continue
                centroid = np.asarray(centroid, dtype=np.float32)
                centroid[0] += offset_x
                centroid[1] += offset_y
                centroid_rows.append(centroid)
            centroids = (
                np.stack(centroid_rows, axis=0)
                if len(centroid_rows) > 0
                else np.empty((0, 2), dtype=np.float32)
            )

        return aligned_map, centroids


def _build_extractors(args: argparse.Namespace):
    nuclei_extractor = None
    if args.nuclei_mode == "hovernet":
        nuclei_extractor = NucleiExtractor(
            pretrained_data=args.nuclei_pretrained_data,
            model_path=None if args.nuclei_model_path is None else str(args.nuclei_model_path),
            batch_size=args.nuclei_batch_size,
        )
    elif args.nuclei_mode == "hovernet-package":
        if args.nuclei_model_path is None:
            raise ValueError(
                "--nuclei-model-path is required when --nuclei-mode hovernet-package"
            )
        nuclei_extractor = HoverNetPackageNucleiExtractor(
            model_path=args.nuclei_model_path,
            model_mode=args.hovernet_pkg_model_mode,
            nr_types=args.hovernet_pkg_nr_types,
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
    nuclei_extractor: Optional[Any],
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

        if nuclei_extractor is None:
            nuclei_map, _ = _extract_nuclei_classical(tile)
        else:
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
    summary_name: str = "processing_summary.csv",
) -> None:
    import pandas as pd

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(output_dir / summary_name, index=False)


def main() -> None:
    args = _parse_args()
    args.output_dir = args.output_root / args.output_subdir

    if args.num_workers > 0:
        import multiprocessing as mp

        try:
            mp.set_start_method("spawn", force=True)
        except RuntimeError:
            pass

    if not args.input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {args.input_dir}")

    slide_paths = sorted(args.input_dir.rglob(args.glob_pattern))
    if len(slide_paths) == 0:
        raise FileNotFoundError(
            f"No files matching pattern '{args.glob_pattern}' were found in {args.input_dir}"
        )

    if args.num_shards <= 0:
        raise ValueError("--num-shards must be >= 1")
    if args.shard_index < 0 or args.shard_index >= args.num_shards:
        raise ValueError("--shard-index must be in [0, --num-shards)")

    if args.num_shards > 1:
        slide_paths = [
            slide_path
            for index, slide_path in enumerate(slide_paths)
            if index % args.num_shards == args.shard_index
        ]
        if len(slide_paths) == 0:
            print(
                f"No slides assigned to shard {args.shard_index}/{args.num_shards}. Nothing to do."
            )
            return

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

    graph_npz_name = "all_wsi_graph_embeddings.npz"
    patch_npz_name = "all_wsi_patch_embeddings.npz"
    summary_name = "processing_summary.csv"
    failures_name = "processing_failures.csv"
    if args.num_shards > 1:
        suffix = f".shard{args.shard_index:03d}-of-{args.num_shards:03d}"
        graph_npz_name = f"all_wsi_graph_embeddings{suffix}.npz"
        patch_npz_name = f"all_wsi_patch_embeddings{suffix}.npz"
        summary_name = f"processing_summary{suffix}.csv"
        failures_name = f"processing_failures{suffix}.csv"

    if len(pooled_graph_embeddings) > 0:
        np.savez(
            out_dirs["base"] / graph_npz_name,
            slide_names=np.array(pooled_graph_slide_names, dtype=object),
            embeddings=np.stack(pooled_graph_embeddings, axis=0),
        )
    if len(pooled_patch_embeddings) > 0:
        np.savez(
            out_dirs["base"] / patch_npz_name,
            slide_names=np.array(pooled_patch_slide_names, dtype=object),
            embeddings=np.stack(pooled_patch_embeddings, axis=0),
        )

    _save_summary(summary_rows, out_dirs["base"], summary_name=summary_name)
    if len(failed_rows) > 0:
        import pandas as pd

        pd.DataFrame(failed_rows).to_csv(out_dirs["base"] / failures_name, index=False)
    print(f"Done. Results written to: {out_dirs['base']}")


if __name__ == "__main__":
    main()
