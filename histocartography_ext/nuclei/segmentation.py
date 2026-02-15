"""
Nuclei segmentation from whole-slide images using HoVer-Net.

Supports two stitching strategies controlled by ``stitch_mode``:

* ``"global"`` (default) – stitch raw HoVer-Net prediction maps into a
  single global tensor, then run post-processing once.  This matches the
  original HistoCartography ``NucleiExtractor`` and avoids boundary
  artefacts.  Requires enough RAM to hold the full prediction tensor
  (approx ``H * W * 3 * 4`` bytes in float32).

* ``"tile"`` – post-process each tile independently and de-duplicate
  nuclei at overlap boundaries via a spatial KD-tree.  Useful for local
  testing when RAM is limited.
"""

import os
import numpy as np
import pandas as pd
import openslide
from openslide import OpenSlide
from tqdm import tqdm
from typing import Optional, List, Dict, Any, Tuple

from .utils import tile_iterator
from .inference import HoverNetInferencer
from .postprocess import process_instance

# Import the canonical tissue mask from preprocessing
from ..preprocessing.tissue_mask import get_tissue_mask


# ======================================================================
# Public API
# ======================================================================

def segment_nuclei(
    slide_path: str,
    level: int = 0,
    tile_size: int = 256,
    overlap: int = 0,
    tissue_mask: Optional[np.ndarray] = None,
    model_path: Optional[str] = None,
    batch_size: int = 16,
    device: str = "cuda",
    min_nucleus_area: int = 10,
    stitch_mode: str = "global",
) -> Tuple[np.ndarray, pd.DataFrame]:
    """
    Segment nuclei in a WSI using HoVer-Net.

    Args:
        slide_path: Path to WSI file.
        level: WSI level to process (0 = max resolution).
        tile_size: Tile dimension in pixels (the HoVer-Net model expects 256).
        overlap: Overlap between adjacent tiles.  Only used when
            ``stitch_mode="tile"`` (the tile-based fallback).  For
            ``"global"`` mode tiles are non-overlapping and predictions
            are stitched before post-processing.
        tissue_mask: Optional binary mask (bool or uint8, any resolution).
            If *None*, one is computed from the slide thumbnail.
        model_path: Path to HoVer-Net checkpoint (full model object, state_dict, or legacy 'desc' format).
        batch_size: Inference batch size.
        device: ``"cuda"`` or ``"cpu"``.
        min_nucleus_area: Discard nuclei smaller than this (pixels²).
        stitch_mode: ``"global"`` (recommended, TACC) or ``"tile"`` (local).

    Returns:
        instance_map: ``np.ndarray`` of shape ``(H, W)`` with unique
            integer labels per nucleus.
        nuclei_df: ``pd.DataFrame`` with per-nucleus properties
            (centroid_x, centroid_y, area, perimeter, …).
    """
    if model_path is None:
        raise ValueError("model_path must be provided")

    inferencer = HoverNetInferencer(
        model_path, device=device, batch_size=batch_size
    )

    slide = OpenSlide(slide_path)
    w, h = slide.level_dimensions[level]

    # ---- Tissue mask ----
    tissue_mask = _resolve_tissue_mask(slide, tissue_mask, w, h)
    mask_scale_x = w / tissue_mask.shape[1]
    mask_scale_y = h / tissue_mask.shape[0]

    # ---- Dispatch to stitching strategy ----
    if stitch_mode == "global":
        instance_map = _stitch_global(
            slide, level, w, h, tile_size,
            tissue_mask, mask_scale_x, mask_scale_y,
            inferencer, batch_size,
        )
    elif stitch_mode == "tile":
        instance_map = _stitch_tile(
            slide, level, w, h, tile_size, overlap,
            tissue_mask, mask_scale_x, mask_scale_y,
            inferencer, batch_size,
        )
    else:
        raise ValueError(f"Unknown stitch_mode: {stitch_mode!r}")

    # ---- Extract per-nucleus properties from global instance map ----
    nuclei_df = _instance_map_to_df(instance_map, min_nucleus_area)

    return instance_map, nuclei_df


# ======================================================================
# Global stitching (faithful to original)
# ======================================================================

def _stitch_global(
    slide: OpenSlide,
    level: int,
    w: int,
    h: int,
    tile_size: int,
    tissue_mask: np.ndarray,
    mask_scale_x: float,
    mask_scale_y: float,
    inferencer: HoverNetInferencer,
    batch_size: int,
) -> np.ndarray:
    """
    Stitch raw HoVer-Net outputs into a global prediction tensor of
    shape ``(H, W, 3)`` and run ``process_instance()`` once.
    """
    pred_map = np.zeros((h, w, 3), dtype=np.float32)

    tile_buffer: List[np.ndarray] = []
    coord_buffer: List[Tuple[int, int, int, int]] = []

    # Non-overlapping tiles — the model outputs are stitched directly
    iterator = list(tile_iterator(w, h, tile_size, overlap=0))

    for x, y, tw, th in tqdm(iterator, desc="Inferencing tiles"):
        if not _tile_has_tissue(
            x, y, tw, th, tissue_mask, mask_scale_x, mask_scale_y
        ):
            continue

        tile = slide.read_region((x, y), level, (tw, th))
        tile = np.array(tile.convert("RGB"))

        tile_buffer.append(tile)
        coord_buffer.append((x, y, tw, th))

        if len(tile_buffer) >= batch_size:
            _flush_raw_buffer(
                inferencer, tile_buffer, coord_buffer, pred_map
            )
            tile_buffer.clear()
            coord_buffer.clear()

    # Remaining
    if tile_buffer:
        _flush_raw_buffer(
            inferencer, tile_buffer, coord_buffer, pred_map
        )

    # Single global post-processing (same as original)
    instance_map = process_instance(pred_map)
    return instance_map


def _flush_raw_buffer(
    inferencer: HoverNetInferencer,
    tiles: List[np.ndarray],
    coords: List[Tuple[int, int, int, int]],
    pred_map: np.ndarray,
) -> None:
    """Run batch inference and place raw predictions into *pred_map*."""
    raw_preds = inferencer.predict_batch(tiles, return_raw=True)
    for raw, (tx, ty, tw, th) in zip(raw_preds, coords):
        # raw has shape (th_actual, tw_actual, 3)
        rh, rw = raw.shape[:2]
        pred_map[ty : ty + rh, tx : tx + rw, :] = raw


# ======================================================================
# Tile-based fallback (for local / low-RAM testing)
# ======================================================================

def _stitch_tile(
    slide: OpenSlide,
    level: int,
    w: int,
    h: int,
    tile_size: int,
    overlap: int,
    tissue_mask: np.ndarray,
    mask_scale_x: float,
    mask_scale_y: float,
    inferencer: HoverNetInferencer,
    batch_size: int,
) -> np.ndarray:
    """
    Legacy tile-based pipeline: post-process each tile independently,
    then create a global instance map by painting each nucleus into the
    full canvas and de-duplicating via KD-tree.
    """
    from scipy.spatial import cKDTree
    from skimage.measure import regionprops

    all_nuclei: List[Dict] = []

    tile_buffer: List[np.ndarray] = []
    coord_buffer: List[Tuple[int, int, int, int]] = []

    iterator = list(tile_iterator(w, h, tile_size, overlap))

    for x, y, tw, th in tqdm(iterator, desc="Processing tiles (tile mode)"):
        if not _tile_has_tissue(
            x, y, tw, th, tissue_mask, mask_scale_x, mask_scale_y
        ):
            continue

        tile = slide.read_region((x, y), level, (tw, th))
        tile = np.array(tile.convert("RGB"))

        tile_buffer.append(tile)
        coord_buffer.append((x, y, tw, th))

        if len(tile_buffer) >= batch_size:
            _process_tile_buffer(
                inferencer, tile_buffer, coord_buffer, all_nuclei
            )
            tile_buffer.clear()
            coord_buffer.clear()

    if tile_buffer:
        _process_tile_buffer(
            inferencer, tile_buffer, coord_buffer, all_nuclei
        )

    # Build global instance map from collected nuclei
    instance_map = np.zeros((h, w), dtype=np.int32)
    if all_nuclei:
        df = pd.DataFrame(all_nuclei)
        df = _remove_duplicates(df, distance_threshold=8.0)

        # We don't have per-pixel masks stored, so just return an
        # empty instance_map – the df is the primary output in tile mode.
        # Callers that need the instance map should use global mode.
        # (We still assign unique IDs to centroids for downstream use.)
        for idx, row in enumerate(df.itertuples(), start=1):
            cx = int(round(row.centroid_x))
            cy = int(round(row.centroid_y))
            if 0 <= cy < h and 0 <= cx < w:
                instance_map[cy, cx] = idx

    return instance_map


def _process_tile_buffer(
    inferencer: HoverNetInferencer,
    tiles: List[np.ndarray],
    coords: List[Tuple[int, int, int, int]],
    all_nuclei: List[Dict],
) -> None:
    """Post-process each tile and accumulate nuclei dicts."""
    from skimage.measure import regionprops

    results = inferencer.predict_batch(tiles, return_raw=False)
    for (inst_map, _centroids), (tx, ty, tw, th) in zip(results, coords):
        props = regionprops(inst_map)
        for prop in props:
            cy, cx = prop.centroid
            all_nuclei.append(
                {
                    "centroid_x": tx + cx,
                    "centroid_y": ty + cy,
                    "area": prop.area,
                    "perimeter": prop.perimeter,
                    "eccentricity": prop.eccentricity,
                    "solidity": prop.solidity,
                    "orientation": prop.orientation,
                }
            )


def _remove_duplicates(
    nuclei_df: pd.DataFrame,
    distance_threshold: float = 8.0,
) -> pd.DataFrame:
    """Remove duplicate nuclei based on centroid distance (tile mode only)."""
    from scipy.spatial import cKDTree

    if nuclei_df.empty:
        return nuclei_df

    nuclei_df = nuclei_df.sort_values("area", ascending=False).reset_index(
        drop=True
    )
    points = nuclei_df[["centroid_x", "centroid_y"]].values
    tree = cKDTree(points)
    pairs = tree.query_pairs(r=distance_threshold)

    to_drop = set()
    for i, j in pairs:
        if i not in to_drop:
            to_drop.add(j)

    return nuclei_df.drop(index=list(to_drop)).reset_index(drop=True)


# ======================================================================
# Shared helpers
# ======================================================================

def _resolve_tissue_mask(
    slide: OpenSlide,
    tissue_mask: Optional[np.ndarray],
    w: int,
    h: int,
) -> np.ndarray:
    """Ensure we have a bool tissue mask at some resolution."""
    # Handle tuple output from get_tissue_mask
    if isinstance(tissue_mask, (tuple, list)):
        if len(tissue_mask) == 2 and tissue_mask[1] is not None:
            tissue_mask = tissue_mask[1]
        elif len(tissue_mask) >= 1:
            tissue_mask = tissue_mask[0]

    if tissue_mask is None:
        downsample = max(w // 2048, 1)
        thumb = slide.get_thumbnail((w // downsample, h // downsample))
        thumb_np = np.array(thumb)
        _, tissue_mask = get_tissue_mask(thumb_np)

    if tissue_mask is None:
        # Fallback: treat full slide as tissue
        return np.ones((h, w), dtype=bool)

    tissue_mask = np.asarray(tissue_mask)
    if tissue_mask.ndim == 3:
        tissue_mask = tissue_mask[:, :, 0]
    return tissue_mask.astype(bool)


def _tile_has_tissue(
    x: int,
    y: int,
    tw: int,
    th: int,
    tissue_mask: np.ndarray,
    mask_scale_x: float,
    mask_scale_y: float,
    min_tissue_frac: float = 0.10,
) -> bool:
    """Return True if the tile overlaps enough tissue."""
    mx = max(0, int(x / mask_scale_x))
    my = max(0, int(y / mask_scale_y))
    mw = min(int(tw / mask_scale_x), tissue_mask.shape[1] - mx)
    mh = min(int(th / mask_scale_y), tissue_mask.shape[0] - my)
    if mw <= 0 or mh <= 0:
        return False
    region = tissue_mask[my : my + mh, mx : mx + mw]
    return np.mean(region) >= min_tissue_frac


def _instance_map_to_df(
    instance_map: np.ndarray,
    min_nucleus_area: int = 10,
) -> pd.DataFrame:
    """Extract per-nucleus properties from a labelled instance map."""
    from skimage.measure import regionprops

    records: List[Dict] = []
    for prop in regionprops(instance_map):
        if prop.area < min_nucleus_area:
            continue
        cy, cx = prop.centroid
        records.append(
            {
                "nucleus_id": prop.label,
                "centroid_x": cx,
                "centroid_y": cy,
                "area": prop.area,
                "perimeter": prop.perimeter,
                "eccentricity": prop.eccentricity,
                "equivalent_diameter": prop.equivalent_diameter,
                "euler_number": prop.euler_number,
                "extent": prop.extent,
                "filled_area": prop.filled_area,
                "major_axis_length": prop.major_axis_length,
                "minor_axis_length": prop.minor_axis_length,
                "orientation": prop.orientation,
                "solidity": prop.solidity,
            }
        )
    return pd.DataFrame(records)
