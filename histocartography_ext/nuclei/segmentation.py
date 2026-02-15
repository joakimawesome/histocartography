"""
Nuclei segmentation from whole-slide images using HoVer-Net.

Supports two stitching strategies controlled by ``stitch_mode``:

* ``"global"`` (default) – stitch raw HoVer-Net prediction maps into a
  single global tensor, then run post-processing once.  This matches the
  original HistoCartography ``NucleiExtractor`` and avoids boundary
  artefacts.  Requires enough RAM to hold the full prediction tensor
  (approx ``H * W * 3 * 4`` bytes in float32).

* ``"tile"`` – post-process each tile independently and stitch per-tile
  instance maps into a global canvas using centroid-based ownership in
  overlap regions. This avoids running watershed on the full-slide tensor
  (which scales with the entire WSI area, including blank background).
"""

import math
import logging
import numpy as np
import pandas as pd
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

    logger = logging.getLogger(__name__)
    inferencer = HoverNetInferencer(model_path, device=device, batch_size=batch_size)

    slide = OpenSlide(slide_path)
    try:
        w, h = slide.level_dimensions[level]

        # ---- Tissue mask ----
        tissue_mask = _resolve_tissue_mask(slide, tissue_mask, w, h)
        mask_scale_x = w / tissue_mask.shape[1]
        mask_scale_y = h / tissue_mask.shape[0]

        logger.info(
            "Segmentation config: stitch_mode=%s level=%s tile=%s overlap=%s batch=%s device=%s",
            stitch_mode,
            level,
            tile_size,
            overlap,
            batch_size,
            device,
        )
        logger.info(
            "Slide level dims: %s x %s; tissue mask: %s x %s",
            w,
            h,
            tissue_mask.shape[1],
            tissue_mask.shape[0],
        )

        # ---- Dispatch to stitching strategy ----
        if stitch_mode == "global":
            instance_map = _stitch_global(
                slide, level, w, h, tile_size, tissue_mask, mask_scale_x, mask_scale_y, inferencer, batch_size
            )
            nuclei_df = _instance_map_to_df(instance_map, min_nucleus_area)
            return instance_map, nuclei_df

        if stitch_mode == "tile":
            instance_map, nuclei_df = _stitch_tile(
                slide, level, w, h, tile_size, overlap, tissue_mask, mask_scale_x, mask_scale_y, inferencer, batch_size,
                min_nucleus_area=min_nucleus_area
            )
            return instance_map, nuclei_df

        raise ValueError(f"Unknown stitch_mode: {stitch_mode!r}")
    finally:
        try:
            slide.close()
        except Exception:
            pass

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
    logger = logging.getLogger(__name__)
    # Crop work to tissue bounding box (with margin) to:
    #   - drastically reduce RAM (avoid allocating full-slide pred_map)
    #   - avoid iterating tiles over large blank backgrounds
    y0, y1, x0, x1 = _tissue_bbox_from_mask(
        tissue_mask,
        mask_scale_x=mask_scale_x,
        mask_scale_y=mask_scale_y,
        w=w,
        h=h,
        margin=tile_size,
    )
    # Align crop to the tiling grid so that (x, y) tile origins map cleanly.
    x0a = max((x0 // tile_size) * tile_size, 0)
    y0a = max((y0 // tile_size) * tile_size, 0)
    x1a = min(w, ((x1 + tile_size - 1) // tile_size) * tile_size)
    y1a = min(h, ((y1 + tile_size - 1) // tile_size) * tile_size)

    pred_map = np.zeros((y1a - y0a, x1a - x0a, 3), dtype=np.float32)

    tile_buffer: List[np.ndarray] = []
    coord_buffer: List[Tuple[int, int, int, int]] = []

    # Non-overlapping tiles — the model outputs are stitched directly
    step = tile_size
    tiles_x = (max(0, x1a - x0a) + step - 1) // step
    tiles_y = (max(0, y1a - y0a) + step - 1) // step
    total_tiles = tiles_x * tiles_y
    logger.info(
        "Global stitch crop: x=[%s,%s) y=[%s,%s) pred_map=%sx%s tiles=%s",
        x0a,
        x1a,
        y0a,
        y1a,
        pred_map.shape[1],
        pred_map.shape[0],
        total_tiles,
    )

    def _iter_crop_tiles():
        for y in range(y0a, y1a, step):
            for x in range(x0a, x1a, step):
                tw = min(tile_size, w - x)
                th = min(tile_size, h - y)
                yield x, y, tw, th

    processed = 0
    skipped = 0
    for x, y, tw, th in tqdm(_iter_crop_tiles(), total=total_tiles, desc="Inferencing tiles"):
        if not _tile_has_tissue(
            x, y, tw, th, tissue_mask, mask_scale_x, mask_scale_y
        ):
            skipped += 1
            continue

        tile = slide.read_region((x, y), level, (tw, th))
        tile = np.array(tile.convert("RGB"))

        tile_buffer.append(tile)
        coord_buffer.append((x, y, tw, th))
        processed += 1
        if processed % 500 == 0:
            logger.info("Global stitch progress: processed=%s skipped=%s", processed, skipped)

        if len(tile_buffer) >= batch_size:
            _flush_raw_buffer(inferencer, tile_buffer, coord_buffer, pred_map, x_offset=x0a, y_offset=y0a)
            tile_buffer.clear()
            coord_buffer.clear()

    # Remaining
    if tile_buffer:
        _flush_raw_buffer(inferencer, tile_buffer, coord_buffer, pred_map, x_offset=x0a, y_offset=y0a)

    # Single global post-processing (same as original)
    #
    # `process_instance()` is CPU-only and scales with pixel area. On WSIs, doing
    # this over the full (H, W) canvas wastes time on blank background. Crop to
    # the tissue mask's bounding box (with a safety margin) first.
    #
    # NOTE: WSI nucleus counts can exceed 65k, so uint16 will overflow.
    inst_crop = process_instance(pred_map, output_dtype="uint32")
    instance_map = np.zeros((h, w), dtype=np.uint32)
    instance_map[y0a:y1a, x0a:x1a] = inst_crop
    logger.info("Global stitch complete: processed=%s skipped=%s", processed, skipped)
    return instance_map


def _flush_raw_buffer(
    inferencer: HoverNetInferencer,
    tiles: List[np.ndarray],
    coords: List[Tuple[int, int, int, int]],
    pred_map: np.ndarray,
    *,
    x_offset: int = 0,
    y_offset: int = 0,
) -> None:
    """Run batch inference and place raw predictions into *pred_map*."""
    raw_preds = inferencer.predict_batch(tiles, return_raw=True)
    for raw, (tx, ty, tw, th) in zip(raw_preds, coords):
        # raw has shape (th_actual, tw_actual, 3)
        rh, rw = raw.shape[:2]
        oy = ty - y_offset
        ox = tx - x_offset
        pred_map[oy : oy + rh, ox : ox + rw, :] = raw

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
    *,
    min_nucleus_area: int,
) -> Tuple[np.ndarray, pd.DataFrame]:
    """
    Tile-based pipeline: post-process each tile independently, then stitch the
    per-tile instance maps into a global canvas.

    Overlap handling uses centroid-based ownership: each nucleus is assigned to
    the tile whose non-overlapping valid region contains its centroid. This
    avoids duplicate labels across overlaps while still allowing us to paint
    full masks.
    """
    logger = logging.getLogger(__name__)
    instance_map = np.zeros((h, w), dtype=np.uint32)
    records: List[Dict[str, Any]] = []
    next_label: int = 1

    tile_buffer: List[np.ndarray] = []
    coord_buffer: List[Tuple[int, int, int, int]] = []

    step = tile_size - overlap
    total_tiles = ((w + step - 1) // step) * ((h + step - 1) // step)
    iterator = tile_iterator(w, h, tile_size, overlap)

    processed = 0
    skipped = 0
    for x, y, tw, th in tqdm(iterator, total=total_tiles, desc="Processing tiles (tile mode)"):
        if not _tile_has_tissue(
            x, y, tw, th, tissue_mask, mask_scale_x, mask_scale_y
        ):
            skipped += 1
            continue

        tile = slide.read_region((x, y), level, (tw, th))
        tile = np.array(tile.convert("RGB"))

        tile_buffer.append(tile)
        coord_buffer.append((x, y, tw, th))
        processed += 1
        if processed % 500 == 0:
            logger.info("Tile stitch progress: processed=%s skipped=%s", processed, skipped)

        if len(tile_buffer) >= batch_size:
            next_label = _stitch_tile_buffer(
                inferencer, tile_buffer, coord_buffer, instance_map, records, next_label,
                slide_w=w, slide_h=h, overlap=overlap, min_nucleus_area=min_nucleus_area
            )
            tile_buffer.clear()
            coord_buffer.clear()

    if tile_buffer:
        next_label = _stitch_tile_buffer(
            inferencer, tile_buffer, coord_buffer, instance_map, records, next_label,
            slide_w=w, slide_h=h, overlap=overlap, min_nucleus_area=min_nucleus_area
        )

    # Filter small nuclei at the stitching stage to keep instance IDs stable.
    # (If we drop later, we'd need to relabel the instance map.)
    nuclei_df = pd.DataFrame.from_records(records)
    logger.info("Tile stitch complete: processed=%s skipped=%s", processed, skipped)
    return instance_map, nuclei_df


def _tile_valid_window(
    tx: int,
    ty: int,
    tw: int,
    th: int,
    overlap: int,
    *,
    slide_w: int,
    slide_h: int,
) -> Tuple[int, int, int, int]:
    """Compute a non-overlapping \"valid\" window in tile-local coordinates."""
    if overlap <= 0:
        return 0, 0, tw, th

    left_margin = overlap // 2 if tx > 0 else 0
    top_margin = overlap // 2 if ty > 0 else 0
    right_margin = overlap - overlap // 2 if (tx + tw) < slide_w else 0
    bottom_margin = overlap - overlap // 2 if (ty + th) < slide_h else 0

    x0 = left_margin
    y0 = top_margin
    x1 = tw - right_margin
    y1 = th - bottom_margin

    # Edge tiles can be smaller than overlap margins; fall back to full tile.
    if x1 <= x0:
        x0, x1 = 0, tw
    if y1 <= y0:
        y0, y1 = 0, th

    return x0, y0, x1, y1


def _stitch_tile_buffer(
    inferencer: HoverNetInferencer,
    tiles: List[np.ndarray],
    coords: List[Tuple[int, int, int, int]],
    instance_map: np.ndarray,
    records: List[Dict[str, Any]],
    next_label: int,
    *,
    slide_w: int,
    slide_h: int,
    overlap: int,
    min_nucleus_area: int,
) -> int:
    """Post-process each tile and stitch into the global instance map."""
    from skimage.measure import regionprops

    results = inferencer.predict_batch(tiles, return_raw=False)
    for (inst_tile, _centroids), (tx, ty, tw, th) in zip(results, coords):
        inst_tile = np.asarray(inst_tile, dtype=np.uint32)
        if inst_tile.max() == 0:
            continue

        x0, y0, x1, y1 = _tile_valid_window(
            tx, ty, tw, th, overlap, slide_w=slide_w, slide_h=slide_h
        )

        props = regionprops(inst_tile)
        if len(props) == 0:
            continue

        max_label = int(inst_tile.max())
        label_map = np.zeros((max_label + 1,), dtype=np.uint32)

        for prop in props:
            if prop.area < min_nucleus_area:
                continue

            cy, cx = prop.centroid  # (row, col)
            if not (x0 <= cx < x1 and y0 <= cy < y1):
                continue

            gid = next_label
            next_label += 1
            label_map[prop.label] = gid

            records.append(
                {
                    "nucleus_id": gid,
                    "centroid_x": tx + cx,
                    "centroid_y": ty + cy,
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

        if not np.any(label_map):
            continue

        mapped = label_map[inst_tile]
        target = instance_map[ty : ty + th, tx : tx + tw]
        write_mask = mapped > 0
        if overlap > 0:
            write_mask &= (target == 0)
        target[write_mask] = mapped[write_mask]

    return next_label

def _tissue_bbox_from_mask(
    tissue_mask: np.ndarray,
    *,
    mask_scale_x: float,
    mask_scale_y: float,
    w: int,
    h: int,
    margin: int,
) -> Tuple[int, int, int, int]:
    """
    Compute a conservative level-space bounding box (y0, y1, x0, x1) that
    contains tissue according to a (possibly downsampled) tissue mask.
    """
    tissue_mask = np.asarray(tissue_mask).astype(bool)
    ys, xs = np.nonzero(tissue_mask)
    if ys.size == 0:
        return 0, h, 0, w

    x0 = int(math.floor(xs.min() * mask_scale_x))
    x1 = int(math.ceil((xs.max() + 1) * mask_scale_x))
    y0 = int(math.floor(ys.min() * mask_scale_y))
    y1 = int(math.ceil((ys.max() + 1) * mask_scale_y))

    x0 = max(0, x0 - margin)
    y0 = max(0, y0 - margin)
    x1 = min(w, x1 + margin)
    y1 = min(h, y1 + margin)

    if x1 <= x0 or y1 <= y0:
        return 0, h, 0, w
    return y0, y1, x0, x1


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
