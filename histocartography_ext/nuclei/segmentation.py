import os
import numpy as np
import pandas as pd
import openslide
from openslide import OpenSlide
from tqdm import tqdm
from typing import Optional, List, Dict, Any
from shapely.geometry import Polygon

try:
    from .utils import get_tissue_mask, tile_iterator, remove_duplicates
    from .inference import HoverNetInferencer
except ImportError:
    pass

def segment_nuclei(
    slide_path: str,
    level: int = 0,
    tile_size: int = 1024,
    overlap: int = 256,
    tissue_mask: Optional[np.ndarray] = None,
    model_path: Optional[str] = None,
    batch_size: int = 4,
    device: str = 'cuda',
    min_nucleus_area: int = 10
) -> pd.DataFrame:
    """
    Segment nuclei in a WSI using HoVer-Net.
    
    Args:
        slide_path: Path to WSI file.
        level: WSI level to process (usually 0 for max res).
        tile_size: Size of tiles to process.
        overlap: Overlap between tiles.
        tissue_mask: Optional binary mask to restrict processing.
        model_path: Path to HoVer-Net checkpoint.
        batch_size: Batch size for inference.
        device: 'cuda' or 'cpu'.
        min_nucleus_area: Filter small nuclei.
        
    Returns:
        DataFrame with nuclei features.
    """
    
    # 1. Initialize Model
    if model_path is None:
        # Try to find default model
        # For now, let's assume user must provide it or we look in checkpoints
        raise ValueError("model_path must be provided")
        
    inferencer = HoverNetInferencer(model_path, device=device, batch_size=batch_size)
    
    # 2. Open Slide
    slide = OpenSlide(slide_path)
    w, h = slide.level_dimensions[level]
    
    # 3. Tissue Mask
    # If not provided, compute from thumbnail. If provided, assume it's a full-slide
    # mask at any resolution (e.g. thumbnail) and infer scale factors accordingly.
    if isinstance(tissue_mask, (tuple, list)):
        # Allow passing outputs from other tissue-mask utilities that return
        # (labeled_regions, binary_mask).
        if len(tissue_mask) == 2 and tissue_mask[1] is not None:
            tissue_mask = tissue_mask[1]
        elif len(tissue_mask) >= 1:
            tissue_mask = tissue_mask[0]

    if tissue_mask is None:
        # Get thumbnail at a reasonable downsample (target ~1024-2048 width)
        downsample = max(w // 2048, 1)
        thumb = slide.get_thumbnail((w // downsample, h // downsample))
        thumb_np = np.array(thumb)
        tissue_mask = get_tissue_mask(thumb_np)

    tissue_mask = np.asarray(tissue_mask)
    if tissue_mask.ndim == 3:
        tissue_mask = tissue_mask[:, :, 0]
    tissue_mask = tissue_mask.astype(bool)

    if tissue_mask.shape[0] == 0 or tissue_mask.shape[1] == 0:
        raise ValueError("tissue_mask must have non-zero height and width")

    mask_scale_x = w / tissue_mask.shape[1]
    mask_scale_y = h / tissue_mask.shape[0]

    # 4. Iterate Tiles
    all_nuclei = []
    
    # We can batch tiles, but reading them from OpenSlide is sequential usually.
    # To use batch inference, we'd need a buffer.
    
    tile_buffer = []
    coord_buffer = []
    
    iterator = tile_iterator(w, h, tile_size, overlap)
    
    def process_buffer(tiles, coords):
        results = inferencer.predict_batch(tiles)
        batch_nuclei = []
        for (inst_map, centroids), (tx, ty, tw, th) in zip(results, coords):
            # Centroids are in tile coordinates (row, col) = (y, x) ???
            # Wait, regionprops in inference returns (y, x), but I flipped it to (x, y).
            # So centroids are (x, y) relative to tile top-left.
            
            # Global coordinates
            if len(centroids) == 0:
                continue
                
            global_centroids = centroids + np.array([tx, ty])
            
            # Extract properties
            # We need to re-run regionprops or extract from what we have?
            # Inference only returned centroids and map. 
            # If we want area, perimeter, etc, we should do it in inference or here.
            # Let's move property extraction to here or inference.
            # Inference `predict_batch` currently returns (instance_map, centroids).
            # We need the instance map to compute area/perimeter.
            
            from skimage.measure import regionprops
            props = regionprops(inst_map)
            
            for prop in props:
                if prop.area < min_nucleus_area:
                    continue
                
                # Centroid (global)
                cy, cx = prop.centroid
                gx = tx + cx
                gy = ty + cy
                
                # Bounding box check for border removal (optional)
                # If nucleus touches tile border and we are not at slide border, we might discard it?
                # Or relying on stitching to handle duplicates.
                # Standard approach: discard objects touching the "inner" border of the tile (the overlap region),
                # UNLESS it's the edge of the slide.
                # Simplest: Keep everything, dedup later.
                
                # Contour
                # Getting exact contour points is expensive. 
                # prop.coords gives pixel list. 
                # Let's store centroid and area first.
                
                # We can store a WKT polygon ideally.
                # cv2.findContours on the instance mask == prop.label
                # Create a mask for this object
                # obj_mask = (inst_map == prop.label).astype(np.uint8)
                # ... too slow for millions of nuclei.
                
                # Let's stick to basic props for now.
                batch_nuclei.append({
                    'centroid_x': gx, 
                    'centroid_y': gy,
                    'area': prop.area,
                    'perimeter': prop.perimeter,
                    'eccentricity': prop.eccentricity,
                    'solidity': prop.solidity,
                    'orientation': prop.orientation,
                    # 'nucleus_id': ... (assign later)
                })
        return batch_nuclei

    for x, y, tw, th in tqdm(iterator, desc="Processing Tiles"):
        # Check tissue mask
        # Map tile rect to mask coordinates
        mx = int(x / mask_scale_x)
        my = int(y / mask_scale_y)
        mw = int(tw / mask_scale_x)
        mh = int(th / mask_scale_y)
        
        # Simple check: if mask region has enough tissue
        # Clip to mask bounds
        mx = max(0, mx)
        my = max(0, my)
        mw = min(mw, tissue_mask.shape[1] - mx)
        mh = min(mh, tissue_mask.shape[0] - my)
        
        if mw <= 0 or mh <= 0:
            continue
            
        mask_region = tissue_mask[my:my+mh, mx:mx+mw]
        if np.sum(mask_region) < (mw * mh * 0.1): # Less than 10% tissue
            continue
            
        # Read Region
        tile = slide.read_region((x, y), level, (tw, th))
        tile = np.array(tile.convert('RGB'))
        
        tile_buffer.append(tile)
        coord_buffer.append((x, y, tw, th))
        
        if len(tile_buffer) >= batch_size:
            nuclei_data = process_buffer(tile_buffer, coord_buffer)
            all_nuclei.extend(nuclei_data)
            tile_buffer = []
            coord_buffer = []
            
    # Process remaining
    if tile_buffer:
        nuclei_data = process_buffer(tile_buffer, coord_buffer)
        all_nuclei.extend(nuclei_data)
        
    df = pd.DataFrame(all_nuclei)
    
    # Dedup
    if not df.empty:
        df = remove_duplicates(df, distance_threshold=8.0)
        df['nucleus_id'] = range(len(df))
        
    return df
