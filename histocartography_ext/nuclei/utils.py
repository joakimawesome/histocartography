import numpy as np
import cv2
from skimage.color import rgb2hsv
from skimage.filters import threshold_otsu
from shapely.geometry import Polygon
from typing import List, Tuple, Generator, Optional, Dict

def get_tissue_mask(
    thumbnail: np.ndarray,
    min_area: int = 5000,
    use_otsu: bool = True
) -> np.ndarray:
    """
    Generate a binary tissue mask from a thumbnail image.
    Args:
        thumbnail: RGB numpy array (e.g. from slide.get_thumbnail)
        min_area: Minimum area of tissue regions to keep.
        use_otsu: If True use Otsu thresholding on Saturation. Else use simple HSV.
    Returns:
        Binary mask (uint8) where 1 is tissue.
    """
    hsv = rgb2hsv(thumbnail)
    # Saturation channel is usually good for H&E
    saturation = hsv[:, :, 1]
    
    if use_otsu:
        try:
            val = threshold_otsu(saturation)
            mask = saturation > val
        except:
            mask = saturation > 0.05
    else:
        # Simple heuristic
        mask = saturation > 0.05

    mask = mask.astype(np.uint8)
    
    # Cleanup
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    
    # Filter small contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    new_mask = np.zeros_like(mask)
    for c in contours:
        if cv2.contourArea(c) > min_area:
            cv2.drawContours(new_mask, [c], -1, 1, -1)
            
    return new_mask

def tile_iterator(
    slide_width: int,
    slide_height: int,
    tile_size: int = 1024,
    overlap: int = 256
) -> Generator[Tuple[int, int, int, int], None, None]:
    """
    Yields (x, y, w, h) for tiles with overlap.
    """
    step = tile_size - overlap
    for y in range(0, slide_height, step):
        for x in range(0, slide_width, step):
            w = min(tile_size, slide_width - x)
            h = min(tile_size, slide_height - y)
            yield x, y, w, h

import pandas as pd
from shapely.wkt import loads as wkt_loads

def solve_stitch_overlap(
    current_nuclei: pd.DataFrame,
    new_nuclei: pd.DataFrame,
    distance_threshold: float = 10.0
) -> pd.DataFrame:
    """
    Merges new nuclei into current set, resolving duplicates.
    This is a naive global merge. For huge WSIs, this should be done continually or using a spatial index.
    For this implementation, we will assume we append everything and dedup at the end or per-row.
    
    Actually, a better approach for batch processing is to just save all and dedup at the very end 
    using a spatial index, OR dedup locally against the "active frontier".
    
    Let's implement a simple concat for now, and a separate deduplication function.
    """
    if current_nuclei.empty:
        return new_nuclei
    if new_nuclei.empty:
        return current_nuclei
        
    return pd.concat([current_nuclei, new_nuclei], ignore_index=True)

def remove_duplicates(
    nuclei_df: pd.DataFrame,
    distance_threshold: float = 8.0
) -> pd.DataFrame:
    """
    Remove duplicate nuclei based on centroid distance.
    Prioritizes larger area.
    """
    if nuclei_df.empty:
        return nuclei_df
        
    # Sort by area descending so we keep the largest of the duplicates
    nuclei_df = nuclei_df.sort_values('area', ascending=False).reset_index(drop=True)
    
    # Use spatial KDTree for fast lookup
    from scipy.spatial import cKDTree
    points = nuclei_df[['centroid_x', 'centroid_y']].values
    tree = cKDTree(points)
    
    # Query pairs within threshold
    pairs = tree.query_pairs(r=distance_threshold)
    
    to_drop = set()
    for i, j in pairs:
        # i and j are indices in the reset dataframe
        # Since we sorted by area, i < j implies area[i] >= area[j] (usually)
        # So we drop j
        if i not in to_drop:
            to_drop.add(j)
            
    return nuclei_df.drop(index=list(to_drop))


