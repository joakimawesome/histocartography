import cv2
import numpy as np
from scipy.ndimage import label, binary_fill_holes
from skimage.morphology import remove_small_objects
from skimage.segmentation import watershed

def process_instance(
    pred_map: np.ndarray,
    output_dtype: str = "uint16"
) -> np.ndarray:
    """
    Post processing script for image tiles.
    Args:
        pred_map: (H, W, 3) or (H, W, 2) output from HoVer-Net.
                  Channel 0: Probability map.
                  Channel 1: Horizontal map.
                  Channel 2: Vertical map.
        output_dtype: Data type of output instance map.
    Returns:
        Instance map (H, W) where each nucleus has a unique integer ID.
    """
    pred_inst = np.squeeze(pred_map)
    pred_inst = process_np_hv_channels(pred_inst)
    pred_inst = pred_inst.astype(output_dtype)
    return pred_inst

def process_np_hv_channels(pred: np.ndarray) -> np.ndarray:
    """
    Process Nuclei Prediction with XY Coordinate Map
    Args:
        pred: (H, W, 3) HoVer-Net output.
    Returns:
        Instance map (H, W).
    """
    # 1. Post-process probability map
    proba_map = np.copy(pred[:, :, 0])
    proba_map[proba_map >= 0.5] = 1
    proba_map[proba_map < 0.5] = 0
    
    # Label independent connected components
    proba_map_labeled, _ = label(proba_map)
    
    # Remove small objects (noise)
    proba_map_labeled = remove_small_objects(proba_map_labeled, min_size=10)
    # Convert back to binary mask
    proba_map = (proba_map_labeled > 0).astype(np.uint8)

    # 2. Extract and Normalize HV maps
    if pred.shape[-1] >= 3:
        h_dir = pred[:, :, 1]
        v_dir = pred[:, :, 2]
    else:
        # If only 2 channels, we can't do watershed based on HV. 
        # Fallback to simple connected components from probability map.
        return proba_map_labeled.astype(np.int32)

    # Normalize to [0, 1]
    h_dir = cv2.normalize(h_dir, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    v_dir = cv2.normalize(v_dir, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

    # 3. Sobel Filtering to find gradients
    sobelh = cv2.Sobel(h_dir, cv2.CV_64F, 1, 0, ksize=21)
    sobelv = cv2.Sobel(v_dir, cv2.CV_64F, 0, 1, ksize=21)

    sobelh = 1 - (cv2.normalize(sobelh, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F))
    sobelv = 1 - (cv2.normalize(sobelv, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F))

    # 4. Marker generation for Watershed
    overall = np.maximum(sobelh, sobelv)
    overall = overall - (1 - proba_map)
    overall[overall < 0] = 0

    dist = (1.0 - overall) * proba_map
    dist = -cv2.GaussianBlur(dist, (3, 3), 0)

    overall[overall >= 0.5] = 1
    overall[overall < 0.5] = 0
    
    marker = proba_map - overall
    marker[marker < 0] = 0
    marker = binary_fill_holes(marker).astype("uint8")
    
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    marker = cv2.morphologyEx(marker, cv2.MORPH_OPEN, kernel)
    
    marker_labeled, _ = label(marker)
    marker_labeled = remove_small_objects(marker_labeled, min_size=10)

    # 5. Watershed
    pred_inst = watershed(dist, marker_labeled, mask=proba_map, watershed_line=False)

    return pred_inst
