"""
Per-node handcrafted feature extraction.

Faithfully reproduces the original HistoCartography
``HandcraftedFeatureExtractor`` which computes **24** features per node:

    16 shape  ·  6 GLCM texture  ·  2 crowdedness

The main entry point is :func:`extract_handcrafted_node_features`.
"""

from __future__ import annotations

import warnings
from typing import List, Optional, Tuple

import cv2
import numpy as np
import torch
from scipy.stats import skew
from skimage.feature import graycomatrix, graycoprops
from skimage.measure import regionprops
from sklearn.metrics.pairwise import euclidean_distances


# ======================================================================
# Public API
# ======================================================================

def extract_handcrafted_node_features(
    input_image: np.ndarray,
    instance_map: np.ndarray,
) -> torch.Tensor:
    """
    Extract 24 handcrafted features for every nucleus instance.

    This is a drop-in replacement for the original
    ``HandcraftedFeatureExtractor._extract_features``.

    Args:
        input_image: RGB image, uint8, shape ``(H, W, 3)``.
        instance_map: Labelled instance map, int, shape ``(H, W)``.
            Background is 0; each nucleus has a unique positive label.

    Returns:
        ``torch.Tensor`` of shape ``(num_nodes, 24)``.
    """
    img_gray = cv2.cvtColor(input_image, cv2.COLOR_RGB2GRAY)
    img_square = np.square(input_image.astype(np.float64))

    regions = regionprops(instance_map)
    if len(regions) == 0:
        return torch.empty(0, 24)

    # Pre-compute crowdedness for all nodes
    centroids = [r.centroid for r in regions]
    mean_crowdedness, std_crowdedness = _compute_crowdedness(centroids)

    node_feats: list[np.ndarray] = []

    for region_id, region in enumerate(regions):
        # ---- Bounding-box crops ----
        r0, c0, r1, c1 = region.bbox
        sp_mask = (
            instance_map[r0:r1, c0:c1] == region.label
        )
        sp_gray = img_gray[r0:r1, c0:c1] * sp_mask

        # ---- Shape features (16) ----
        feats_shape = _shape_features(region, sp_mask)

        # ---- GLCM texture features (6) ----
        feats_texture = _glcm_features(sp_gray)

        # ---- Crowdedness features (2) ----
        feats_crowdedness = [
            float(mean_crowdedness[region_id].item()),
            float(std_crowdedness[region_id].item()),
        ]

        feats = np.hstack(feats_shape + feats_texture + feats_crowdedness)
        node_feats.append(feats)

    return torch.tensor(np.vstack(node_feats), dtype=torch.float32)


# ======================================================================
# Feature groups
# ======================================================================

def _shape_features(region, sp_mask: np.ndarray) -> List[float]:
    """16 shape descriptors — identical to original."""
    area = region.area
    convex_area = getattr(region, 'area_convex', getattr(region, 'convex_area', 0))
    eccentricity = region.eccentricity
    equivalent_diameter = getattr(region, 'equivalent_diameter_area', getattr(region, 'equivalent_diameter', 0))
    euler_number = region.euler_number
    extent = region.extent
    filled_area = getattr(region, 'area_filled', getattr(region, 'filled_area', 0))
    major_axis_length = getattr(region, 'axis_major_length', getattr(region, 'major_axis_length', 0))
    minor_axis_length = getattr(region, 'axis_minor_length', getattr(region, 'minor_axis_length', 0))
    orientation = region.orientation
    perimeter = region.perimeter
    solidity = region.solidity

    convex_hull_perimeter = _convex_hull_perimeter(sp_mask)

    # Guard against zero-division
    if perimeter == 0:
        roughness = 0.0
        roundness = 0.0
    else:
        roughness = convex_hull_perimeter / perimeter
        roundness = (4 * np.pi * area) / (perimeter ** 2)

    if convex_hull_perimeter == 0:
        shape_factor = 0.0
    else:
        shape_factor = 4 * np.pi * area / convex_hull_perimeter ** 2

    if major_axis_length == 0:
        ellipticity = 0.0
    else:
        ellipticity = minor_axis_length / major_axis_length

    return [
        area,
        convex_area,
        eccentricity,
        equivalent_diameter,
        euler_number,
        extent,
        filled_area,
        major_axis_length,
        minor_axis_length,
        orientation,
        perimeter,
        solidity,
        roughness,
        shape_factor,
        ellipticity,
        roundness,
    ]


def _glcm_features(sp_gray: np.ndarray) -> List[float]:
    """6 GLCM texture features — identical to original."""
    try:
        glcm = graycomatrix(sp_gray, [1], [0])
        # Filter out zero-row / zero-col (background)
        filt_glcm = glcm[1:, 1:, :, :]

        contrast = float(graycoprops(filt_glcm, prop="contrast")[0, 0])
        dissimilarity = float(graycoprops(filt_glcm, prop="dissimilarity")[0, 0])
        homogeneity = float(graycoprops(filt_glcm, prop="homogeneity")[0, 0])
        energy = float(graycoprops(filt_glcm, prop="energy")[0, 0])
        asm = float(graycoprops(filt_glcm, prop="ASM")[0, 0])
        dispersion = float(np.std(filt_glcm))
    except Exception:
        # Degenerate region
        contrast = dissimilarity = homogeneity = energy = asm = dispersion = 0.0

    return [contrast, dissimilarity, homogeneity, energy, asm, dispersion]


def _compute_crowdedness(
    centroids: list,
    k: int = 10,
) -> Tuple[np.ndarray, np.ndarray]:
    """Mean and std of distances to k nearest neighbours — identical to original."""
    n = len(centroids)
    if n < 3:
        return np.zeros((n, 1)), np.zeros((n, 1))

    if n < k + 2:
        k = n - 2

    dist = euclidean_distances(centroids, centroids)
    idx = np.argpartition(dist, kth=k + 1, axis=-1)
    x = np.take_along_axis(dist, idx, axis=-1)[:, : k + 1]
    mean_crow = np.mean(x, axis=1)
    std_crow = np.std(x, axis=1)
    return mean_crow, std_crow


def _convex_hull_perimeter(sp_mask: np.ndarray) -> float:
    """Perimeter of the convex hull of *sp_mask* — identical to original."""
    contours, _ = cv2.findContours(
        np.uint8(sp_mask), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
    )
    if len(contours) == 0:
        return 0.0
    hull = cv2.convexHull(contours[0])
    return float(cv2.arcLength(hull, True))
