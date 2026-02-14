"""Unit tests for features/node_features.py (handcrafted per-node features)."""
import unittest
import numpy as np
import torch
import sys
from unittest.mock import MagicMock

# Mock heavy optional deps to avoid import errors
sys.modules.setdefault('dgl', MagicMock())
sys.modules.setdefault('dgl.data.utils', MagicMock())
sys.modules.setdefault('histocartography_ext.preprocessing.feature_extraction', MagicMock())
sys.modules.setdefault('histocartography_ext.preprocessing.graph_builders', MagicMock())
sys.modules.setdefault('histocartography_ext.preprocessing.io', MagicMock())
sys.modules.setdefault('histocartography_ext.preprocessing.nuclei_concept_extraction', MagicMock())
sys.modules.setdefault('histocartography_ext.preprocessing.nuclei_extraction', MagicMock())
sys.modules.setdefault('histocartography_ext.preprocessing.stain_normalizers', MagicMock())
sys.modules.setdefault('histocartography_ext.preprocessing.stats', MagicMock())
sys.modules.setdefault('histocartography_ext.preprocessing.superpixel', MagicMock())
sys.modules.setdefault('histocartography_ext.preprocessing.assignment_matrix', MagicMock())

from histocartography_ext.features.node_features import (
    extract_handcrafted_node_features,
)


def _make_synthetic_data(num_circles: int = 5, img_size: int = 200):
    """Create a synthetic RGB image with circular nuclei and an instance map."""
    rng = np.random.RandomState(42)
    image = np.full((img_size, img_size, 3), 220, dtype=np.uint8)  # light bg
    instance_map = np.zeros((img_size, img_size), dtype=np.int32)

    for i in range(1, num_circles + 1):
        cx = rng.randint(30, img_size - 30)
        cy = rng.randint(30, img_size - 30)
        r = rng.randint(8, 15)
        # Draw on image
        color = tuple(rng.randint(50, 200, 3).tolist())
        cv2.circle(image, (cx, cy), r, color, -1)
        # Draw on instance map
        cv2.circle(instance_map, (cx, cy), r, int(i), -1)

    return image, instance_map


import cv2


class TestHandcraftedNodeFeatures(unittest.TestCase):
    def setUp(self):
        self.image, self.instance_map = _make_synthetic_data(num_circles=5)

    def test_output_shape(self):
        features = extract_handcrafted_node_features(self.image, self.instance_map)
        self.assertIsInstance(features, torch.Tensor)
        self.assertEqual(features.shape[0], 5)  # 5 nuclei
        self.assertEqual(features.shape[1], 24)  # 24 features

    def test_no_nan_or_inf(self):
        features = extract_handcrafted_node_features(self.image, self.instance_map)
        self.assertFalse(torch.isnan(features).any(), "Features contain NaN")
        self.assertFalse(torch.isinf(features).any(), "Features contain Inf")

    def test_empty_instance_map(self):
        empty_map = np.zeros((100, 100), dtype=np.int32)
        image = np.full((100, 100, 3), 200, dtype=np.uint8)
        features = extract_handcrafted_node_features(image, empty_map)
        self.assertEqual(features.shape, (0, 24))

    def test_single_nucleus(self):
        img = np.full((50, 50, 3), 180, dtype=np.uint8)
        imap = np.zeros((50, 50), dtype=np.int32)
        # Draw a circle in the center
        cv2.circle(img, (25, 25), 10, (100, 60, 60), -1)
        cv2.circle(imap, (25, 25), 10, 1, -1)

        features = extract_handcrafted_node_features(img, imap)
        self.assertEqual(features.shape, (1, 24))
        # Area should be > 0
        self.assertGreater(features[0, 0].item(), 0)
        # Solidity of a circle should be close to 1
        self.assertGreater(features[0, 11].item(), 0.9)

    def test_feature_values_positive_area(self):
        """All area values should be positive."""
        features = extract_handcrafted_node_features(self.image, self.instance_map)
        areas = features[:, 0]
        self.assertTrue((areas > 0).all())


if __name__ == '__main__':
    unittest.main()
