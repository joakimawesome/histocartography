"""Unit tests for features/deep_features.py (CNN per-node features)."""
import unittest
import numpy as np
import torch
import cv2
import sys
from unittest.mock import MagicMock

# Mock heavy optional deps
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

from histocartography_ext.features.deep_features import (
    DeepNodeFeatureExtractor,
    InstanceMapPatchDataset,
    PatchFeatureExtractor,
)


def _make_synthetic_data(num_circles: int = 3, img_size: int = 200):
    rng = np.random.RandomState(42)
    image = np.full((img_size, img_size, 3), 220, dtype=np.uint8)
    instance_map = np.zeros((img_size, img_size), dtype=np.int32)
    for i in range(1, num_circles + 1):
        cx = rng.randint(40, img_size - 40)
        cy = rng.randint(40, img_size - 40)
        r = rng.randint(10, 18)
        color = tuple(rng.randint(50, 200, 3).tolist())
        cv2.circle(image, (cx, cy), r, color, -1)
        cv2.circle(instance_map, (cx, cy), r, int(i), -1)
    return image, instance_map


class TestInstanceMapPatchDataset(unittest.TestCase):
    def setUp(self):
        self.image, self.instance_map = _make_synthetic_data()

    def test_dataset_length(self):
        ds = InstanceMapPatchDataset(
            self.image, self.instance_map, patch_size=32, stride=32
        )
        self.assertGreater(len(ds), 0)

    def test_dataset_item_shape(self):
        ds = InstanceMapPatchDataset(
            self.image, self.instance_map, patch_size=32, stride=32
        )
        if len(ds) > 0:
            idx, patch = ds[0]
            self.assertIsInstance(idx, int)
            self.assertEqual(patch.dim(), 3)  # (C, H, W)
            self.assertEqual(patch.shape[0], 3)

    def test_resize(self):
        ds = InstanceMapPatchDataset(
            self.image, self.instance_map,
            patch_size=32, stride=32, resize_size=64
        )
        if len(ds) > 0:
            _, patch = ds[0]
            self.assertEqual(patch.shape[1], 64)
            self.assertEqual(patch.shape[2], 64)


class TestPatchFeatureExtractor(unittest.TestCase):
    def test_resnet18_features(self):
        """ResNet-18 with fc removed should produce 512-dim features."""
        extractor = PatchFeatureExtractor(
            architecture="resnet18",
            patch_size=72,
            device=torch.device("cpu"),
        )
        self.assertEqual(extractor.num_features, 512)

    def test_call(self):
        extractor = PatchFeatureExtractor(
            architecture="resnet18",
            patch_size=72,
            device=torch.device("cpu"),
        )
        dummy = torch.randn(2, 3, 72, 72)
        out = extractor(dummy)
        self.assertEqual(out.shape, (2, 512))


class TestDeepNodeFeatureExtractor(unittest.TestCase):
    def setUp(self):
        self.image, self.instance_map = _make_synthetic_data(num_circles=3)

    def test_extract_shape(self):
        """Output should be (num_instances, num_features) tensor."""
        extractor = DeepNodeFeatureExtractor(
            architecture="resnet18",
            patch_size=32,
            resize_size=72,
            batch_size=4,
        )
        features = extractor.extract(self.image, self.instance_map)
        self.assertIsInstance(features, torch.Tensor)
        self.assertEqual(features.shape[0], 3)  # 3 nuclei
        self.assertEqual(features.shape[1], 512)  # ResNet-18 features

    def test_no_nan(self):
        extractor = DeepNodeFeatureExtractor(
            architecture="resnet18",
            patch_size=32,
            resize_size=72,
        )
        features = extractor.extract(self.image, self.instance_map)
        self.assertFalse(torch.isnan(features).any())


if __name__ == '__main__':
    unittest.main()
