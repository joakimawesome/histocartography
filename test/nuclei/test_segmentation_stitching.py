"""Unit tests for nuclei segmentation stitching logic."""
import unittest
import sys
from unittest.mock import MagicMock

# Mock DGL and legacy preprocessing modules to avoid import chain failures
sys.modules.setdefault('dgl', MagicMock())
sys.modules.setdefault('dgl.data.utils', MagicMock())
sys.modules.setdefault('openslide', MagicMock())
sys.modules.setdefault('histocartography_ext.preprocessing.feature_extraction', MagicMock())
sys.modules.setdefault('histocartography_ext.preprocessing.graph_builders', MagicMock())
sys.modules.setdefault('histocartography_ext.preprocessing.io', MagicMock())
sys.modules.setdefault('histocartography_ext.preprocessing.nuclei_concept_extraction', MagicMock())
sys.modules.setdefault('histocartography_ext.preprocessing.nuclei_extraction', MagicMock())
sys.modules.setdefault('histocartography_ext.preprocessing.stain_normalizers', MagicMock())
sys.modules.setdefault('histocartography_ext.preprocessing.stats', MagicMock())
sys.modules.setdefault('histocartography_ext.preprocessing.superpixel', MagicMock())
sys.modules.setdefault('histocartography_ext.preprocessing.assignment_matrix', MagicMock())

import numpy as np

from histocartography_ext.nuclei.postprocess import process_instance
from histocartography_ext.nuclei.utils import tile_iterator


class TestTileIterator(unittest.TestCase):
    def test_covers_full_image(self):
        """All pixels should be covered by at least one tile."""
        W, H = 512, 512
        tile_size = 256
        covered = np.zeros((H, W), dtype=bool)
        for x, y, tw, th in tile_iterator(W, H, tile_size, overlap=0):
            covered[y:y+th, x:x+tw] = True
        self.assertTrue(covered.all())

    def test_overlap_coverage(self):
        W, H = 300, 300
        tile_size = 128
        overlap = 32
        coords = list(tile_iterator(W, H, tile_size, overlap))
        # Should have more tiles than without overlap
        coords_no_overlap = list(tile_iterator(W, H, tile_size, overlap=0))
        self.assertGreater(len(coords), len(coords_no_overlap))

    def test_invalid_overlap(self):
        with self.assertRaises(ValueError):
            list(tile_iterator(100, 100, tile_size=50, overlap=50))


class TestGlobalStitching(unittest.TestCase):
    """
    Verify that stitching a prediction map from non-overlapping tiles
    produces the same instance map as processing the full prediction
    at once.
    """

    def _make_synthetic_pred_map(self, h: int, w: int) -> np.ndarray:
        """Create a synthetic HoVer-Net output with a few 'nuclei'."""
        pred = np.zeros((h, w, 3), dtype=np.float32)
        rng = np.random.RandomState(42)

        yy_full, xx_full = np.mgrid[0:h, 0:w]

        for _ in range(8):
            cx = rng.randint(30, w - 30)
            cy = rng.randint(30, h - 30)
            r = rng.randint(5, 12)
            dist_sq = (xx_full - cx) ** 2 + (yy_full - cy) ** 2
            mask = dist_sq <= r ** 2
            pred[mask, 0] = 0.9
            pred[mask, 1] = ((xx_full[mask] - cx) / (r + 1e-6)).astype(np.float32)
            pred[mask, 2] = ((yy_full[mask] - cy) / (r + 1e-6)).astype(np.float32)

        return pred

    def test_stitched_equals_full(self):
        """Stitched tiles → single post-process should match full pred_map processing."""
        H, W = 256, 256
        full_pred = self._make_synthetic_pred_map(H, W)

        # Process the full prediction map at once (ground truth)
        instance_gt = process_instance(full_pred)

        # Now simulate stitching from non-overlapping tiles
        tile_size = 128
        stitched = np.zeros_like(full_pred)
        for x, y, tw, th in tile_iterator(W, H, tile_size, overlap=0):
            tile_pred = full_pred[y:y+th, x:x+tw, :]
            stitched[y:y+th, x:x+tw, :] = tile_pred

        # Post-process the stitched map
        instance_stitched = process_instance(stitched)

        # The instance maps should be identical (same labels, same shapes)
        # Because with non-overlapping tiles placeed back exactly, the pred
        # is bitwise identical.
        np.testing.assert_array_equal(instance_stitched, instance_gt)

    def test_nuclei_at_boundaries_preserved(self):
        """
        A nucleus straddling a tile boundary should be a single instance
        in the stitched result.
        """
        H, W = 256, 256
        pred = np.zeros((H, W, 3), dtype=np.float32)

        # Place a nucleus right at the center (tile boundary at 128)
        cy, cx, r = 128, 128, 10
        yy, xx = np.mgrid[0:H, 0:W]
        dist_sq = (xx - cx) ** 2 + (yy - cy) ** 2
        mask = dist_sq <= r ** 2
        pred[mask, 0] = 0.9
        pred[mask, 1] = ((xx[mask] - cx) / (r + 1e-6)).astype(np.float32)
        pred[mask, 2] = ((yy[mask] - cy) / (r + 1e-6)).astype(np.float32)

        # Stitch from 128px tiles
        stitched = np.zeros_like(pred)
        for x, y, tw, th in tile_iterator(W, H, 128, overlap=0):
            stitched[y:y+th, x:x+tw, :] = pred[y:y+th, x:x+tw, :]

        inst = process_instance(stitched)

        # The nucleus should appear as exactly one instance
        unique_labels = np.unique(inst[inst > 0])
        self.assertEqual(len(unique_labels), 1,
                         f"Expected 1 instance at boundary, got {len(unique_labels)}")


class TestModelLoading(unittest.TestCase):
    def test_rejects_state_dict(self):
        """_load_model should raise TypeError for a state dict."""
        import tempfile, torch, os
        from histocartography_ext.nuclei.inference import HoverNetInferencer

        # Save a plain dict (not a model)
        fd, path = tempfile.mkstemp(suffix=".pt")
        os.close(fd)
        try:
            torch.save({"some_key": torch.zeros(1)}, path)
            with self.assertRaises(TypeError):
                HoverNetInferencer(path, device="cpu")
        finally:
            os.unlink(path)


if __name__ == '__main__':
    unittest.main()
