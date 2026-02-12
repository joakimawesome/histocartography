import sys
from unittest.mock import MagicMock

# Mock DGL to avoid import error in environment
sys.modules['dgl'] = MagicMock()
sys.modules['dgl.data.utils'] = MagicMock()
sys.modules['dgl.distributed'] = MagicMock()
sys.modules['dgl.graphbolt'] = MagicMock()
# Mock OpenSlide to avoid DLL error
sys.modules['openslide'] = MagicMock()
sys.modules['openslide.OpenSlide'] = MagicMock()

import unittest
import numpy as np
import pandas as pd
from unittest.mock import MagicMock, patch
from histocartography_ext.nuclei.utils import tile_iterator, remove_duplicates, get_tissue_mask, solve_stitch_overlap
from histocartography_ext.nuclei.segmentation import segment_nuclei

class TestNucleiUtils(unittest.TestCase):
    def test_tile_iterator(self):
        # 100x100 image, 50 tile, 0 overlap -> 4 tiles
        tiles = list(tile_iterator(100, 100, 50, 0))
        self.assertEqual(len(tiles), 4)
        self.assertEqual(tiles[0], (0, 0, 50, 50))
        self.assertEqual(tiles[3], (50, 50, 50, 50))
        
        # Overlap
        tiles = list(tile_iterator(100, 100, 60, 10))
        # Step = 50. 
        # x=0, w=60. x=50, w=50 (100-50).
        # y=0 ... y=50 ...
        # Should be 4 tiles.
        self.assertEqual(len(tiles), 4)
        self.assertEqual(tiles[0], (0, 0, 60, 60))
        self.assertEqual(tiles[1], (50, 0, 50, 60))

    def test_remove_duplicates(self):
        df = pd.DataFrame({
            'centroid_x': [10, 10.5, 20],
            'centroid_y': [10, 10.5, 20],
            'area': [100, 50, 100]
        })
        # Should remove the second one (smaller area, close distance)
        # Dist = sqrt(0.5^2 + 0.5^2) = sqrt(0.5) = 0.707 < 8.0
        clean_df = remove_duplicates(df, distance_threshold=2.0)
        self.assertEqual(len(clean_df), 2)
        self.assertIn(100, clean_df['area'].values)
        self.assertNotIn(50, clean_df['area'].values)

class TestSegmentationFlow(unittest.TestCase):
    @patch('histocartography_ext.nuclei.segmentation.OpenSlide')
    @patch('histocartography_ext.nuclei.segmentation.HoverNetInferencer')
    def test_segment_nuclei(self, MockInferencer, MockOpenSlide):
        # Mock Slide
        slide = MagicMock()
        slide.level_dimensions = [(1000, 1000)]
        # Mock read_region to return a random tile
        def read_region(loc, level, size):
            return MagicMock(convert=lambda x: np.zeros((size[1], size[0], 3), dtype=np.uint8))
        slide.read_region.side_effect = read_region
        # Mock thumbnail
        slide.get_thumbnail.return_value = np.ones((100, 100, 3), dtype=np.uint8) * 255 # All white = tissue? No, saturation needs color.
        # Let's mock get_tissue_mask to return ones
        
        MockOpenSlide.return_value = slide
        
        # Mock Model
        inferencer = MockInferencer.return_value
        # Mock predict_batch to return dummy nuclei
        # Returns list of (instance_map, centroids)
        # Let's just return empty lists to test flow, or synthetic data
        
        # Synthetic instance map (100x100)
        inst_map = np.zeros((100, 100), dtype=np.uint16)
        inst_map[10:20, 10:20] = 1 # One nucleus
        centroids = np.array([[10, 10]]) # (x, y) relative to tile
        
        inferencer.predict_batch.return_value = [(inst_map, centroids)]
        
        # Run
        # We need to mock get_tissue_mask or ensure our thumbnail passes it
        with patch('histocartography_ext.nuclei.segmentation.get_tissue_mask') as mock_mask:
            mock_mask.return_value = np.ones((100, 100), dtype=np.uint8)
            
            df = segment_nuclei(
                "dummy.svs",
                model_path="dummy.pt",
                tile_size=100,
                overlap=0
            )
            
        # Check calls
        self.assertTrue(MockOpenSlide.called)
        self.assertTrue(inferencer.predict_batch.called)
        
        # We expect some nuclei if we mocked correctly.
        # Our mock predict_batch returns 1 nucleus per batch
        # We iterate 10x10 = 100 tiles? 1000x1000 image, 100 tile size.
        # So 100 tiles. 
        # predict_batch called for groups of tiles.
        # Should have ~100 nuclei.
        self.assertFalse(df.empty)
        self.assertGreater(len(df), 0)
        self.assertIn('centroid_x', df.columns)

    @patch('histocartography_ext.nuclei.segmentation.OpenSlide')
    @patch('histocartography_ext.nuclei.segmentation.HoverNetInferencer')
    def test_segment_nuclei_with_tuple_tissue_mask(self, MockInferencer, MockOpenSlide):
        # Mock Slide
        slide = MagicMock()
        slide.level_dimensions = [(200, 200)]

        def read_region(loc, level, size):
            return MagicMock(convert=lambda x: np.zeros((size[1], size[0], 3), dtype=np.uint8))

        slide.read_region.side_effect = read_region
        MockOpenSlide.return_value = slide

        # Mock Model
        inferencer = MockInferencer.return_value

        inst_map = np.zeros((100, 100), dtype=np.uint16)
        inst_map[10:20, 10:20] = 1
        centroids = np.array([[10, 10]])

        def predict_batch(tiles):
            return [(inst_map, centroids)] * len(tiles)

        inferencer.predict_batch.side_effect = predict_batch

        labeled_regions = np.zeros((20, 20), dtype=np.int32)
        binary_mask = np.ones((20, 20), dtype=np.uint8)
        tissue_mask = (labeled_regions, binary_mask)

        with patch('histocartography_ext.nuclei.segmentation.get_tissue_mask') as mock_mask:
            df = segment_nuclei(
                "dummy.svs",
                model_path="dummy.pt",
                tile_size=100,
                overlap=0,
                tissue_mask=tissue_mask,
                batch_size=2,
            )

        self.assertFalse(mock_mask.called)
        self.assertFalse(df.empty)
        self.assertEqual(len(df), 4)

if __name__ == '__main__':
    unittest.main()
