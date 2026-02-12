import unittest
from unittest.mock import MagicMock, patch
import logging
import os
import shutil
import tempfile
from pathlib import Path
import pandas as pd
import numpy as np
import torch
from torch_geometric.data import Data

# Import the module to test
# We need to make sure the package is in path
# Mock dgl before importing anything that might use it
import sys
from unittest.mock import MagicMock
sys.modules['dgl'] = MagicMock()
sys.modules['dgl.data'] = MagicMock()
sys.modules['dgl.data.utils'] = MagicMock()
# Also mock graphbolt just in case
sys.modules['dgl.graphbolt'] = MagicMock()
# Mock openslide to avoid DLL errors
sys.modules['openslide'] = MagicMock()

from histocartography_ext import pipeline_runner

class TestPipelineRunner(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.slide_path = os.path.join(self.test_dir, "slide_1.svs")
        # Create a dummy slide file
        with open(self.slide_path, 'w') as f:
            f.write("dummy slide content")
            
        self.output_dir = os.path.join(self.test_dir, "out")
        self.model_path = "dummy_model.pth"
        
    def tearDown(self):
        # Close logger handlers to release file locks
        slide_out = os.path.join(self.output_dir, "slide_1")
        log_file = os.path.join(slide_out, "pipeline.log")
        logger = logging.getLogger(str(log_file))
        for handler in logger.handlers[:]:
            handler.close()
            logger.removeHandler(handler)
            
        shutil.rmtree(self.test_dir)

    @patch('histocartography_ext.pipeline_runner.get_tissue_mask')
    @patch('histocartography_ext.pipeline_runner.segment_nuclei')
    @patch('histocartography_ext.pipeline_runner.build_nuclei_graph')
    @patch('histocartography_ext.pipeline_runner.extract_gnn_embeddings')
    @patch('histocartography_ext.pipeline_runner.extract_graph_stats')
    @patch('histocartography_ext.pipeline_runner.save_nuclei_graph')
    @patch('histocartography_ext.pipeline_runner.torch.load')
    @patch('openslide.OpenSlide') 
    def test_pipeline_flow(self, mock_slide, mock_torch_load, mock_save_graph, mock_stats, mock_gnn, mock_build_graph, mock_segment, mock_mask):
        """Test the full pipeline flow with mocks."""
        
        # Setup Mocks
        # 1. OpenSlide
        mock_slide_obj = MagicMock()
        mock_slide_obj.dimensions = (2048, 2048)
        mock_slide_obj.get_thumbnail.return_value = np.zeros((100, 100, 3), dtype=np.uint8)
        mock_slide.return_value = mock_slide_obj
        
        # 2. Mask
        mock_mask.return_value = np.zeros((100, 100), dtype=np.uint8)
        
        # 3. Segmentation
        # Return a dummy dataframe
        mock_segment.return_value = pd.DataFrame({
            'centroid_x': [10, 20],
            'centroid_y': [10, 20],
            'area': [100, 100]
        })
        
        # 4. Graph Builder
        dummy_graph = Data(x=torch.randn(2, 5), edge_index=torch.zeros((2, 2)))
        dummy_graph.num_nodes = 2
        dummy_graph.num_edges = 2
        mock_build_graph.return_value = dummy_graph
        
        # 5. Features
        mock_gnn.return_value = np.random.rand(1, 128)
        mock_stats.return_value = {'stat1': 0.5}
        
        # Mock torch.load to return the dummy graph when loading "graph.pt"
        # and maybe a dummy model when loading "model.pth"
        def side_effect_torch_load(path, **kwargs):
            if "graph.pt" in str(path):
                return dummy_graph
            return MagicMock() # For model
        
        mock_torch_load.side_effect = side_effect_torch_load

        # --- Run First Time (Expect all calls) ---
        print("\n--- Running Pipeline First Time ---")
        pipeline_runner.run_pipeline(
            slide_path=self.slide_path,
            output_dir=self.output_dir,
            model_path=self.model_path,
            config={'feat_mode': 'gnn'}
        )
        
        # Verify calls
        mock_mask.assert_called()
        mock_segment.assert_called()
        mock_build_graph.assert_called()
        mock_gnn.assert_called()
        
        # Check files exist
        slide_out = os.path.join(self.output_dir, "slide_1")
        self.assertTrue(os.path.exists(os.path.join(slide_out, "nuclei.parquet")))
        # graph.pt is "saved" via mock_save_graph, we mocked it so file won't exist unless we mock side effect to create it.
        # But run_pipeline logic checks `mock_save_graph` call.
        # Wait, run_pipeline checks `if not graph_out_path.exists()` for caching.
        # Since we mocked the saving, the file WON'T exist physically unless we create it in the mock.
        # So the second run would try to run again if files are missing.
        
        # valid point. Verification script needs to actually create files or we mock `exists`.
        # Taking "create files" approach for better simulation.
        
    @patch('histocartography_ext.pipeline_runner.get_tissue_mask')
    @patch('histocartography_ext.pipeline_runner.segment_nuclei')
    @patch('histocartography_ext.pipeline_runner.build_nuclei_graph')
    @patch('histocartography_ext.pipeline_runner.extract_gnn_embeddings')
    @patch('histocartography_ext.pipeline_runner.save_nuclei_graph')
    @patch('histocartography_ext.pipeline_runner.torch.load')
    @patch('openslide.OpenSlide') 
    def test_pipeline_caching(self, mock_slide, mock_load, mock_save_graph, mock_gnn, mock_build, mock_segment, mock_mask):
        # Setup similar mocks but we will ensure files are created
        
        # 1. Slide
        mock_slide.return_value.dimensions = (100, 100)
        mock_slide.return_value.get_thumbnail.return_value = np.zeros((10, 10, 3), dtype=np.uint8)
        
        # 2. Segment
        def side_effect_segment(*args, **kwargs):
            return pd.DataFrame({'x': [1], 'y': [1]})
        mock_segment.side_effect = side_effect_segment
        
        # 3. Graph
        def side_effect_build(*args, **kwargs):
            return Data(x=torch.randn(1, 5))
        mock_build.side_effect = side_effect_build
        
        # 4. Save Graph
        def side_effect_save(data, path):
            # Create empty file
            with open(path, 'w') as f:
                f.write("graph data")
        mock_save_graph.side_effect = side_effect_save
        
        # 5. Extract
        def side_effect_extract(*args, **kwargs):
            return np.array([0.1, 0.2])
        mock_gnn.side_effect = side_effect_extract

        # Mock torch load for graph
        mock_load.return_value = Data(x=torch.randn(1, 5))

        # Run 1
        print("\n--- Running Pipeline Run 1 (Fresh) ---")
        pipeline_runner.run_pipeline(
            self.slide_path, 
            self.output_dir, 
            self.model_path,
            config={'feat_mode': 'gnn'}
        )
        
        # Counts
        c_mask = mock_mask.call_count
        c_seg = mock_segment.call_count
        c_build = mock_build.call_count
        c_feat = mock_gnn.call_count
        
        self.assertEqual(c_seg, 1)
        self.assertEqual(c_build, 1)
        self.assertEqual(c_feat, 1)
        
        # Files should exist (nuclei.parquet is saved by to_parquet in code, graph.pt by our mock, features.npy by code np.save)
        # Wait, I didn't verify `to_parquet` writing. `pipeline_runner.py` calls `nuclei_df.to_parquet`.
        # Pandas `to_parquet` requires library support. I should mock `to_parquet` or ensure `fastparquet`/`pyarrow` is available.
        # Or I can check if file exists. If `pd.to_parquet` works, it works.
        
        slide_out = os.path.join(self.output_dir, "slide_1")
        
        # Run 2
        print("\n--- Running Pipeline Run 2 (Cached) ---")
        pipeline_runner.run_pipeline(
            self.slide_path, 
            self.output_dir, 
            self.model_path,
            config={'feat_mode': 'gnn'}
        )
        
        # Counts should NOT increase
        self.assertEqual(mock_segment.call_count, c_seg, "Segmentation should be cached")
        self.assertEqual(mock_build.call_count, c_build, "Graph build should be cached")
        self.assertEqual(mock_gnn.call_count, c_feat, "Feature extraction should be cached")
        
if __name__ == '__main__':
    unittest.main()
