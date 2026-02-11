
import torch
import os
import unittest
import pandas as pd
import numpy as np
import shutil
import sys
from unittest.mock import MagicMock

# Mock dgl to avoid import errors
sys.modules['dgl'] = MagicMock()
sys.modules['dgl.data.utils'] = MagicMock()

# Mock other modules to avoid skimage incompatibility and other legacy issues
sys.modules['histocartography_ext.preprocessing.feature_extraction'] = MagicMock()
sys.modules['histocartography_ext.preprocessing.graph_builders'] = MagicMock()
sys.modules['histocartography_ext.preprocessing.io'] = MagicMock()
sys.modules['histocartography_ext.preprocessing.nuclei_concept_extraction'] = MagicMock()
sys.modules['histocartography_ext.preprocessing.nuclei_extraction'] = MagicMock()
sys.modules['histocartography_ext.preprocessing.stain_normalizers'] = MagicMock()
sys.modules['histocartography_ext.preprocessing.stats'] = MagicMock()
sys.modules['histocartography_ext.preprocessing.superpixel'] = MagicMock()
sys.modules['histocartography_ext.preprocessing.tissue_mask'] = MagicMock()
sys.modules['histocartography_ext.preprocessing.assignment_matrix'] = MagicMock()

from torch_geometric.data import Data
from histocartography_ext.preprocessing.graph_builder_pyg import build_nuclei_graph, save_nuclei_graph

class TestGraphBuilder(unittest.TestCase):
    def setUp(self):
        # Create dummy data
        self.num_nodes = 20
        self.nuclei_table = pd.DataFrame({
            'centroid_x': np.random.rand(self.num_nodes) * 100,
            'centroid_y': np.random.rand(self.num_nodes) * 100,
            'feat1': np.random.randn(self.num_nodes),
            'feat2': np.random.randn(self.num_nodes),
            'id': range(self.num_nodes)
        })
        self.out_dir = 'test_outputs'
        os.makedirs(self.out_dir, exist_ok=True)

    def tearDown(self):
        if os.path.exists(self.out_dir):
            shutil.rmtree(self.out_dir)

    def test_knn_graph(self):
        k = 5
        params = {'k': k, 'coord_space': 'level-0', 'mpp': 0.5}
        # Explicitly select features to avoid including 'id'
        data = build_nuclei_graph(self.nuclei_table, 'knn', params, feature_cols=['feat1', 'feat2'])

        self.assertIsInstance(data, Data)
        self.assertEqual(data.x.shape, (self.num_nodes, 2))
        self.assertEqual(data.pos.shape, (self.num_nodes, 2))
        self.assertTrue(data.edge_index.shape[1] > 0)
        self.assertEqual(data.edge_index.shape[0], 2)
        # Check that each node has at most k neighbors (out-degree)
        # Note: knn_graph returns directed edges. 
        # But we are using default knn_graph which is directed.
        # Check degree
        from torch_geometric.utils import degree
        deg = degree(data.edge_index[0], self.num_nodes)
        # deg should be <= k or exactly k? knn_graph connects to k nearest neighbors.
        # But if points are equidistant, it might be slightly different depending on implementation.
        # Or if N < k.
        self.assertTrue(torch.all(deg <= k))
        
        self.assertEqual(data.coord_space, 'level-0')
        self.assertEqual(data.mpp, 0.5)

    def test_radius_graph(self):
        r = 30.0
        params = {'r': r}
        data = build_nuclei_graph(self.nuclei_table, 'radius', params)
        self.assertIsInstance(data, Data)
        
        # Check edge distances
        if data.edge_index.numel() > 0:
            dist = (data.pos[data.edge_index[0]] - data.pos[data.edge_index[1]]).norm(dim=-1)
            self.assertTrue(torch.all(dist <= r))

    def test_pruning_max_edge_length(self):
        # Create points that are far apart
        df = pd.DataFrame({
            'centroid_x': [0, 10, 100],
            'centroid_y': [0, 0, 0],
            'f': [1, 1, 1]
        })
        # KNN with k=2. 0 connects to 10 (dist 10) and 100 (dist 100).
        # Pruning with max_len = 50 should remove edge to 100.
        params = {'k': 2, 'max_edge_length': 50.0}
        data = build_nuclei_graph(df, 'knn', params)
        
        # Edges (0,10) dist 10 - KEEP
        # Edges (10,0) dist 10 - KEEP
        # Edges (10,100) dist 90 - PRUNE
        # Edges (100,10) dist 90 - PRUNE
        # Edges (0, 100) dist 100 - PRUNE
        
        # In standard KNN: 
        # 0 neighbors: 10, 100
        # 10 neighbors: 0, 100
        # 100 neighbors: 10, 0
        
        # Allowed edges: (0,10), (10,0). (10,100) is > 50.
        
        for i in range(data.edge_index.shape[1]):
            src, dst = data.edge_index[:, i]
            d = (data.pos[src] - data.pos[dst]).norm()
            self.assertTrue(d <= 50.0)

    def test_export(self):
        params = {'k': 3}
        data = build_nuclei_graph(self.nuclei_table, 'knn', params)
        
        pt_path = os.path.join(self.out_dir, 'graph.pt')
        save_nuclei_graph(data, pt_path)
        self.assertTrue(os.path.exists(pt_path))
        # weights_only=False is required for loading custom objects like PyG Data without safety setup
        loaded_data = torch.load(pt_path, weights_only=False)
        self.assertIsInstance(loaded_data, Data)
        
        npz_path = os.path.join(self.out_dir, 'graph.npz')
        save_nuclei_graph(data, npz_path)
        self.assertTrue(os.path.exists(npz_path))

if __name__ == '__main__':
    unittest.main()
