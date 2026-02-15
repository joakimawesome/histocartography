import argparse
import unittest

import batch_runner


class TestBatchRunnerCLI(unittest.TestCase):
    def test_build_pipeline_config_hybrid(self):
        args = argparse.Namespace(
            graph_method="knn",
            k=5,
            r=50.0,
            feat_mode="hybrid",
            feat_architecture="resnet50",
            feat_patch_size=72,
            feat_resize_size=None,
            feat_batch_size=64,
            feat_num_workers=4,
            feat_pin_memory=True,
            gnn_model_path=None,
            seg_device="cuda",
            seg_batch_size=32,
            seg_tile_size=256,
            seg_overlap=0,
            seg_level=0,
            seg_min_nucleus_area=10,
            stitch_mode="global",
        )

        config = batch_runner.build_pipeline_config(args)

        self.assertEqual(config.features.mode, "hybrid")
        self.assertEqual(config.features.batch_size, 64)
        self.assertEqual(config.features.num_workers, 4)
        self.assertTrue(config.features.pin_memory)
        self.assertEqual(config.segmentation.batch_size, 32)
