import importlib.util
import tempfile
from pathlib import Path
from unittest import TestCase
from unittest.mock import MagicMock

import sys
import torch

# Mock optional deps to avoid import-time errors
sys.modules['dgl'] = MagicMock()
sys.modules['dgl.data'] = MagicMock()
sys.modules['dgl.data.utils'] = MagicMock()
sys.modules['dgl.graphbolt'] = MagicMock()
sys.modules['openslide'] = MagicMock()

from histocartography_ext.nuclei.checkpoints import (
    load_hovernet_checkpoint,
    _legacy_desc_key_for_state_key,
)


def _load_hovernet_class():
    root = Path(__file__).resolve().parents[2]
    hovernet_path = root / "histocartography_ext" / "ml" / "models" / "hovernet.py"
    spec = importlib.util.spec_from_file_location("hovernet_mod", str(hovernet_path))
    if spec is None or spec.loader is None:
        raise RuntimeError("Failed to load HoverNet module spec.")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.HoverNet


class TestCheckpointLoader(TestCase):
    def test_load_legacy_desc_checkpoint(self):
        hovernet_class = _load_hovernet_class()
        model = hovernet_class()

        desc = {}
        for k, v in model.state_dict().items():
            if k.endswith("num_batches_tracked"):
                continue
            desc_key = _legacy_desc_key_for_state_key(k)
            self.assertIsNotNone(desc_key)
            desc[desc_key] = v.clone()

        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_path = Path(tmpdir) / "hovernet_legacy.pth"
            torch.save({"desc": desc}, ckpt_path)

            loaded = load_hovernet_checkpoint(str(ckpt_path), device=None)
            self.assertIsInstance(loaded, torch.nn.Module)

            x = torch.zeros(1, 3, 256, 256)
            with torch.no_grad():
                y = loaded(x)
            self.assertEqual(y.shape[-1], 3)
            self.assertEqual(y.dtype, torch.float32)

    def test_load_state_dict_checkpoint(self):
        hovernet_class = _load_hovernet_class()
        model = hovernet_class()

        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_path = Path(tmpdir) / "hovernet_state_dict.pth"
            torch.save(model.state_dict(), ckpt_path)

            loaded = load_hovernet_checkpoint(str(ckpt_path), device=None)
            self.assertIsInstance(loaded, torch.nn.Module)
