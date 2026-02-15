import torch
import numpy as np
import warnings
from typing import Optional, Tuple, List, Union
from .postprocess import process_instance
from .checkpoints import load_hovernet_checkpoint


def _resolve_device(device: Union[str, torch.device]) -> torch.device:
    """Resolve a user-provided device to a valid torch.device.

    - Accepts "auto" to mean "cuda if available else cpu".
    - If CUDA is requested but not available, falls back to CPU instead of
      crashing during torch.load / tensor.to().
    """
    if device is None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if isinstance(device, str) and device.lower() == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dev = device if isinstance(device, torch.device) else torch.device(str(device))
    if dev.type == "cuda" and not torch.cuda.is_available():
        warnings.warn(
            f"CUDA device requested ({device!r}) but torch.cuda.is_available() is False; "
            "falling back to CPU.",
            category=RuntimeWarning,
            stacklevel=2,
        )
        return torch.device("cpu")

    return dev


class HoverNetInferencer:
    """
    Wrapper for HoVer-Net inference.

    Supports two output modes:
        - return_raw=True  → returns the raw float32 prediction map (H, W, 3)
          so that callers can stitch tiles before post-processing.
        - return_raw=False → returns (instance_map, centroids) per tile,
          matching the legacy per-tile pipeline.
    """

    def __init__(
        self,
        model_path: str,
        device: Union[str, torch.device] = "cuda" if torch.cuda.is_available() else "cpu",
        batch_size: int = 4,
    ):
        self.device = _resolve_device(device)
        self.batch_size = batch_size
        self.model = self._load_model(model_path)
        self.model.to(self.device)
        self.model.eval()

    # ------------------------------------------------------------------
    # Model loading – full model object only (matches original repo)
    # ------------------------------------------------------------------
    def _load_model(self, model_path: str) -> torch.nn.Module:
        return load_hovernet_checkpoint(model_path, device=self.device)

    # ------------------------------------------------------------------
    # Single tile
    # ------------------------------------------------------------------
    def predict_tile(
        self,
        tile: np.ndarray,
        return_raw: bool = False,
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Run inference on a single tile (H, W, 3) RGB uint8.

        Args:
            tile: RGB uint8 numpy array.
            return_raw: If True, return the raw float32 pred_map (H, W, 3).
                        If False, return (instance_map, centroids).
        """
        tile_f = tile.astype(np.float32) / 255.0
        tile_tensor = (
            torch.from_numpy(tile_f).permute(2, 0, 1).unsqueeze(0).float()
        )
        tile_tensor = tile_tensor.to(self.device)

        with torch.no_grad():
            output = self.model(tile_tensor)

        pred_map = output.cpu().numpy()[0]  # (H, W, C)

        if return_raw:
            return pred_map

        # Legacy path: post-process immediately
        instance_map = process_instance(pred_map)
        from skimage.measure import regionprops

        props = regionprops(instance_map)
        if len(props) > 0:
            centroids = np.array([p.centroid for p in props])
            centroids = centroids[:, ::-1]  # (y, x) → (x, y)
        else:
            centroids = np.empty((0, 2))
        return instance_map, centroids

    # ------------------------------------------------------------------
    # Batch
    # ------------------------------------------------------------------
    def predict_batch(
        self,
        tiles: List[np.ndarray],
        return_raw: bool = False,
    ) -> Union[List[np.ndarray], List[Tuple[np.ndarray, np.ndarray]]]:
        """
        Run inference on a list of tiles.

        Args:
            tiles: List of RGB uint8 numpy arrays.
            return_raw: If True, each result is a raw float32 pred_map (H, W, 3).
                        If False, each result is (instance_map, centroids).
        """
        results: list = []

        for i in range(0, len(tiles), self.batch_size):
            batch_tiles = tiles[i : i + self.batch_size]
            tensors = []
            for t in batch_tiles:
                t_f = t.astype(np.float32) / 255.0
                tensors.append(torch.from_numpy(t_f).permute(2, 0, 1))

            batch_tensor = torch.stack(tensors).to(self.device)

            with torch.no_grad():
                output = self.model(batch_tensor)

            output_np = output.cpu().numpy()  # (B, H, W, C)

            for j in range(output_np.shape[0]):
                pred_map = output_np[j]

                if return_raw:
                    results.append(pred_map)
                    continue

                # Legacy path
                instance_map = process_instance(pred_map)
                from skimage.measure import regionprops

                props = regionprops(instance_map)
                if len(props) > 0:
                    centroids = np.array([p.centroid for p in props])
                    centroids = centroids[:, ::-1]
                else:
                    centroids = np.empty((0, 2))
                results.append((instance_map, centroids))

        return results
