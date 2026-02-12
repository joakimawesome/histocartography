import torch
import numpy as np
import cv2
import os
from typing import Optional, Tuple, List, Union
from .postprocess import process_instance

try:
    from ..ml.models.hovernet import HoverNet
except ImportError:
    # If strictly running from this package without parent context
    pass

class HoverNetInferencer:
    """
    Wrapper for HoVer-Net inference.
    """
    def __init__(
        self, 
        model_path: str, 
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        batch_size: int = 4
    ):
        self.device = device
        self.batch_size = batch_size
        self.model = self._load_model(model_path)
        self.model.to(self.device)
        self.model.eval()

    def _load_model(self, model_path: str) -> torch.nn.Module:
        # Load the checkpoint
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # If it's a full model object
        if isinstance(checkpoint, torch.nn.Module):
            return checkpoint
            
        # Otherwise, we need to instantiate the model and load state
        try:
            from ..ml.models.hovernet import HoverNet
        except ImportError:
            # Fallback for standalone script execution
            try:
                from histocartography_ext.ml.models.hovernet import HoverNet
            except ImportError:
                 # Last resort: try importing relative to this file's package structure logic
                 # Assuming inference.py is in histocartography_ext/nuclei/
                 pass

        # We must have HoverNet class now
        if 'HoverNet' not in locals():
             # Try one more time with a broader import or assume it handles it
             from ..ml.models.hovernet import HoverNet

        model = HoverNet()
        
        if isinstance(checkpoint, dict):
            if 'desc' in checkpoint:
                 # Official HoVer-Net checkpoints usually wrap state dict in 'desc' or just are keys
                 # Let's check keys.
                 # Actually, usually keys are 'module.eqn...' or just 'eqn...'
                 # Check for 'model_state_dict'
                 pass
            
            # Standard wrapper
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            else:
                state_dict = checkpoint
                
            # Handle DataParallel prefix 'module.' if present
            new_state_dict = {}
            for k, v in state_dict.items():
                if k.startswith('module.'):
                    new_state_dict[k[7:]] = v
                else:
                    new_state_dict[k] = v
                    
            try:
                model.load_state_dict(new_state_dict)
            except RuntimeError as e:
                # If Strict loading fails, maybe try strict=False or warn
                print(f"Warning: strict loading failed, trying strict=False. Error: {e}")
                model.load_state_dict(new_state_dict, strict=False)
                
        return model

    def predict_tile(
        self, 
        tile: np.ndarray, 
        return_centroids: bool = True
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Run inference on a single tile (H, W, 3).
        Tile should be RGB uint8.
        """
        # Preprocess
        # HoVer-Net expects normalized or raw? 
        # Existing code: transforms.ToTensor() which scales [0,1].
        # And it uses specific patch extraction.
        # Let's just standardise to tensor [0,1] (C, H, W)
        
        tile = tile.astype(np.float32) / 255.0
        tile_tensor = torch.from_numpy(tile).permute(2, 0, 1).unsqueeze(0).float()
        tile_tensor = tile_tensor.to(self.device)
        
        with torch.no_grad():
            # Output shape: (B, H, W, 3) 
            # Or (B, H, W, 2) ?
            output = self.model(tile_tensor)
            
        pred_map = output.cpu().numpy()[0] # (H, W, Channels)
        
        # Post-process
        instance_map = process_instance(pred_map)
        
        if return_centroids:
            from skimage.measure import regionprops
            props = regionprops(instance_map)
            centroids = np.array([p.centroid for p in props]) # (row, col) -> (y, x)
            if len(centroids) > 0:
                # regionprops returns (row, col) = (y, x)
                # We usually want (x, y)
                centroids = centroids[:, ::-1] 
            else:
                centroids = np.empty((0, 2))
            return instance_map, centroids
            
        return instance_map

    def predict_batch(
        self, 
        tiles: List[np.ndarray]
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Run inference on a batch of tiles.
        """
        results = []
        # Simple loop for now, can optimize with DataLoader if needed for very large batches
        # But for WSI tiling, we often fetch tiles lazily.
        # If tiles are pre-loaded:
        
        for i in range(0, len(tiles), self.batch_size):
            batch_tiles = tiles[i : i + self.batch_size]
            # Convert to tensor
            tensors = []
            for t in batch_tiles:
                t = t.astype(np.float32) / 255.0
                tensors.append(torch.from_numpy(t).permute(2, 0, 1))
            
            batch_tensor = torch.stack(tensors).to(self.device)
            
            with torch.no_grad():
                output = self.model(batch_tensor)
                
            output_np = output.cpu().numpy() # (B, H, W, C)
            
            for j in range(output_np.shape[0]):
                pred_map = output_np[j]
                instance_map = process_instance(pred_map)
                
                from skimage.measure import regionprops
                props = regionprops(instance_map)
                if len(props) > 0:
                    centroids = np.array([p.centroid for p in props])
                    centroids = centroids[:, ::-1] # (y, x) -> (x, y)
                else:
                    centroids = np.empty((0, 2))
                    
                results.append((instance_map, centroids))
                
        return results
