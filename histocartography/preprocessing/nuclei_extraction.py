"""Detect and Classify nuclei from an image with the HoverNet model."""

import os
from pathlib import Path
from typing import List, Optional, Tuple, Union

import cv2
import numpy as np
import torch
from PIL import Image

from skimage.measure import regionprops
from skimage.morphology import remove_small_objects
from skimage.segmentation import watershed

from scipy.ndimage import measurements
from scipy.ndimage.morphology import binary_fill_holes

from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

from ..pipeline import PipelineStep
from ..ml.models.hovernet import HoverNet
from ..utils.image import extract_patches_from_image
from ..utils import download_box_link

DATASET_TO_CHECKPOINT_URL = {
    "pannuke": "https://drive.google.com/uc?export=download&id=1SbSArI3KOOWHxRlxnjchO7_MbWzB4lNR",
    "monusac": "https://drive.google.com/uc?export=download&id=13qkxDqv7CUqxN-l5CpeFVmc24mDw6CeV",
}

CHECKPOINT_PATH = "../../checkpoints"

GPU_DEFAULT_BATCH_SIZE = 16
CPU_DEFAULT_BATCH_SIZE = 2


class NucleiExtractor(PipelineStep):
    """Nuclei extraction"""

    def __init__(
        self,
        pretrained_data: str = "pannuke",
        model_path: Optional[str] = None,
        batch_size: Optional[int] = None,
        **kwargs,
    ) -> None:
        """Create a nuclei extractor

        Args:
            pretrained_data (str): Load checkpoint pretrained on some data. Options are 'pannuke' or 'monusac'. Default to 'pannuke'.
            model_path (str): Path to a pre-trained model. If none, the checkpoint specified in pretrained_data will be used. Default to None.
            batch_size (int, optional): Batch size. Defaults to None.
        """
        self.pretrained_data = pretrained_data
        super().__init__(**kwargs)

        # set class attributes
        cuda = torch.cuda.is_available()
        self.device = torch.device("cuda:0" if cuda else "cpu")
        if batch_size is None:
            # bs set to 16 if GPU, otherwise 2.
            self.batch_size = GPU_DEFAULT_BATCH_SIZE if cuda else CPU_DEFAULT_BATCH_SIZE
        else:
            self.batch_size = batch_size

        checkpoint_dir = Path(os.path.dirname(__file__)) / CHECKPOINT_PATH
        candidate_paths: List[Path] = []

        if model_path is not None:
            candidate_paths.append(Path(model_path))
        else:
            assert pretrained_data in [
                "pannuke",
                "monusac",
            ], 'Unsupported pretrained data checkpoint. Options are "pannuke" and "monusac".'

            local_checkpoint = checkpoint_dir / f"hovernet_{pretrained_data}.pth"
            downloaded_checkpoint = checkpoint_dir / f"{pretrained_data}.pt"

            if local_checkpoint.is_file():
                candidate_paths.append(local_checkpoint)
            candidate_paths.append(downloaded_checkpoint)

        load_errors: List[str] = []
        loaded = False
        for candidate_path in candidate_paths:
            candidate_path = candidate_path.resolve()

            if not candidate_path.is_file() and pretrained_data in DATASET_TO_CHECKPOINT_URL:
                download_box_link(DATASET_TO_CHECKPOINT_URL[pretrained_data], str(candidate_path))

            if candidate_path.is_file() and self._looks_like_html(candidate_path):
                if pretrained_data in DATASET_TO_CHECKPOINT_URL:
                    try:
                        candidate_path.unlink(missing_ok=True)
                        download_box_link(DATASET_TO_CHECKPOINT_URL[pretrained_data], str(candidate_path))
                    except Exception:
                        pass

            if candidate_path.is_file() and self._looks_like_html(candidate_path):
                load_errors.append(
                    f"{candidate_path}: looks like an HTML response instead of a checkpoint binary"
                )
                continue

            try:
                self._load_model_from_path(str(candidate_path))
                loaded = True
                break
            except RuntimeError as exception:
                load_errors.append(f"{candidate_path}: {exception}")

        if not loaded:
            raise RuntimeError(
                "Unable to load a valid nuclei checkpoint. Tried:\n- "
                + "\n- ".join(load_errors)
                + "\nPass a known-good file via model_path/--nuclei-model-path."
            )

        self.model = self.model.to(self.device)
        self.model.eval()

    def _load_model_from_path(self, model_path):
        """Load nuclei extraction model from provided model path."""
        try:
            checkpoint = torch.load(model_path, map_location=self.device)
        except Exception as exception:
            raise RuntimeError(
                f"Unable to load nuclei model checkpoint from '{model_path}'. "
                "The file may be corrupted or may not be a valid PyTorch checkpoint."
            ) from exception

        if isinstance(checkpoint, torch.nn.Module):
            self.model = checkpoint
            return

        state_dict = self._extract_state_dict(checkpoint)
        if isinstance(state_dict, dict) and len(state_dict) > 0:
            cleaned_state_dict = {}
            for key, value in state_dict.items():
                if isinstance(key, str) and key.startswith("module."):
                    cleaned_state_dict[key.replace("module.", "", 1)] = value
                else:
                    cleaned_state_dict[key] = value

            model = HoverNet()
            try:
                model.load_state_dict(cleaned_state_dict, strict=True)
                self.model = model
                return
            except RuntimeError as exception:
                raise RuntimeError(
                    f"Checkpoint '{model_path}' does not match the expected HoverNet architecture. "
                    "Provide a compatible checkpoint via model_path/--nuclei-model-path."
                ) from exception

        if isinstance(checkpoint, dict):
            key_summary = ", ".join(
                [f"{k}:{type(v).__name__}" for k, v in list(checkpoint.items())[:10]]
            )
            raise RuntimeError(
                f"Unsupported checkpoint format in '{model_path}'. Top-level keys/types: {key_summary}"
            )

        raise RuntimeError(
            f"Unsupported checkpoint format in '{model_path}'. Expected a serialized torch.nn.Module, "
            f"but got '{type(checkpoint)}'."
        )

    def _extract_state_dict(self, checkpoint_obj):
        """Recursively extract a plausible torch state_dict from heterogeneous checkpoint formats."""
        if isinstance(checkpoint_obj, torch.nn.Module):
            return checkpoint_obj.state_dict()

        if isinstance(checkpoint_obj, dict):
            # Common containers in training checkpoints
            for key in ["state_dict", "model_state_dict", "model", "net", "weights", "params"]:
                if key in checkpoint_obj:
                    extracted = self._extract_state_dict(checkpoint_obj[key])
                    if extracted is not None:
                        return extracted

            # Direct state-dict style: dotted string keys + tensor values
            if len(checkpoint_obj) > 0 and all(isinstance(key, str) for key in checkpoint_obj.keys()):
                tensor_items = {
                    key: value for key, value in checkpoint_obj.items() if torch.is_tensor(value)
                }
                if len(tensor_items) > 0 and any("." in key for key in tensor_items.keys()):
                    return tensor_items

            # Search nested objects in dict values
            for value in checkpoint_obj.values():
                extracted = self._extract_state_dict(value)
                if extracted is not None:
                    return extracted

        if isinstance(checkpoint_obj, (list, tuple)):
            # Sometimes checkpoints are serialized as list/tuple of (key, tensor) pairs
            if len(checkpoint_obj) > 0 and all(
                isinstance(item, (list, tuple)) and len(item) == 2 for item in checkpoint_obj
            ):
                try:
                    as_dict = dict(checkpoint_obj)
                    extracted = self._extract_state_dict(as_dict)
                    if extracted is not None:
                        return extracted
                except Exception:
                    pass

            for value in checkpoint_obj:
                extracted = self._extract_state_dict(value)
                if extracted is not None:
                    return extracted

        return None

    @staticmethod
    def _looks_like_html(path: Path) -> bool:
        """Detect whether a downloaded file is likely an HTML response page instead of a checkpoint."""
        try:
            with open(path, "rb") as handle:
                head = handle.read(256).lstrip().lower()
            return head.startswith(b"<") or b"<html" in head or b"<!doctype" in head
        except Exception:
            return False

    def _process(  # type: ignore[override]
        self,
        input_image: np.ndarray,
        tissue_mask: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Extract nuclei from the input_image
        Args:
            input_image (np.array): Original RGB image
            tissue_mask (None, np.array): Input tissue mask.
        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: instance_map, instance_centroids
        """
        return self._extract_nuclei(input_image, tissue_mask)

    def _extract_nuclei(
        self,
        input_image: np.ndarray,
        tissue_mask: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Extract features from the input_image for the defined structure

        Args:
            input_image (np.array): Original RGB image
            tissue_mask (Optional[np.ndarray]): Tissue mask to extract nuclei on. Defaults to None.

        Returns:
            Tuple[np.ndarray, np.ndarray]: instance_map, instance_centroids
        """
        if tissue_mask is not None:
            input_image[tissue_mask == 0] = (255, 255, 255)

        image_dataset = ImageToPatchDataset(input_image)

        def collate(batch):
            coords = [x[0] for x in batch]
            patches = torch.stack([x[1] for x in batch])
            return coords, patches

        image_loader = DataLoader(
            image_dataset,
            shuffle=False,
            batch_size=self.batch_size,
            collate_fn=collate)
        pred_map = torch.empty(
            size=(image_dataset.max_x_coord, image_dataset.max_y_coord, 3),
            dtype=torch.float32,
            device=self.device,
        )

        for coords, image_batch in tqdm(
            image_loader, desc="Patch-level nuclei detection"
        ):
            image_batch = image_batch.to(self.device)
            with torch.no_grad():
                out = self.model(image_batch).cpu()
                for i in range(out.shape[0]):
                    left = coords[i][0]  # left, bottom, right, top
                    bottom = coords[i][1]
                    right = coords[i][2]
                    top = coords[i][3]
                    pred_map[bottom:top, left:right, :] = out[i, :, :, :]

        # crop to original image size
        pred_map = pred_map.cpu().detach().numpy()
        pred_map = pred_map[: image_dataset.im_h, : image_dataset.im_w, :]

        # post process instance map
        instance_map = process_instance(pred_map)

        # extract the centroid location in the instance map
        regions = regionprops(instance_map)
        instance_centroids = np.empty((len(regions), 2))
        for i, region in enumerate(regions):
            center_y, center_x = region.centroid  # row, col
            center_x = int(round(center_x))
            center_y = int(round(center_y))
            instance_centroids[i, 0] = center_x
            instance_centroids[i, 1] = center_y

        return instance_map, instance_centroids

    def precompute(
        self,
        link_path: Union[None, str, Path] = None,
        precompute_path: Union[None, str, Path] = None,
    ) -> None:
        """Precompute all necessary information

        Args:
            link_path (Union[None, str, Path], optional): Path to link to. Defaults to None.
            precompute_path (Union[None, str, Path], optional): Path to save precomputation outputs. Defaults to None.
        """
        if self.save_path is not None and link_path is not None:
            self._link_to_path(Path(link_path) / "nuclei_maps")


class ImageToPatchDataset(Dataset):
    """Helper class to transform an image as a set of patched wrapped in a pytorch dataset"""

    def __init__(
        self,
        image: np.ndarray,
    ) -> None:
        """Create a dataset for a given image and extracted instance maps with desired patches.
           Patches have shape of (3, 256, 256) as defined by HoverNet model.

        Args:
            image (np.ndarray): RGB input image
        """
        self.image = image
        self.dataset_transform = transforms.Compose(
            [
                transforms.ToTensor(),
            ]
        )
        self.im_h = image.shape[0]
        self.im_w = image.shape[1]
        self.all_patches, self.coords = extract_patches_from_image(
            image, self.im_h, self.im_w
        )
        self.nr_patches = len(self.all_patches)
        self.max_y_coord = max([coord[-2] for coord in self.coords])
        self.max_x_coord = max([coord[-1] for coord in self.coords])

    def __getitem__(self, index: int) -> Tuple[int, torch.Tensor]:
        """Loads an image for a given instance maps index

        Args:
            index (int): patch index

        Returns:
            Tuple[int, torch.Tensor]: index, image as tensor
        """
        patch = self.all_patches[index]
        coord = self.coords[index]
        transformed_image = self.dataset_transform(Image.fromarray(patch))
        return coord, transformed_image

    def __len__(self) -> int:
        """Returns the length of the dataset

        Returns:
            int: Length of the dataset
        """
        return self.nr_patches


def process_np_hv_channels(pred: np.ndarray) -> np.ndarray:
    """
    Process Nuclei Prediction with XY Coordinate Map

    Args:
        pred (np.ndarray): HoverNet model output, that contains:
                            - channel 0 contain probability map of nuclei
                            - channel 1 contains X-map
                            - channel 2 contains Y-map
    Returns:
         pred_instance (np.ndarray): instance map
    """

    # post-process probability map
    proba_map = np.copy(pred[:, :, 0])  # extract proba maps
    proba_map[proba_map >= 0.5] = 1
    proba_map[proba_map < 0.5] = 0
    proba_map = measurements.label(proba_map)[0]
    proba_map = remove_small_objects(proba_map, min_size=10)
    proba_map[proba_map > 0] = 1

    h_dir = pred[:, :, 1]  # extract horizontal map
    v_dir = pred[:, :, 2]  # extract vertical map

    # normalizing
    h_dir = cv2.normalize(
        h_dir,
        None,
        alpha=0,
        beta=1,
        norm_type=cv2.NORM_MINMAX,
        dtype=cv2.CV_32F)
    v_dir = cv2.normalize(
        v_dir,
        None,
        alpha=0,
        beta=1,
        norm_type=cv2.NORM_MINMAX,
        dtype=cv2.CV_32F)

    # apply sobel filtering
    sobelh = cv2.Sobel(h_dir, cv2.CV_64F, 1, 0, ksize=21)
    sobelv = cv2.Sobel(v_dir, cv2.CV_64F, 0, 1, ksize=21)

    sobelh = 1 - (cv2.normalize(sobelh, None, alpha=0, beta=1,
                                norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F))
    sobelv = 1 - (cv2.normalize(sobelv, None, alpha=0, beta=1,
                                norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F))

    # binarize
    overall = np.maximum(sobelh, sobelv)
    overall = overall - (1 - proba_map)
    overall[overall < 0] = 0

    dist = (1.0 - overall) * proba_map
    dist = -cv2.GaussianBlur(dist, (3, 3), 0)

    overall[overall >= 0.5] = 1
    overall[overall < 0.5] = 0
    marker = proba_map - overall
    marker[marker < 0] = 0
    marker = binary_fill_holes(marker).astype("uint8")
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    marker = cv2.morphologyEx(marker, cv2.MORPH_OPEN, kernel)
    marker = measurements.label(marker)[0]
    marker = remove_small_objects(marker, min_size=10)

    pred_inst = watershed(dist, marker, mask=proba_map, watershed_line=False)

    return pred_inst


def process_instance(
        pred_map: np.ndarray,
        output_dtype: str = "uint16") -> np.ndarray:
    """
    Post processing script for image tiles

    Args:
        pred_map (np.ndarray): commbined output of np and hv branches
        output_dtype (str): data type of output

    Returns:
        pred_inst (np.ndarray): pixel-wise nuclear instance segmentation prediction
    """

    pred_inst = np.squeeze(pred_map)
    pred_inst = process_np_hv_channels(pred_inst)
    pred_inst = pred_inst.astype(output_dtype)
    return pred_inst
