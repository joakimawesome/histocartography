"""
Per-node deep feature extraction using a pre-trained CNN.

Reproduces the original HistoCartography ``DeepFeatureExtractor`` +
``InstanceMapPatchDataset`` pipeline:

1. For each nucleus instance, extract image patches centred on the
   instance centroid (stride-based, covering the bounding box).
2. Feed patches through a pre-trained CNN (e.g. ResNet-50) with the
   classification head removed.
3. Average the per-patch embeddings to produce one feature vector per
   node.

The main entry point is :class:`DeepNodeFeatureExtractor`.
"""

from __future__ import annotations

import copy
import math
import warnings
from typing import Callable, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torchvision
from skimage.measure import regionprops
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm.auto import tqdm


# ======================================================================
# Patch dataset  (port of original InstanceMapPatchDataset)
# ======================================================================

class InstanceMapPatchDataset(Dataset):
    """
    Extract patches around every nucleus instance in an image.

    Each ``__getitem__`` returns ``(region_index, patch_tensor)``.
    """

    def __init__(
        self,
        image: np.ndarray,
        instance_map: np.ndarray,
        patch_size: int = 72,
        stride: Optional[int] = None,
        resize_size: Optional[int] = None,
        fill_value: int = 255,
        mean: Optional[List[float]] = None,
        std: Optional[List[float]] = None,
        transform: Optional[Callable] = None,
        with_instance_masking: bool = False,
    ) -> None:
        self.image = image
        self.instance_map = instance_map
        self.patch_size = patch_size
        self.with_instance_masking = with_instance_masking
        self.fill_value = fill_value
        self.stride = stride if stride is not None else patch_size
        self.resize_size = resize_size
        self.mean = mean
        self.std = std

        # Pad image and instance map by patch_size on each side
        self.image = np.pad(
            self.image,
            ((patch_size, patch_size), (patch_size, patch_size), (0, 0)),
            mode="constant",
            constant_values=fill_value,
        )
        self.instance_map = np.pad(
            self.instance_map,
            ((patch_size, patch_size), (patch_size, patch_size)),
            mode="constant",
            constant_values=0,
        )

        self.patch_size_2 = patch_size // 2
        self.threshold = int(patch_size * patch_size * 0.25)
        self.properties = regionprops(self.instance_map)
        self.warning_threshold = 0.75

        self.patch_coordinates: List[List[int]] = []
        self.patch_region_count: List[int] = []
        self.patch_instance_ids: List[int] = []
        self.patch_overlap: List[int] = []

        # Build transforms
        basic_transforms: list = [transforms.ToPILImage()]
        if self.resize_size is not None:
            basic_transforms.append(transforms.Resize(self.resize_size))
        if transform is not None:
            basic_transforms.append(transform)
        basic_transforms.append(transforms.ToTensor())
        if self.mean is not None and self.std is not None:
            basic_transforms.append(transforms.Normalize(self.mean, self.std))
        self.dataset_transform = transforms.Compose(basic_transforms)

        self._precompute()
        self._warning()

    # ---- internal helpers ----

    def _add_patch(
        self, center_x: int, center_y: int, instance_index: int, region_count: int
    ) -> None:
        mask = self.instance_map[
            center_y - self.patch_size_2 : center_y + self.patch_size_2,
            center_x - self.patch_size_2 : center_x + self.patch_size_2,
        ]
        overlap = int(np.sum(mask == instance_index))
        if overlap > self.threshold:
            loc = [center_x - self.patch_size_2, center_y - self.patch_size_2]
            self.patch_coordinates.append(loc)
            self.patch_region_count.append(region_count)
            self.patch_instance_ids.append(instance_index)
            self.patch_overlap.append(overlap)

    def _get_patch(self, loc: list, region_id: Optional[int] = None) -> np.ndarray:
        min_x, min_y = loc
        max_x = min_x + self.patch_size
        max_y = min_y + self.patch_size
        patch = copy.deepcopy(self.image[min_y:max_y, min_x:max_x])
        if self.with_instance_masking and region_id is not None:
            instance_mask = ~(
                self.instance_map[min_y:max_y, min_x:max_x] == region_id
            )
            patch[instance_mask, :] = self.fill_value
        return patch

    def _precompute(self) -> None:
        for region_count, region in enumerate(self.properties):
            center_y, center_x = region.centroid
            center_x = int(round(center_x))
            center_y = int(round(center_y))
            min_y, min_x, max_y, max_x = region.bbox

            # Quadrant 1 (includes centroid)
            y_ = center_y
            while y_ >= min_y:
                x_ = center_x
                while x_ >= min_x:
                    self._add_patch(x_, y_, region.label, region_count)
                    x_ -= self.stride
                y_ -= self.stride

            # Quadrant 4
            y_ = center_y
            while y_ >= min_y:
                x_ = center_x + self.stride
                while x_ <= max_x:
                    self._add_patch(x_, y_, region.label, region_count)
                    x_ += self.stride
                y_ -= self.stride

            # Quadrant 2
            y_ = center_y + self.stride
            while y_ <= max_y:
                x_ = center_x
                while x_ >= min_x:
                    self._add_patch(x_, y_, region.label, region_count)
                    x_ -= self.stride
                y_ += self.stride

            # Quadrant 3
            y_ = center_y + self.stride
            while y_ <= max_y:
                x_ = center_x + self.stride
                while x_ <= max_x:
                    self._add_patch(x_, y_, region.label, region_count)
                    x_ += self.stride
                y_ += self.stride

    def _warning(self) -> None:
        if not self.patch_overlap:
            return
        overlap_arr = np.array(self.patch_overlap) / (self.patch_size ** 2)
        if np.mean(overlap_arr) < self.warning_threshold:
            warnings.warn(
                "Provided patch size is large – many patches have low instance overlap. "
                "Consider reducing patch_size.",
                stacklevel=2,
            )

    def __getitem__(self, index: int) -> Tuple[int, torch.Tensor]:
        patch = self._get_patch(
            self.patch_coordinates[index],
            self.patch_instance_ids[index],
        )
        patch = self.dataset_transform(patch)
        return self.patch_region_count[index], patch

    def __len__(self) -> int:
        return len(self.patch_coordinates)


# ======================================================================
# CNN patch feature extractor  (port of original PatchFeatureExtractor)
# ======================================================================

class PatchFeatureExtractor:
    """Wraps a pre-trained CNN and strips the classification head."""

    def __init__(
        self,
        architecture: str = "resnet50",
        device: Optional[torch.device] = None,
        patch_size: int = 72,
        extraction_layer: Optional[str] = None,
    ) -> None:
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device

        model = self._get_torchvision_model(architecture)
        model = self._remove_layers(model, extraction_layer)
        model = model.to(self.device)
        model.eval()
        self.model = model
        self.num_features = self._get_num_features(patch_size)

    def _get_torchvision_model(self, architecture: str) -> nn.Module:
        try:
            model_fn = getattr(torchvision.models, architecture)
        except AttributeError:
            raise ValueError(f"Unknown torchvision architecture: {architecture!r}")
        # Use 'weights' kwarg for newer torchvision, fall back to 'pretrained'
        try:
            model = model_fn(weights="DEFAULT")
        except TypeError:
            model = model_fn(pretrained=True)
        return model

    @staticmethod
    def _remove_layers(
        model: nn.Module, extraction_layer: Optional[str] = None
    ) -> nn.Module:
        if hasattr(model, "model"):
            model = model.model
        if isinstance(model, torchvision.models.resnet.ResNet):
            if extraction_layer is None:
                model.fc = nn.Sequential()
            else:
                model = _remove_modules(model, extraction_layer)
        elif hasattr(model, "classifier"):
            model.classifier = nn.Sequential()
            if extraction_layer is not None and hasattr(model, "avgpool"):
                model.avgpool = nn.Sequential()
                if hasattr(model, "features"):
                    model.features = _remove_modules(
                        model.features, extraction_layer
                    )
        else:
            # Best effort: remove the last linear layer
            children = list(model.children())
            if children and isinstance(children[-1], nn.Linear):
                model = nn.Sequential(*children[:-1], nn.Flatten())
        return model

    def _get_num_features(self, patch_size: int) -> int:
        dummy = torch.zeros(1, 3, patch_size, patch_size, device=self.device)
        with torch.no_grad():
            out = self.model(dummy)
        return out.shape[-1]

    def __call__(self, patch: torch.Tensor) -> torch.Tensor:
        patch = patch.to(self.device)
        with torch.no_grad():
            return self.model(patch).squeeze()


# ======================================================================
# Main deep extractor
# ======================================================================

class DeepNodeFeatureExtractor:
    """
    Per-node CNN feature extractor — port of the original
    ``DeepFeatureExtractor``.

    Usage::

        extractor = DeepNodeFeatureExtractor(architecture="resnet50")
        features = extractor.extract(image, instance_map)
        # features.shape == (num_nodes, num_cnn_features)
    """

    def __init__(
        self,
        architecture: str = "resnet50",
        patch_size: int = 72,
        resize_size: Optional[int] = None,
        stride: Optional[int] = None,
        downsample_factor: int = 1,
        normalizer: Optional[dict] = None,
        batch_size: int = 32,
        fill_value: int = 255,
        num_workers: int = 0,
        pin_memory: bool = False,
        verbose: bool = False,
        with_instance_masking: bool = False,
        extraction_layer: Optional[str] = None,
    ) -> None:
        self.patch_size = patch_size
        self.resize_size = resize_size
        self.stride = stride if stride is not None else patch_size
        self.downsample_factor = downsample_factor
        self.with_instance_masking = with_instance_masking
        self.verbose = verbose
        self.fill_value = fill_value
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory

        cuda = torch.cuda.is_available()
        self.device = torch.device("cuda:0" if cuda else "cpu")

        if normalizer is not None:
            self.normalizer_mean = normalizer.get("mean", [0, 0, 0])
            self.normalizer_std = normalizer.get("std", [1, 1, 1])
        else:
            # ImageNet defaults
            self.normalizer_mean = [0.485, 0.456, 0.406]
            self.normalizer_std = [0.229, 0.224, 0.225]

        self.patch_feature_extractor = PatchFeatureExtractor(
            architecture,
            device=self.device,
            patch_size=resize_size or patch_size,
            extraction_layer=extraction_layer,
        )

        if self.num_workers in (0, 1):
            torch.set_num_threads(1)

    @property
    def num_features(self) -> int:
        return self.patch_feature_extractor.num_features

    def extract(
        self,
        input_image: np.ndarray,
        instance_map: np.ndarray,
        transform: Optional[Callable] = None,
    ) -> torch.Tensor:
        """
        Extract per-node CNN features.

        Args:
            input_image: RGB uint8 image ``(H, W, 3)``.
            instance_map: Labelled instance map ``(H, W)`` (background = 0).
            transform: Optional extra augmentation transform.

        Returns:
            ``torch.Tensor`` of shape ``(num_instances, num_features)``.
        """
        if self.downsample_factor != 1:
            input_image = _downsample(input_image, self.downsample_factor)
            instance_map = _downsample(instance_map, self.downsample_factor)

        dataset = InstanceMapPatchDataset(
            image=input_image,
            instance_map=instance_map,
            patch_size=self.patch_size,
            stride=self.stride,
            resize_size=self.resize_size,
            fill_value=self.fill_value,
            mean=self.normalizer_mean,
            std=self.normalizer_std,
            transform=transform,
            with_instance_masking=self.with_instance_masking,
        )

        loader = DataLoader(
            dataset,
            shuffle=False,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=_collate_patches,
        )

        num_instances = len(dataset.properties)
        num_feats = self.patch_feature_extractor.num_features
        features = torch.zeros(
            num_instances, num_feats, dtype=torch.float32, device=self.device
        )
        embeddings: dict = {}

        for instance_indices, patches in tqdm(
            loader, total=len(loader), disable=not self.verbose, desc="CNN features"
        ):
            emb = self.patch_feature_extractor(patches)
            if emb.dim() == 1:
                emb = emb.unsqueeze(0)
            for j, key in enumerate(instance_indices):
                if key in embeddings:
                    embeddings[key][0] += emb[j]
                    embeddings[key][1] += 1
                else:
                    embeddings[key] = [emb[j].clone(), 1]

        for k, v in embeddings.items():
            features[k, :] = v[0] / v[1]

        return features.cpu().detach()


# ======================================================================
# Helpers
# ======================================================================

def _collate_patches(batch):
    instance_indices = [item[0] for item in batch]
    patches = torch.stack([item[1] for item in batch])
    return instance_indices, patches


def _downsample(image: np.ndarray, factor: int) -> np.ndarray:
    h, w = image.shape[:2]
    new_h = math.floor(h / factor)
    new_w = math.floor(w / factor)
    return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_NEAREST)


def _remove_modules(model: nn.Module, extraction_layer: str) -> nn.Module:
    """Remove all modules after *extraction_layer*."""
    new_modules = []
    for name, module in model.named_children():
        new_modules.append(module)
        if name == extraction_layer:
            break
    return nn.Sequential(*new_modules)
