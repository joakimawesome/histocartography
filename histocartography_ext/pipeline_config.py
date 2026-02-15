from __future__ import annotations

from dataclasses import dataclass, asdict, field
from typing import Any, Optional


@dataclass
class ReproducibilityConfig:
    seed: int = 42
    save_metadata: bool = True


@dataclass
class SegmentationConfig:
    level: int = 0
    tile_size: int = 256
    overlap: int = 0
    device: Optional[str] = None
    batch_size: int = 16
    min_nucleus_area: int = 10
    stitch_mode: str = "global"


@dataclass
class GraphConfig:
    method: str = "knn"
    k: int = 5
    r: float = 50.0
    max_edge_length: Optional[float] = None
    remove_isolated_nodes: bool = False
    coord_space: str = "level-0-pixels"


@dataclass
class FeatureConfig:
    mode: str = "handcrafted"
    architecture: str = "resnet50"
    patch_size: int = 72
    resize_size: Optional[int] = None
    stride: Optional[int] = None
    downsample_factor: int = 1
    batch_size: int = 32
    num_workers: int = 0
    pin_memory: bool = False
    with_instance_masking: bool = False
    extraction_layer: Optional[str] = None
    gnn_model_path: Optional[str] = None


@dataclass
class PipelineConfig:
    segmentation: SegmentationConfig = field(default_factory=SegmentationConfig)
    graph: GraphConfig = field(default_factory=GraphConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    reproducibility: ReproducibilityConfig = field(
        default_factory=ReproducibilityConfig
    )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


__all__ = [
    "PipelineConfig",
    "SegmentationConfig",
    "GraphConfig",
    "FeatureConfig",
    "ReproducibilityConfig",
]
