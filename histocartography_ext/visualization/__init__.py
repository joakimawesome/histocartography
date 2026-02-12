try:
    from .visualization import (
        OverlayGraphVisualization,
        InstanceImageVisualization,
        HACTVisualization,
    )
except Exception:
    # Allow import to succeed if DGL is missing/broken, 
    # but these classes won't be available.
    pass

from .qc import (
    generate_qc_thumbnail,
    compute_qc_metrics,
    save_qc_metrics,
    plot_qc_distributions
)

__all__ = [
    "OverlayGraphVisualization",
    "InstanceImageVisualization",
    "HACTVisualization",
    "generate_qc_thumbnail",
    "compute_qc_metrics",
    "save_qc_metrics",
    "plot_qc_distributions",
]