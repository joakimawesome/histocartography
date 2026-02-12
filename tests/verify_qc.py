import sys
import os
import shutil
from pathlib import Path
import pandas as pd
import numpy as np
import torch
from PIL import Image
from torch_geometric.data import Data

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from histocartography_ext.visualization.qc import (
    generate_qc_thumbnail,
    compute_qc_metrics,
    save_qc_metrics,
    plot_qc_distributions
)

def test_qc_functions():
    print("Testing QC functions...")
    
    # 1. Setup Dummies
    out_dir = Path("test_qc_output")
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir()
    
    # Dummy Image (Thumbnail)
    # 512x512 white image
    thumb = Image.new("RGB", (512, 512), (255, 255, 255))
    
    # Dummy Nuclei (in real coords, assuming downsample factor of 10 for simplicity)
    # 10 nuclei
    nuclei_data = {
        'centroid_x': np.random.randint(0, 5000, 10),
        'centroid_y': np.random.randint(0, 5000, 10),
        'nucleus_id': range(10)
    }
    nuclei_df = pd.DataFrame(nuclei_data)
    
    # Dummy Graph
    # Fully connected first 3 nodes
    edge_index = torch.tensor([[0, 0, 1, 1, 2, 2], [1, 2, 0, 2, 0, 1]], dtype=torch.long)
    graph_data = Data(x=torch.randn(10, 5), edge_index=edge_index)
    
    downsample = 10.0
    
    # 2. Test Thumbnail Generation
    print("  - Generating QC Thumbnail...")
    thumb_path = out_dir / "qc_thumbnail.png"
    generate_qc_thumbnail(
        thumb,
        nuclei_df,
        graph_data,
        str(thumb_path),
        downsample_factor=downsample
    )
    
    if thumb_path.exists():
        print("    [PASS] Thumbnail created.")
    else:
        print("    [FAIL] Thumbnail not created.")
        
    # 3. Test Metrics
    print("  - Computing QC Metrics...")
    slide_info = {'mpp': 0.5, 'width': 10000, 'height': 10000, 'tissue_area_px': 5000*5000}
    metrics = compute_qc_metrics(nuclei_df, graph_data, slide_info)
    
    print(f"    Metrics: {metrics}")
    
    metrics_path = out_dir / "qc_metrics.json"
    save_qc_metrics(metrics, str(metrics_path))
    
    if metrics_path.exists():
        print("    [PASS] Metrics saved.")
    else:
        print("    [FAIL] Metrics not saved.")
        
    # 4. Test Distributions
    print("  - Plotting Distributions...")
    dist_path = out_dir / "qc_distributions.png"
    # Need pos for distribution plot of edge lengths, or it uses nuclei centroids if missing
    # Our graph_data doesn't have pos, so it should use nuclei_df
    try:
        plot_qc_distributions(graph_data, nuclei_df, str(dist_path))
        if dist_path.exists():
            print("    [PASS] Distributions plotted.")
        else:
            print("    [FAIL] Distribution plot not created.")
    except Exception as e:
        print(f"    [FAIL] Distribution plot raised exception: {e}")

    # Cleanup
    # shutil.rmtree(out_dir)
    print("Done.")

if __name__ == "__main__":
    test_qc_functions()
