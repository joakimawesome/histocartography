# Quickstart: Nuclei Graph Extraction

This guide demonstrates how to extract nuclei and build a cell graph from a pathology image using `histocartography_ext`.

## Installation

```bash
pip install -e .
```

## Smoke Test

```bash
python -c "import histocartography_ext; print(histocartography_ext.__file__)"
```

## Extract Nuclei Graph

```python
import numpy as np
from histocartography_ext.preprocessing import NucleiExtractor, KNNGraphBuilder
from histocartography_ext.visualization import OverlayGraphVisualization
import matplotlib.pyplot as plt

# 1. Load Image (Example: create a dummy RGB image)
# In practice, load WSI patch: image = np.array(Image.open('patch.png'))
image = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)

# 2. Extract Nuclei (Using HoverNet or similar checkpoint)
# Note: Requires pre-trained model. If not available, this step might need a mock.
# nuclei_detector = NucleiExtractor(pretrained_data="pannuke") 
# nuclei_map, nuclei_centroids = nuclei_detector.process(image)

# Mocking output for demonstration if model not present:
nuclei_map = np.zeros((512, 512), dtype=np.int32)
num_nuclei = 50
nuclei_centroids = np.random.randint(0, 512, (num_nuclei, 2))
for i in range(num_nuclei):
    nuclei_map[nuclei_centroids[i, 1]-5:nuclei_centroids[i, 1]+5, 
               nuclei_centroids[i, 0]-5:nuclei_centroids[i, 0]+5] = i + 1

# 3. Build Graph (k-NN)
features = np.ones((num_nuclei, 10)) # Dummy features
graph_builder = KNNGraphBuilder(k=5, thresh=50)
graph = graph_builder.process(nuclei_map, features)

print(f"Graph built with {graph.num_nodes()} nodes and {graph.num_edges()} edges.")

# 4. Visualize
visualizer = OverlayGraphVisualization()
viz = visualizer.process(image, graph, instance_map=nuclei_map)
plt.imshow(viz)
plt.show()
```
