# FLAME Clustering Algorithm

This is a Python implementation of the FLAME (Fuzzy clustering by Local Approximation of MEmberships) algorithm, based on the original C implementation.

## Overview

FLAME is a clustering algorithm that:
1. Identifies Cluster Supporting Objects (CSOs) and outliers based on local density
2. Uses a fuzzy membership approach where each object has partial membership to multiple clusters
3. Approximates memberships through a local propagation process
4. Works well for datasets with irregular-shaped clusters

FLAME was first described in:
> "FLAME, a novel fuzzy clustering method for the analysis of DNA microarray data", BMC Bioinformatics, 2007, 8:3.
> Available from: http://www.biomedcentral.com/1471-2105/8/3

## Features

- **Flexible distance metrics**: Supports any distance metric available in scikit-learn
- **Automatic parameter selection**: Can automatically determine the appropriate number of neighbors
- **Outlier detection**: Identifies outliers as part of the clustering process
- **Visualization tools**: Includes utility functions to visualize clustering results

## Requirements

- NumPy
- SciPy
- scikit-learn
- Matplotlib (for visualization)

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/flame-clustering.git
cd flame-clustering

# Install requirements
pip install numpy scipy scikit-learn matplotlib
```

## Usage

```python
from FLAME import FLAME
import numpy as np
from sklearn.datasets import make_blobs

# Generate sample data
X, y = make_blobs(n_samples=300, centers=4, cluster_std=0.6, random_state=0)

# Initialize and run FLAME clustering
flame = FLAME(k=10, dist_metric='euclidean', threshold=1.0)
labels = flame.fit_predict(X)

# Visualize results
flame.plot_clusters()
```

## Parameters

The `FLAME` class accepts the following parameters:

- `k` (int, optional): Number of nearest neighbors to consider. If None, will be calculated based on data size.
- `dist_metric` (str, default='euclidean'): Distance metric to use. Any metric supported by scikit-learn's NearestNeighbors can be used.
- `threshold` (float, default=1.0): Density threshold for detecting outliers. Higher values result in more objects being classified as outliers.

The `fit` method accepts:

- `X` (array-like): Input data matrix of shape (n_samples, n_features)
- `steps` (int, default=100): Maximum number of steps for fuzzy membership approximation
- `convergence_threshold` (float, default=1e-6): Convergence criterion for fuzzy membership approximation
- `cluster_threshold` (float, default=-1.0): Threshold for cluster assignment:
  - If < 0, assign each object to the cluster with highest membership
  - If in [0, 1], assign to clusters with membership > threshold (or to outlier group)

## Examples

See `flame_example.py` for comprehensive examples including:

1. Clustering different types of datasets (blobs, moons, circles)
2. Handling datasets with outliers
3. Comparing different distance metrics
4. Visualizing clustering results

Run the example script with:

```bash
python flame_example.py
```

## Algorithm Details

The FLAME algorithm consists of the following steps:

1. **Compute nearest neighbors** for each data point
2. **Define Cluster Supporting Objects (CSOs) and outliers**:
   - CSOs are objects with higher density than all their neighbors
   - Outliers are objects with lower density than all neighbors and below a threshold
3. **Local approximation of fuzzy memberships**:
   - Iteratively update memberships based on weighted combinations of neighbors
4. **Assign objects to clusters** based on final fuzzy memberships

## License

This implementation is provided under the same license as the original C implementation:

Permission is granted to anyone to use this software for any purpose, including commercial applications, and to alter it and redistribute it freely, subject to the following restrictions:

1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
2. The origin of this software must not be misrepresented; you must not claim that you wrote the original software.
3. Altered source versions must be plainly marked as such, and must not be misrepresented as being the original software.

## Acknowledgements

- Original C implementation by Fu Limin (phoolimin@gmail.com)
