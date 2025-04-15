import numpy as np
from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.metrics import pairwise_distances

class DensityPeaksClustering(BaseEstimator, ClusterMixin):
    """
    Density Peaks Clustering algorithm based on the paper by Rodriguez and Laio (2014).

    Parameters:
    ----------
    dc: float, default=None
        Cutoff distance for density estimation. If None, estimated automatically.

    percent : float, default=2.0
        Percentage used to estimate dc (average number of neighbors ~ percent% of N).

    n_clusters : int, default=None
        Number of clusters. If None, estimated automatically using
    
    density_estimator : str, default='cutoff'
        Method for density estimation ('cutoff' or 'gaussian').
    """
    def __init__(
        self,
        dc=None,
        percent=2.0,
        n_clusters=None,
        density_estimator='cutoff',
    ):
        self.dc = dc
        self.percent = percent
        self.n_clusters = n_clusters
        self.density_estimator = density_estimator

    def fit(self, X, y=None):
        """
        Fit the model to the data.
        """
        # Compute pairwise distances (using Euclidean distance)
        distances = pairwise_distances(X)
        N = distances.shape[0]

        # Estimate dc if not provided
        self.dc_ = self.dc if self.dc is not None else self._estimate_dc(distances)

        # Compute local densities
        self.rho_ = self._compute_density(distances)

        # Compute delta values for each point (distance to nearest point of higher density)
        self.delta_ = self._compute_delta(distances, self.rho_)

        # Identify cluster centers
        self.centers_ = self._find_centers(self.rho_, self.delta_)
        self.n_clusters_ = len(self.centers_)

        # Initialize labels for centers
        self.labels_ = -1 * np.ones(N, dtype=int)
        self.labels_[self.centers_] = np.arange(self.n_clusters_)

        # Assign the rest of points to cluster centers using the nearest neighbor with higher density
        self.labels_ = self._assign_clusters(distances, self.rho_, self.labels_)

        # Identify halo points for the clusters
        self.halo_ = self._find_halo(distances, self.labels_, self.rho_)

        return self
    
    def fit_predict(self, X, y=None):
        """
        Fit the model and return the predicted labels for the data.
        """
        return self.fit(X).labels_

    def _estimate_dc(self, distances):
        """
        Estimate cutoff distance dc such that average number of neighbors
        is around percent% of the total number of points in the dataset.
        """
        N = distances.shape[0]
        p = self.percent / 100.0

        # Extract upper triangular distances (excluding diagonal)
        all_distances = distances[np.triu_indices(N, k=1)]

        # Compute the p-quantile of the distances
        dc = np.quantile(all_distances, p)

        return dc

    def _compute_density(self, distances):
        """
        Compute local density ρ_i (rho_i) for each point.
        """
        if self.density_estimator == 'cutoff':
            # ρ_i = number of points within dc
            rho = np.sum(distances < self.dc_, axis=1)
        elif self.density_estimator == 'gaussian':
            # ρ_i = sum(exp(-(d_ij / dc)^2))
            rho = np.sum(np.exp(-(distances / self.dc_)**2), axis=1)
        else:
            raise ValueError("density_estimator must be 'cutoff' or 'gaussian'")
        return rho
    
    def _compute_delta(self, distances, rhos):
        """
        Compute δ_i (delta_i) for each point.
        """
        N = distances.shape[0]
        delta = np.zeros(N)
        sorted_indices = np.argsort(-rhos)  # descending order of density

        # For the point with highest density, delta is maximum distance
        highest_density_idx = sorted_indices[0]
        delta[highest_density_idx] = np.max(distances[highest_density_idx, :])

        # For remaining points, compute delta as minimum distance to any point with higher density
        for pos in range(1, N):
            idx = sorted_indices[pos]
            # Consider only distances to points with higher density
            delta[idx] = np.min(distances[idx, sorted_indices[:pos]])

        return delta
    
    def _find_centers(self, rho, delta):
        """
        Identify cluster centers based on the product gamma = rho * delta.

        If self.n_clusters is provided, the algorithm selects that many centers
        by taking the points with the highest gamma scores. Otherwise, an automatic
        criterion is used: the gamma values are sorted, and the maximum drop between 
        consecutive sorted values is used to set the number of clusters.
        """
        # Compute gamma for every point
        gamma = rho * delta

        # Determine the number of centers
        if self.n_clusters is not None:
            n_centers = self.n_clusters
        else:
            # Automatic detection: sort gamma in descending order
            sorted_gamma = np.sort(gamma)[::-1]
            # Compute differences between successive gamma values
            diffs = sorted_gamma[:-1] - sorted_gamma[1:]
            # The number of clusters is chosen at the position with the maximum drop
            n_centers = np.argmax(diffs) + 1

        # Identify the indices of the n_centers points with largest gamma
        centers = np.argsort(-gamma)[:n_centers]
        return centers
    
    def _assign_clusters(self, distances, rhos, labels):
        """
        For points not yet labeled (i.e., not cluster centers), assign each point the same
        cluster as its nearest neighbor with higher density.
        """
        N = distances.shape[0]

        # Sort points by decreasing density
        sorted_indices = np.argsort(-rhos)

        # Array to store the nearest higher-density neighbor for each point
        nneigh = np.empty(N, dtype=int)
        # Highest-density point has no higher-density neighbor
        nneigh[sorted_indices[0]] = -1

        # Propagate labels in descending order of density
        for pos in range(1, N):
            idx = sorted_indices[pos]
            higher_density_indices = sorted_indices[:pos]
            nearest = higher_density_indices[np.argmin(distances[idx, higher_density_indices])]
            nneigh[idx] = nearest
            if labels[idx] == -1:
                labels[idx] = labels[nearest]

        return labels
    
    def _find_halo(self, distances, labels, rho):
        """
        Identify halo points for each cluster.
        """
        N = distances.shape[0]
        halo = np.zeros(N, dtype=bool)
        unique_labels = np.unique(labels)

        # The border matrix identifies for each point, those other points
        # in a different cluster that are within self.dc_ distance
        below_threshold = distances < self.dc_
        different_clusters = labels[:, None] != labels[None, :]
        border_matrix = below_threshold & different_clusters

        # Process each cluster individually
        for cl in unique_labels:
            cluster_mask = (labels == cl)

            # For each point in the current cluster, check if there exists any border point.
            border_exists = np.any(border_matrix[cluster_mask, :], axis=1)

            # If no border points exist, then all points are considered core (not halo)
            if not np.any(border_exists):
                continue

            # Compute the border density threshold (r_b) as the maximum density among border points.
            cluster_rho = rho[cluster_mask]
            rb = np.max(cluster_rho[border_exists])
            
            # In this cluster, points with density <= r_b are considered halo.
            halo[cluster_mask] = cluster_rho <= rb
        
        return halo