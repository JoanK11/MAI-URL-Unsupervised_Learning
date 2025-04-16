import numpy as np
from sklearn.metrics import pairwise_distances
from DensityPeaksClustering import DensityPeaksClustering


class DPCV(DensityPeaksClustering):
    """
    Density Peaks Clustering based on Variance (DPCV)

    This implementation adjusts the similarity measurement by replacing
    the Euclidean-distance–based approach with one that uses the sample
    variance between points. It also implements an improved allocation
    strategy for non-center points based on a density core expansion.
    
    Parameters:
    -----------
    dc : float, default=None
        Cutoff distance for density estimation. If None, estimated automatically.

    percent : float, default=1.8
        Percentage used to estimate dc (as recommended for the S3 dataset).

    n_clusters : int, default=None
        Number of clusters. If None, estimated automatically.

    density_estimator : str, default='cutoff'
        Method for density estimation. (This implementation works with the
        variance-based adjustment regardless of the method string.)

    K : int, default=1
        The number of nearest neighbors used to compute the search radius 
        for the density core allocation stage.

    alpha : float, default=0.4
        Threshold factor for density core expansion. A point j is added to a 
        cluster if ρⱼ > alpha * ρ(center).
    """
    
    def __init__(self, dc=None, percent=1.8, n_clusters=None, density_estimator='cutoff', K=1, alpha=0.8):
        self.dc = dc
        self.percent = percent
        self.n_clusters = n_clusters
        self.density_estimator = density_estimator
        self.K = K
        self.alpha = alpha

    def _compute_variance_matrix(self, X):
        """
        Compute the pairwise variance-based adjustment factor φᵢⱼ.
        
        Parameters:
        -----------
        X : ndarray of shape (N, m)
            Input data matrix.

        Returns:
        --------
        phi : ndarray of shape (N, N)
            Matrix computed as φᵢⱼ = (σᵢⱼ / 2) + 1, where σᵢⱼ is the standard deviation
            derived from the sample variance between points i and j.
        """
        N, m = X.shape
        # Compute all pairwise differences: shape (N, N, m)
        diff = X[:, np.newaxis, :] - X[np.newaxis, :, :]
        # Compute the mean difference μᵢⱼ for each pair: shape (N, N)
        mean_diff = np.mean(diff, axis=2)
        # Compute sample variance for each pair using: σ²ᵢⱼ = (1/(m-1)) * Σ[(diff - μᵢⱼ)²]
        variance = np.sum((diff - mean_diff[..., np.newaxis])**2, axis=2) / (m - 1)
        # Standard deviation
        sigma = np.sqrt(variance)
        # Compute the adjustment factor: φᵢⱼ = (σᵢⱼ / 2) + 1
        phi = sigma / 2 + 1
        return phi

    def _compute_density(self, distances, X):
        """
        Compute local density for each point using the variance-adjusted similarity.
        
        Parameters:
        -----------
        distances : ndarray of shape (N, N)
            Euclidean distance matrix.
        X : ndarray of shape (N, m)
            Original data matrix.

        Returns:
        --------
        rho : ndarray of shape (N,)
            Local density computed as:
              ρᵢ = Σⱼ exp[-(dᵢⱼ / (d_c · φᵢⱼ))²]
        """
        phi = self._compute_variance_matrix(X)
        rho = np.sum(np.exp(-(distances / (self.dc_ * phi))**2), axis=1)
        return rho

    def _allocate_with_density_core(self, distances, rho, centers):
        """
        Allocate non-center points using a density core expansion strategy.
        
        For each cluster center, compute a search radius R based on the
        distance from the center to its k-th nearest neighbor. Then, iteratively,
        allocate any unassigned point j to that center's cluster if:
        
            (i) d(center, j) < R  and 
            (ii) ρⱼ > alpha * ρ(center)
        
        Parameters:
        -----------
        distances : ndarray of shape (N, N)
            The pairwise Euclidean distance matrix.
        rho : ndarray of shape (N,)
            Array of local densities.
        centers : array-like
            Indices of the cluster centers.

        Returns:
        --------
        labels : ndarray of shape (N,)
            Cluster labels after density core allocation. Unallocated points
            remain labeled as -1.
        """
        N = distances.shape[0]
        labels = -1 * np.ones(N, dtype=int)
        
        # Assign unique cluster labels to the centers.
        for i, center in enumerate(centers):
            labels[center] = i

        # Process centers in descending order of density.
        sorted_centers = sorted(centers, key=lambda idx: rho[idx], reverse=True)
        
        for center in sorted_centers:
            # Compute search radius R using the k-th nearest neighbor of the center.
            center_distances = distances[center]
            sorted_idx = np.argsort(center_distances)
            if len(sorted_idx) > self.K:
                R = center_distances[sorted_idx[self.K]]
            else:
                R = np.max(center_distances)

            # Initialize a queue with the center to perform a breadth-first expansion.
            queue = [center]
            while queue:
                current = queue.pop(0)
                for j in range(N):
                    # Only consider unassigned points.
                    if labels[j] == -1:
                        # Use the original center as the reference for the search radius.
                        if distances[center, j] < R and rho[j] > self.alpha * rho[center]:
                            labels[j] = labels[center]
                            queue.append(j)
        return labels

    def fit(self, X, y=None):
        """
        Fit the DPCV model to the data X.

        This method computes the pairwise distances, estimates the cutoff distance dc,
        computes the variance-adjusted local density, determines delta values, finds 
        cluster centers, and then allocates points using an improved two-stage allocation:
          1) Density core expansion (using hyperparameters K and alpha).
          2) Standard DPC assignment for any remaining unallocated points.
        
        Parameters:
        -----------
        X : ndarray of shape (N, m)
            Input data.
        y : Ignored

        Returns:
        --------
        self : object
            Fitted estimator.
        """
        distances = pairwise_distances(X, metric='manhattan')
        N = distances.shape[0]
        
        # Estimate the cutoff distance.
        self.dc_ = self.dc if self.dc is not None else self._estimate_dc(distances)
        
        # Compute the local density using the variance-adjusted formula.
        self.rho_ = self._compute_density(distances, X)
        
        # Compute delta values (distance to the nearest higher density point).
        self.delta_ = self._compute_delta(distances, self.rho_)
        
        # Identify cluster centers based on the product gamma = rho * delta.
        self.centers_ = self._find_centers(self.rho_, self.delta_)
        self.n_clusters_ = len(self.centers_)
        
        # First, allocate points using the density core allocation method.
        labels_core = self._allocate_with_density_core(distances, self.rho_, self.centers_)
        
        # For any remaining unallocated points (label == -1), fall back on the standard DPC assignment.
        self.labels_ = self._assign_clusters(distances, self.rho_, labels_core)
        
        # Identify halo points.
        self.halo_ = self._find_halo(distances, self.labels_, self.rho_)
        return self

    def fit_predict(self, X, y=None):
        """
        Fit the DPCV model to the data and return the cluster labels.
        """
        return self.fit(X, y).labels_