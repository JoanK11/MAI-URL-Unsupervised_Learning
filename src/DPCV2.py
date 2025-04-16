import numpy as np
from DensityPeaksClustering import DensityPeaksClustering
from sklearn.metrics import pairwise_distances

class DPCV(DensityPeaksClustering):
    """
    Density Peaks Clustering based on Variance (DPCV).
    
    This implementation extends the original Density Peaks Clustering algorithm
    by replacing the Euclidean distance-based similarity measurement with one
    that uses the sample variance between points.
    
    Parameters:
    ----------
    dc: float, default=None
        Cutoff distance for density estimation. If None, estimated automatically.

    percent : float, default=2.0
        Percentage used to estimate dc (average number of neighbors ~ percent% of N).

    n_clusters : int, default=None
        Number of clusters. If None, estimated automatically.
    
    density_estimator : str, default='gaussian'
        Method for density estimation. Only 'gaussian' is supported in DPCV.
    """
    
    def __init__(
        self,
        dc=None,
        percent=2.0,
        n_clusters=None,
        density_estimator='gaussian',
    ):
        # Force density estimator to be 'gaussian' for DPCV
        super().__init__(
            dc=dc,
            percent=percent,
            n_clusters=n_clusters,
            density_estimator='gaussian'
        )
    
    def fit(self, X, y=None):
        """
        Fit the model to the data using variance-based similarity.
        """
        # Store the dataset
        self.X_ = X
        N = X.shape[0]
        
        # Calculate Euclidean distance matrix
        self.distances_ = pairwise_distances(X, metric='euclidean')
        
        # Calculate variance matrix between all pairs of points
        self.variance_matrix_ = self._compute_variance_matrix(X)
        
        # Estimate dc if not provided
        self.dc_ = self.dc if self.dc is not None else self._estimate_dc(self.distances_)
        
        # Compute local densities using variance-based approach
        self.rho_ = self._compute_density_variance(self.distances_, self.variance_matrix_)
        
        # Compute delta values (distance to nearest point of higher density)
        self.delta_ = self._compute_delta(self.distances_, self.variance_matrix_, self.rho_)
        
        # Identify cluster centers
        self.centers_ = self._find_centers(self.rho_, self.delta_)
        self.n_clusters_ = len(self.centers_) if self.n_clusters is None else self.n_clusters
        
        # Initialize labels for centers
        self.labels_ = -1 * np.ones(N, dtype=int)
        self.labels_[self.centers_] = np.arange(self.n_clusters_)
        
        # Assign the rest of points to cluster centers
        self.labels_ = self._assign_clusters(self.distances_, self.variance_matrix_, self.rho_, self.labels_)
        
        # Identify halo points for the clusters
        self.halo_ = self._find_halo(self.distances_, self.variance_matrix_, self.labels_, self.rho_)
        
        return self
    
    def _compute_variance_matrix(self, X):
        """
        Compute the variance matrix between all pairs of points.
        
        For points x_i and x_j in m-dimensional space, we compute:
        σ²_ij = (1/(m-1)) * Σ[(x_ip - x_jp - μ)²]
        
        where μ = (1/m) * Σ(x_iq - x_jq)
        """
        N, m = X.shape
        
        # Expand dimensions for broadcasting: (N, 1, m) and (1, N, m)
        X_i = X[:, np.newaxis, :]
        X_j = X[np.newaxis, :, :]
        
        # Calculate differences (N, N, m)
        differences = X_i - X_j
        
        # Calculate mu (mean difference across dimensions) (N, N, 1)
        # Keepdims=True is important for broadcasting in the next step
        mu = np.mean(differences, axis=2, keepdims=True)
        
        # Calculate sum of squared differences from mean (N, N)
        squared_diff_from_mean = np.sum((differences - mu) ** 2, axis=2)
        
        # Calculate variance
        if m > 1:
            variance_matrix = squared_diff_from_mean / (m - 1)
        else:
            # Handle m=1 case (variance is just the squared difference)
            variance_matrix = squared_diff_from_mean 
            
        # Ensure diagonal is zero (variance of a point with itself)
        np.fill_diagonal(variance_matrix, 0)
            
        return variance_matrix
    
    def _compute_density_variance(self, distances, variance_matrix):
        """
        Compute local density ρ_i for each point using the formula (vectorized):
        
        ρ_i = Σ exp(-(d_ij / (d_c * φ_ij))²)
        
        where φ_ij = σ_ij/2 + 1, which adjusts the variance and avoids errors 
        when variance is zero.
        """
        N = variance_matrix.shape[0]
        
        # Calculate φ_ij = σ_ij/2 + 1, ensuring variance is non-negative
        # Add small epsilon to variance before sqrt for numerical stability if needed
        # but variance should theoretically be non-negative. Using maximum(0, ...) is safer.
        sigma_ij = np.sqrt(np.maximum(0, variance_matrix)) 
        phi_ij = sigma_ij / 2.0 + 1.0
        
        # Avoid division by zero on the diagonal (phi_ij will be 1 on diagonal)
        # The exponent term for i==j will be -(0 / (dc * 1))**2 = 0, exp(0)=1
        # We need to exclude this self-contribution from the sum.
        
        # Calculate exponent term, handle potential division by zero if dc_ * phi_ij is zero
        # (unlikely as dc > 0 and phi_ij >= 1)
        denominator = self.dc_ * phi_ij
        # Prevent division by zero, although unlikely here
        denominator[denominator == 0] = 1e-12 # Avoid division by zero
        exponent = -(distances / denominator) ** 2
        
        # Calculate density sum using Gaussian kernel, excluding self-contribution (i==j)
        # np.fill_diagonal doesn't work directly on the exp result as it modifies inplace.
        # Create a mask for the diagonal
        identity = np.eye(N, dtype=bool)
        exp_term = np.exp(exponent)
        exp_term[identity] = 0 # Set diagonal contribution to 0 before summing
        
        rho = np.sum(exp_term, axis=1)
        
        return rho
    
    def _estimate_dc(self, distances):
        """
        Estimate cutoff distance dc using the Euclidean distance matrix.
        """
        N = distances.shape[0]
        p = self.percent / 100.0
        
        # Extract upper triangular distances (excluding diagonal)
        # Use pre-calculated distances matrix
        if N > 1:
            all_distances = distances[np.triu_indices(N, k=1)]
            # Compute the p-quantile of the distances
            # Ensure all_distances is not empty before calling quantile
            if all_distances.size > 0:
                 dc = np.quantile(all_distances, p)
            else: # Handle case with only one point
                 dc = np.mean(distances) # Or some default, e.g., 1.0
        else:
            # Handle case with only one point or no points
             dc = 1.0 # Default value if no distances to compare

        # Ensure dc is not zero to avoid division by zero later
        return max(dc, 1e-12) # Return dc or a small positive value
        
    def fit_predict(self, X, y=None):
        """
        Fit the model and return the predicted labels for the data.
        """
        return self.fit(X).labels_
        
    def _find_halo(self, distances, variance_matrix, labels, rho):
        """
        Identify halo points for each cluster using the variance-based approach (vectorized).
        """
        N = variance_matrix.shape[0]
        halo = np.zeros(N, dtype=bool)
        unique_labels = np.unique(labels[labels != -1]) # Exclude unassigned points if any remain
        
        if not unique_labels.size: # No clusters found
            return halo # Return all false

        # Calculate φ_ij = σ_ij/2 + 1
        sigma_ij = np.sqrt(np.maximum(0, variance_matrix))
        phi_ij = sigma_ij / 2.0 + 1.0
        
        # Adjust distance by the variance factor for border check
        # Using broadcasting for efficiency
        adjusted_dc_threshold = self.dc_ * phi_ij
        
        # Identify border points: distance < adjusted dc AND different labels
        # Use broadcasting for label comparison: (N, 1) vs (1, N) -> (N, N)
        is_border = (distances < adjusted_dc_threshold) & (labels[:, np.newaxis] != labels[np.newaxis, :])
        # Ensure diagonal is False (a point cannot be a border point with itself)
        np.fill_diagonal(is_border, False)
        
        # Process each cluster individually
        for cl in unique_labels:
            cluster_mask = (labels == cl)
            
            # Find points within the cluster that have at least one border point connection
            # Check rows corresponding to cluster points against all columns
            has_border_connection = np.any(is_border[cluster_mask, :], axis=1)
            
            # If no points in the cluster have border connections, skip (all are core)
            if not np.any(has_border_connection):
                continue
                
            # Identify the indices within the original data that correspond to these border points
            cluster_indices = np.where(cluster_mask)[0]
            border_point_indices_in_cluster = cluster_indices[has_border_connection]

            # Compute the border density threshold (rb)
            # rb is the maximum density among points *in this cluster* that have a border connection
            if border_point_indices_in_cluster.size > 0:
                 rb = np.max(rho[border_point_indices_in_cluster])
            else:
                 # This case should technically be covered by the 'continue' above, but for safety:
                 rb = -np.inf 

            # In this cluster, points with density <= rb are considered halo
            # Apply only to points within the current cluster
            halo[cluster_mask] = (rho[cluster_mask] <= rb) & cluster_mask[cluster_mask] # Ensure mask alignment
            
        return halo

    def _assign_clusters(self, distances, variance_matrix, rhos, labels):
        """
        Assign points to clusters based on nearest higher-density neighbor (vectorized distance).
        """
        N = variance_matrix.shape[0]
        
        # Calculate φ_ij = σ_ij/2 + 1
        sigma_ij = np.sqrt(np.maximum(0, variance_matrix))
        phi_ij = sigma_ij / 2.0 + 1.0
        
        # Adjust distances by the variance factor
        # Avoid division by zero (phi_ij >= 1)
        # Set diagonal to infinity to prevent selecting self as neighbor
        adjusted_distances = distances / phi_ij
        np.fill_diagonal(adjusted_distances, np.inf)

        # Sort points by decreasing density
        sorted_indices = np.argsort(-rhos)
        
        # Assign labels based on nearest higher-density neighbor
        # This part still requires iteration in order of density,
        # but finding the minimum distance is vectorized.
        nneigh = np.empty(N, dtype=int) # Store nearest neighbor index
        
        for i, idx in enumerate(sorted_indices):
            if labels[idx] != -1: # Skip cluster centers (already labeled)
                continue
                
            if i == 0: # Highest density point (should be a center, but handle edge case)
                 # This point should already be labeled as a center. If not, something is wrong.
                 # Or, if only one point, it has no higher density neighbor.
                 # Assign it to its own cluster or handle as an error/outlier?
                 # Based on original DPC, highest density point becomes a center.
                 # If somehow it reaches here unlabeled, we might assign it cluster 0?
                 # For now, assume centers are correctly pre-labeled.
                 # We could raise an error if the highest density point is unlabeled.
                 # Or find the max distance point as its 'neighbor' conceptually?
                 # Let's rely on centers being labeled. If i==0 and unlabeled, it's an issue.
                 # Safest might be to skip, assuming it was handled, or assign arbitrarily if needed.
                 # Revisit if this scenario causes issues.
                 continue # Should be labeled as a center

            # Indices of points with higher density than current point idx
            higher_density_indices = sorted_indices[:i]
            
            # Find the index (among higher_density_indices) of the point with minimum adjusted distance to idx
            # adjusted_distances[idx, higher_density_indices] gives distances from idx to all higher density points
            if higher_density_indices.size > 0:
                min_dist_idx_in_higher = np.argmin(adjusted_distances[idx, higher_density_indices])
                # Get the actual index in the full dataset
                nearest_neighbor_idx = higher_density_indices[min_dist_idx_in_higher]
                
                # Assign the label of the nearest higher-density neighbor
                labels[idx] = labels[nearest_neighbor_idx]
                nneigh[idx] = nearest_neighbor_idx # Optional: store neighbor
            else:
                # This case (point is not highest density but has no points with higher density)
                # should not happen if densities are unique. If densities are tied, argsort order matters.
                # If it happens, assign to itself or handle as outlier?
                # Assigning to nearest overall neighbor might be an alternative.
                # For now, assign to a default 'unassigned' or handle based on DPC paper specifics for ties.
                # Let's assume it inherits from the closest labeled point found so far if needed.
                # Assigning label -1 might be safest if this edge case isn't clearly defined.
                # Given centers are labeled, this point *should* find a path back to a center.
                 pass # Keep label -1 for now, or implement tie-breaking/outlier handling

        return labels

    def _compute_delta(self, distances, variance_matrix, rhos):
        """
        Compute δ_i (delta_i) for each point using the variance-based approach (vectorized distance).
        """
        N = variance_matrix.shape[0]
        delta = np.zeros(N)
        nneigh_delta = np.empty(N, dtype=int) # Store index of nearest higher-density neighbor for delta

        # Calculate φ_ij = σ_ij/2 + 1
        sigma_ij = np.sqrt(np.maximum(0, variance_matrix))
        phi_ij = sigma_ij / 2.0 + 1.0
        
        # Adjust distances by the variance factor
        # Set diagonal to infinity
        adjusted_distances = distances / phi_ij
        np.fill_diagonal(adjusted_distances, np.inf)
        
        # Sort points by decreasing density
        sorted_indices = np.argsort(-rhos)
        
        # Highest density point: delta is max adjusted distance to any other point
        highest_density_idx = sorted_indices[0]
        if N > 1:
             delta[highest_density_idx] = np.max(adjusted_distances[highest_density_idx, :])
             # The 'neighbor' for delta calculation for the highest density point is conventionally itself or undefined.
             # Sometimes the point furthest away is stored. Let's store the furthest point's index.
             nneigh_delta[highest_density_idx] = np.argmax(adjusted_distances[highest_density_idx, :]) 
        else:
             delta[highest_density_idx] = 0 # Only one point
             nneigh_delta[highest_density_idx] = 0


        # For remaining points: delta is min adjusted distance to any point with *higher* density
        for i in range(1, N):
            idx = sorted_indices[i]
            higher_density_indices = sorted_indices[:i]
            
            # Find minimum adjusted distance to points with higher density
            if higher_density_indices.size > 0:
                 min_dist_idx_in_higher = np.argmin(adjusted_distances[idx, higher_density_indices])
                 delta[idx] = adjusted_distances[idx, higher_density_indices[min_dist_idx_in_higher]]
                 nneigh_delta[idx] = higher_density_indices[min_dist_idx_in_higher]
            else:
                 # Should not happen if densities are unique and N > 1
                 # If it does, delta might be considered infinite or max distance.
                 # Setting to max distance is consistent with the highest density point logic.
                 delta[idx] = np.max(adjusted_distances[idx, :]) # Fallback: like highest density point
                 nneigh_delta[idx] = np.argmax(adjusted_distances[idx, :]) # Fallback


        # Store nearest neighbor indices used for delta calculation (optional)
        self.nneigh_delta_ = nneigh_delta

        return delta
