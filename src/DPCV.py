import numpy as np
from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.metrics import pairwise_distances

class DPCV(BaseEstimator, ClusterMixin):
    """
    Improved Density Peaks Clustering based on Variance (DPCV) algorithm.
    
    Parameters:
    ----------
    n_clusters : int, default=2
        Number of clusters (NCLUST)
    
    n_neighbors : int, default=5
        Number of neighbors (K) used for KNN calculation
        
    alpha : float, default=0.5
        Alpha parameter used in point allocation (Step 8)
    """
    def __init__(self, n_clusters=2, n_neighbors=5, alpha=0.5):
        self.n_clusters = n_clusters
        self.n_neighbors = n_neighbors
        self.alpha = alpha
        
    def fit(self, X, y=None):
        """
        Fit the DPCV algorithm to the data.
        
        Parameters:
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data
            
        Returns:
        -------
        self : object
            Returns self
        """
        # Step 1: Calculate Euclidean distance between points
        self.distance_matrix_ = pairwise_distances(X)
        n_samples, n_features = X.shape # Get number of features
        
        # Step 2: Calculate variance matrix using formula (8)
        self.variance_matrix_ = np.zeros((n_samples, n_samples))
        for i in range(n_samples):
            for j in range(i + 1, n_samples): # Optimization: Calculate only for j > i
                # Calculate difference vector between features of points i and j
                diff = X[i] - X[j]
                
                # Calculate μ (mean of feature differences) using formula (9)
                # mu = (1/m) * sum(x_iq - x_jq for q=1 to m)
                mu = np.mean(diff)
                
                # Calculate σ² (variance of feature differences) using formula (8)
                # sigma_sq = (1/(m-1)) * sum((x_ip - x_jp - mu)^2 for p=1 to m)
                # Use ddof=1 for sample variance (division by m-1)
                if n_features > 1:
                    sigma_sq = np.var(diff, ddof=1) 
                else:
                    sigma_sq = 0 # Variance is undefined for a single feature
                
                self.variance_matrix_[i, j] = sigma_sq
                self.variance_matrix_[j, i] = sigma_sq # Matrix is symmetric
        
        # Step 3: Calculate KNN of each point
        self.knn_indices_ = np.zeros((n_samples, self.n_neighbors), dtype=int)
        for i in range(n_samples):
            self.knn_indices_[i] = np.argsort(self.distance_matrix_[i])[1:self.n_neighbors+1]
            
        # Step 4: Calculate local density ρ using formula (10)
        self.rho_ = np.zeros(n_samples)
        for i in range(n_samples):
            for j in range(n_samples):
                if i != j:
                    d_ij = self.distance_matrix_[i, j]
                    d_c_ij = self.distance_matrix_[i, self.knn_indices_[i][-1]]
                    self.rho_[i] += np.exp(-(d_ij/d_c_ij)**2)
        
        # Calculate relative distance δ using formula (3)
        self.delta_ = np.zeros(n_samples)
        for i in range(n_samples):
            higher_density_points = np.where(self.rho_ > self.rho_[i])[0]
            lower_density_points = np.where(self.rho_ < self.rho_[i])[0]
            
            if len(higher_density_points) > 0:
                # δ_i = min_{j:ρ_j>ρ_i} d_ij
                self.delta_[i] = np.min(self.distance_matrix_[i, higher_density_points])
            else:
                # For the highest density point, δ_i = max_{j:ρ_j<ρ_i} d_ij
                if len(lower_density_points) > 0:
                    self.delta_[i] = np.max(self.distance_matrix_[i, lower_density_points])
                else:
                    self.delta_[i] = np.max(self.distance_matrix_[i])
                    
        # Step 5: Calculate decision index γ using formula (4) and select cluster centers
        self.gamma_ = self.rho_ * self.delta_
        self.centers_ = np.argsort(-self.gamma_)[:self.n_clusters]
        
        # Step 6: Sort cluster centers in descending order of γ
        self.centers_ = self.centers_[np.argsort(-self.gamma_[self.centers_])]
        
        # Initialize labels (unassigned = -1)
        self.labels_ = -np.ones(n_samples, dtype=int)
        
        # Assign centers to their respective clusters
        for i, center in enumerate(self.centers_):
            self.labels_[center] = i
            
        # Initialize queried points list
        queried = np.zeros(n_samples, dtype=bool)
        queried[self.centers_] = True
        
        # Step 7-9: Cluster assignment process
        for i, center in enumerate(self.centers_):
            # Calculate search radius d_C(i),K using formula (12)
            d_C_i_K = np.max(self.distance_matrix_[center, self.knn_indices_[center]])
            
            # Find unallocated points within search radius
            unallocated_mask = (self.labels_ == -1)
            in_radius_mask = self.distance_matrix_[center] <= d_C_i_K
            points_to_assign = np.where(unallocated_mask & in_radius_mask)[0]
            
            # Assign these points to the current cluster
            self.labels_[points_to_assign] = i
            
            # Process the sub-cluster (Steps 8-9)
            unprocessed_queue = list(points_to_assign)
            while unprocessed_queue:
                j = unprocessed_queue.pop(0)
                if queried[j]:
                    continue
                    
                queried[j] = True
                
                # Check density condition: ρ_j > α * ρ_C(i)
                if self.rho_[j] > self.alpha * self.rho_[center]:
                    # Find unallocated points within d_C(i),K of point j
                    in_j_radius_mask = self.distance_matrix_[j] <= d_C_i_K
                    points_to_assign_j = np.where(unallocated_mask & in_j_radius_mask)[0]
                    
                    # Assign these points to the current cluster
                    self.labels_[points_to_assign_j] = i
                    
                    # Add newly assigned points to the processing queue
                    for point in points_to_assign_j:
                        if not queried[point] and point not in unprocessed_queue:
                            unprocessed_queue.append(point)
        
        # Step 10: For remaining unallocated points, use the DPC allocation strategy
        # Sort points by decreasing density
        remaining_unallocated = np.where(self.labels_ == -1)[0]
        
        if len(remaining_unallocated) > 0:
            # Sort all points by decreasing density
            sorted_indices = np.argsort(-self.rho_)
            
            # For each point with undefined cluster, assign it to the same cluster as its
            # nearest neighbor with higher density
            for i in sorted_indices:
                if self.labels_[i] == -1:
                    # Find neighbors with higher density
                    higher_density_neighbors = np.where(self.rho_ > self.rho_[i])[0]
                    if len(higher_density_neighbors) > 0:
                        # Find the nearest one
                        nearest_higher_density = higher_density_neighbors[
                            np.argmin(self.distance_matrix_[i, higher_density_neighbors])
                        ]
                        # Assign the same cluster
                        self.labels_[i] = self.labels_[nearest_higher_density]
                    else:
                        # If no higher density neighbors, assign to the cluster of the nearest center
                        self.labels_[i] = np.argmin(self.distance_matrix_[i, self.centers_])
        
        return self
    
    def fit_predict(self, X, y=None):
        """
        Fit the model and return the predicted labels.
        
        Parameters:
        ----------
        X : array-like of shape (n_samples, n_features)
            Input data
            
        Returns:
        -------
        labels : ndarray of shape (n_samples,)
            Cluster labels
        """
        self.fit(X)
        return self.labels_ 