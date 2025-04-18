import numpy as np
from sklearn.cluster import AgglomerativeClustering, KMeans

class ClusteringAggregation:
    """
    Implementation of Clustering Aggregation algorithm based on the paper "Clustering Aggregation".
    This algorithm takes multiple clustering results and produces a consensus clustering.
    """
    
    def __init__(self, n_clusters=None):
        """
        Initialize the Clustering Aggregation algorithm.
        
        Parameters:
        -----------
        n_clusters : int, optional
            Number of clusters in the final clustering. If None, it will be determined automatically.
        """
        self.n_clusters = n_clusters
        
    def fit_predict(self, X):
        """
        Compute the clustering aggregation by:
        1. Computing clusters with 5 different techniques
        2. Creating a consensus clustering that maximizes agreement
        
        Parameters:
        -----------
        X : array-like
            Input data to cluster
            
        Returns:
        --------
        labels : array-like
            Cluster labels for each data point
        """
        # Compute clusters with 5 different techniques
        n_clusters = self.n_clusters or 7
        
        # 1. Single linkage
        single_linkage = AgglomerativeClustering(n_clusters=n_clusters, linkage='single')
        labels_single = single_linkage.fit_predict(X)
        
        # 2. Complete linkage
        complete_linkage = AgglomerativeClustering(n_clusters=n_clusters, linkage='complete')
        labels_complete = complete_linkage.fit_predict(X)
        
        # 3. Average linkage
        average_linkage = AgglomerativeClustering(n_clusters=n_clusters, linkage='average')
        labels_average = average_linkage.fit_predict(X)
        
        # 4. Ward's clustering
        ward_linkage = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward')
        labels_ward = ward_linkage.fit_predict(X)
        
        # 5. K-means
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        labels_kmeans = kmeans.fit_predict(X)
        
        # Store all clustering results
        all_clusterings = [labels_single, labels_complete, labels_average, labels_ward, labels_kmeans]
        
        # Compute co-association matrix
        n_samples = X.shape[0]
        co_association = np.zeros((n_samples, n_samples))
        
        for clustering in all_clusterings:
            for i in range(n_samples):
                for j in range(i+1, n_samples):
                    if clustering[i] == clustering[j]:
                        co_association[i, j] += 1
                        co_association[j, i] += 1
        
        # Normalize to get probabilities
        co_association /= len(all_clusterings)
        
        # Convert to distance matrix (1 - co_association)
        distance_matrix = 1 - co_association
        
        # Apply hierarchical clustering on the distance matrix
        agg_clustering = AgglomerativeClustering(n_clusters=n_clusters, 
                                               metric='precomputed', 
                                               linkage='average')
        aggregated_labels = agg_clustering.fit_predict(distance_matrix)
        
        return aggregated_labels