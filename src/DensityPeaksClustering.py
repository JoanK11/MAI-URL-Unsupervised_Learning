import numpy as np
from sklearn.base import BaseEstimator, ClusterMixin
from scipy.spatial.distance import cdist

import dtcwt
from skimage.metrics import structural_similarity
from scipy.signal import convolve2d

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

    similarity_metric : str, default='euclidean'
        Metric to compute pairwise distances: 'euclidean', 'cw-ssim', or 'ssim'.

    cwssim_level: int, default=6
        Number of DTCWT levels.

    cwssim_guardb: int, default=0
        Guard-band to discard at each side.
        
    cwssim_K: float, default=0.0
        Small constant in CW-SSIM formula.

    ssim_win_size: tuple, default=None
        Window size for SSIM computation.

    ssim_gaussian_weights: bool, default=False
        Whether to use Gaussian weights in SSIM computation.
    """
    def __init__(
        self,
        dc=None,
        percent=2.0,
        n_clusters=None,
        density_estimator='cutoff',
        similarity_metric='euclidean',
        cwssim_level=6,
        cwssim_guardb=0,
        cwssim_K=0.0,
        ssim_win_size=None,
        ssim_gaussian_weights=False,
        restrictive=False
    ):
        self.dc = dc
        self.percent = percent
        self.n_clusters = n_clusters
        self.density_estimator = density_estimator
        self.similarity_metric = similarity_metric
        self.cwssim_level = cwssim_level
        self.cwssim_guardb = cwssim_guardb
        self.cwssim_K = cwssim_K
        self.ssim_win_size = ssim_win_size
        self.ssim_gaussian_weights = ssim_gaussian_weights
        self.restrictive = restrictive
        
        # DT‑CWT transformer
        self._dtcwt = dtcwt.Transform2d()

        # precompute the uniform 7×7 window once
        self._cw_window = np.ones((7, 7), dtype=np.float32)
        self._cw_window /= self._cw_window.sum()

    def fit(self, X, y=None):
        if self.similarity_metric == 'cw-ssim':
            if X.ndim != 3:
                raise ValueError("X must be a 3D array of shape (n_samples, height, width) for cw-ssim metric.")
            self.image_shape_ = X.shape[1:]
            distances = self._compute_cwssim_distance_matrix(X)
        elif self.similarity_metric == 'ssim':
            if X.ndim != 3:
                raise ValueError("X must be a 3D array of shape (n_samples, height, width) for ssim metric.")
            self.image_shape_ = X.shape[1:]
            self.data_range_ = X.max() - X.min()
            distances = self._compute_ssim_distance_matrix(X)
        else:
            self.X_ = X
            distances = cdist(X, X, metric=self.similarity_metric)
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
        
        if self.similarity_metric in ('cw-ssim', 'ssim'):
            self.center_coords_ = X[self.centers_]
        else:
            self.center_coords_ = self.X_[self.centers_]

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

    def predict(self, X):
        """
        Predict the closest cluster for new samples.
        """
        if not hasattr(self, 'center_coords_'):
            raise RuntimeError("The model has not been fitted yet. Call 'fit' before 'predict'.")

        if self.similarity_metric == 'cw-ssim':
            # Support input as 2D feature vectors or 3D images
            imgs = self._prepare_images(X)
            center_imgs = self._prepare_images(self.center_coords_)
            n, k = len(imgs), len(center_imgs)
            distances_to_centers = np.zeros((n, k), dtype=float)
            for i, img in enumerate(imgs):
                for j, cimg in enumerate(center_imgs):
                    sim = self._cwssim_index(img, cimg)
                    distances_to_centers[i, j] = 1.0 - sim

        elif self.similarity_metric == 'ssim':
            if X.ndim != 3:
                raise ValueError("X must be a 3D array of shape (n_samples, height, width) for ssim metric.")
            n, k = X.shape[0], len(self.center_coords_)
            distances_to_centers = np.zeros((n, k))
            # Prepare SSIM keyword arguments
            sim_kwargs = {}
            if self.ssim_win_size is not None:
                sim_kwargs['win_size'] = self.ssim_win_size
            if self.ssim_gaussian_weights:
                sim_kwargs['gaussian_weights'] = True
            for i in range(n):
                for j in range(k):
                    sim = structural_similarity(
                        X[i], self.center_coords_[j], data_range=self.data_range_, **sim_kwargs)
                    distances_to_centers[i, j] = 1.0 - sim

        else:
            distances_to_centers = cdist(X, self.center_coords_, metric=self.similarity_metric)

        nearest = np.argmin(distances_to_centers, axis=1)
        return self.labels_[self.centers_][nearest]

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
                if self.similarity_metric in ('cw-ssim', 'ssim') and self.restrictive:
                    if distances[idx, nearest] < self.dc_:
                        labels[idx] = labels[nearest]
                else:
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
    
    def _prepare_images(self, X):
        """
        Ensure we have a list of 2D arrays.
        Supports X of shape (N, H, W) or (N, H*W).
        """
        if X.ndim == 3:
            return [x.astype(np.float32) for x in X]
        elif X.ndim == 2:
            N, feat = X.shape
            side = int(np.sqrt(feat))
            if side * side != feat:
                raise ValueError("Can't reshape feature vectors into square images")
            return [X[i].reshape(side, side).astype(np.float32) for i in range(N)]
        else:
            raise ValueError("X must be 2D or 3D array")

    def _compute_ssim_distance_matrix(self, X):
        """
        Compute pairwise 1 - SSIM distances for image dataset X using skimage.metrics.structural_similarity.
        """
        n = X.shape[0]
        dmat = np.zeros((n, n))
        # Prepare SSIM keyword arguments based on parameters
        sim_kwargs = {}
        if self.ssim_win_size is not None:
            sim_kwargs['win_size'] = self.ssim_win_size
        if self.ssim_gaussian_weights:
            sim_kwargs['gaussian_weights'] = True
        for i in range(n):
            for j in range(i+1, n):
                sim = structural_similarity(
                    X[i], X[j], data_range=self.data_range_, **sim_kwargs)
                d = 1.0 - sim
                dmat[i, j] = dmat[j, i] = d
        return dmat
    
    def _compute_cwssim_distance_matrix(self, X):
        imgs = self._prepare_images(X)
        N = len(imgs)
        D = np.zeros((N, N), dtype=float)
        for i in range(N):
            for j in range(i, N):
                sim = self._cwssim_index(imgs[i], imgs[j])
                d = 1.0 - sim
                D[i, j] = D[j, i] = d
        return D

    def _cwssim_index(self, img1, img2):
        """
        Compute CW-SSIM exactly over ALL high-pass subbands and all orientations,
        skipping any subband too small for the 7x7 window.
        """
        coeffs1 = self._dtcwt.forward(img1, nlevels=self.cwssim_level)
        coeffs2 = self._dtcwt.forward(img2, nlevels=self.cwssim_level)

        all_cssim = []

        # loop over each scale (high-pass subband)
        for lvl, (b1, b2) in enumerate(zip(coeffs1.highpasses, coeffs2.highpasses)):
            # 1) optional guard-band crop
            gb = int(self.cwssim_guardb / (2 ** lvl))
            if gb > 0:
                b1 = b1[gb:-gb, gb:-gb, :]
                b2 = b2[gb:-gb, gb:-gb, :]

            h, w, ori = b1.shape

            # skip if too small for 7×7 valid convolution
            if h < 7 or w < 7:
                continue

            # 2) precompute the Gaussian pooling weights for this subband
            h_loc = h - 7 + 1
            w_loc = w - 7 + 1
            y = np.arange(h_loc)
            x = np.arange(w_loc)
            Y, X = np.meshgrid(y, x, indexing='ij')
            cy, cx = (h_loc - 1) / 2.0, (w_loc - 1) / 2.0
            sigma = h / 4.0
            weight = np.exp(-(((Y - cy) ** 2 + (X - cx) ** 2) / (2 * sigma ** 2)))
            weight /= weight.sum()

            # 3) for each orientation
            for o in range(ori):
                c1 = b1[:, :, o]
                c2 = b2[:, :, o]

                corr = c1 * np.conj(c2)
                varr = np.abs(c1)**2 + np.abs(c2)**2

                # local sums via "valid" convolution
                corr_loc = convolve2d(corr, self._cw_window, mode='valid')
                varr_loc = convolve2d(varr, self._cw_window, mode='valid')

                # CW-SSIM map
                cssim_map = (2 * np.abs(corr_loc) + self.cwssim_K) / (varr_loc + self.cwssim_K)

                # weighted pooling
                all_cssim.append(np.sum(cssim_map * weight))

        if not all_cssim:
            return 0.0

        return float(np.mean(all_cssim))