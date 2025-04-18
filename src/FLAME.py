"""
FLAME - Fuzzy clustering by Local Approximation of MEmbership
"""
from __future__ import print_function

import numpy as np
from scipy import sparse
from scipy.sparse import lil_matrix
from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.utils import check_array
from scipy.spatial.distance import cdist
from sklearn.preprocessing import normalize
from math import sqrt

class FLAME(BaseEstimator, ClusterMixin):
	def __init__(self, metric="euclidean", cluster_neighbors=5, iteration_neighbors=5, max_iter=np.inf, eps=1e-10, thd=-2, verbose=0):
		self.metric = metric
		self.cluster_neighbors = cluster_neighbors
		self.iteration_neighbors = iteration_neighbors
		self.max_iter = max_iter
		self.eps = eps
		self.thd = thd
		self.verbose = verbose

	def _get_nearest(self, distances, n_neighbors, n_samples):
		sample_range = np.arange(n_samples)[:, None]
		nearest_np = np.argpartition(distances, n_neighbors, axis=1)
		nearest_np = nearest_np[:, :n_neighbors + 1]

		largest_distance = distances[sample_range, nearest_np[sample_range, -1]]

		nearest = []
		nearest_with_ties = []
		for i in range(n_samples):
			ties_for_largest_distance = np.where(distances[i] == largest_distance[i])
			nearest.append(set(nearest_np[i, :].tolist()))
			nearest[-1].remove(i)
			ties_for_largest_distance = set(ties_for_largest_distance[0].tolist())
			ties_for_largest_distance.discard(i)
			nearest_with_ties.append(nearest[i] | ties_for_largest_distance)
		return nearest, nearest_with_ties

	def _get_densities(self, distances, nearest, n_samples):
		sample_range = np.arange(n_samples)[:, None]
		nearest_np = np.array([list(s) for s in nearest])
		n_shortest_distances = distances[sample_range, nearest_np]
		local_distance_sums = n_shortest_distances.sum(axis=1)
		largest_local_sum = local_distance_sums.max(axis=0)
		densities = np.asarray(largest_local_sum / local_distance_sums)
		return densities

	def _get_supports(self, densities, nearest_with_ties, n_samples):
		density_sum = densities.sum()
		density_mean = density_sum / n_samples
		density_sum2 = (densities * densities).sum()
		thd = density_mean + self.thd * sqrt(density_sum2 / n_samples - density_mean * density_mean)
		csos = []
		outliers = []
		remaining = []
		for i in range(n_samples):
			if densities[i] < thd:
				outliers.append(i)
			elif densities[i] > densities[list(nearest_with_ties[i])].max():
				csos.append(i)
			else:
				remaining.append(i)
		return csos, outliers, remaining

	def _get_weights(self, distances, nearest_with_ties, fixed, n_samples):
		nearest_with_ties = [sorted(list(s)) for s in nearest_with_ties]
		weights = lil_matrix((n_samples, n_samples))
		for i in range(n_samples):
			if i in fixed:
				weights[i, i] = 1
			else:
				for j in nearest_with_ties[i]:
					weights[i, j] = distances[i, j]
			if self.verbose: print("Assigned weights {0}.".format(i))
		weights = weights.tocsr()
		weights = normalize(weights, norm='l1', axis=1, copy=False)
		return weights

	def _get_starting_membership(self, csos, outliers, fixed, n_samples):
		M = len(csos) + 1
		starting_membership = np.zeros(shape=(n_samples, M))
		general_row = np.ndarray(shape=(1, M))
		general_row.fill(1. / M)
		for i in range(n_samples):
			if i not in fixed:
				starting_membership[i, :] = general_row
		for index, value in enumerate(csos):
			starting_membership[value, index] = 1
		for i in outliers:
			starting_membership[i, -1] = 1
		return starting_membership

	def _flame(self, X):
		if sparse.issparse(X) and self.metric not in {"precomputed", "cityblock", "cosine", "euclidean", "l1", "l2",
												 "manhattan"} and not callable(self.metric):
			raise TypeError("The metric {0} does not support sparse data.".format(self.metric))

		if 'pandas' in str(X.__class__):
			X = X.values

		X = check_array(X, accept_sparse="csr", dtype=None)
		n_samples, _ = X.shape
		distances = cdist(X, X, metric=self.metric)
		nearest, nearest_with_ties = self._get_nearest(distances, self.cluster_neighbors, n_samples)
		if self.verbose: print("Got distances and nearest.")
		densities = self._get_densities(distances, nearest, n_samples)
		if self.verbose: print("Got densities.")
		csos, outliers, _ = self._get_supports(densities, nearest_with_ties, n_samples)
		if self.verbose: print("Got suppports.")
		if self.verbose: print("There are {0} clusters and {1} outliers.".format(len(csos), len(outliers)))
		fixed = set(csos) | set(outliers)
		_, nearest_with_ties_for_iteration = self._get_nearest(distances, self.iteration_neighbors, n_samples)
		weights = self._get_weights(distances, nearest_with_ties_for_iteration, fixed, n_samples)
		if self.verbose: print("Got weights.")
		membership_proba = self._get_starting_membership(csos, outliers, fixed, n_samples)
		if self.verbose: print("Got starting memberships.")
		i = 0
		while i < self.max_iter:
			lastMembership = membership_proba.copy()
			membership_proba = weights.dot(membership_proba)
			delta = np.absolute(membership_proba - lastMembership).max()
			i += 1
			if self.verbose: print("Done iteration {0}.".format(i))
			if delta < self.eps:
				break
		num_clusters = membership_proba.shape[1] - 1
		pred = np.argmax(membership_proba, axis=1)
		pred[pred == num_clusters] = -1
		return membership_proba, pred, csos, outliers, densities

	def fit(self, X):
		self.membership_proba_, self.labels_, self.csos_, self.outliers_, self.densities_ = \
			self._flame(X)

		return self

	def fit_predict(self, X, y=None):
		y = self.fit(X).labels_
		return y

	def fit_predict_proba(self, X, y=None):
		y = self.fit(X).membership_proba_
		return y