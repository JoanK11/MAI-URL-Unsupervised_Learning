from sklearn.cluster import DBSCAN
from sklearn.base import BaseEstimator, ClusterMixin

class DBSCANWrapper(BaseEstimator, ClusterMixin):
    def __init__(self, eps=0.5, min_samples=5):
        self.eps = eps
        self.min_samples = min_samples

    def fit(self, X, y=None):
        model = DBSCAN(eps=self.eps, min_samples=self.min_samples)
        self.labels_ = model.fit(X)
        return self

    def predict(self, X):
        return self.labels_

    def fit_predict(self, X, y=None):
        return self.fit(X).labels_