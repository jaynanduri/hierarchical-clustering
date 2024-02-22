import numpy as np
from sklearn.base import BaseEstimator, ClusterMixin
from scipy.linalg import norm
import heapq


class HierarchicalClustering(BaseEstimator, ClusterMixin):
    def __init__(self, k=1):
        self.clusters = k
        self.data = None
        self.inverse_idx = {}  # Maps the index of each point to its cluster centroid
        self.cluster_assignments = {}  # Maps cluster centroids to the indices of points in that cluster
        self.priorityQueue = []

    @staticmethod
    def compute_distance(mean_a, mean_b, n_a, n_b):
        # Corrected to use Ward's method formula for distance
        mu = np.array(mean_a) - np.array(mean_b)
        return norm(mu) ** 2 * (n_a * n_b / (n_a + n_b))

    def setup(self):
        # Initial cluster assignment for each point
        for index, point in enumerate(self.data):
            self.inverse_idx[index] = tuple(point)
            self.cluster_assignments[tuple(point)] = [index]
        # Compute pairwise distances for each point and insert into a priority queue
        n = len(self.data)
        for i in range(n - 1):
            for j in range(i + 1, n):
                mean_a, mean_b = self.data[i], self.data[j]  # Use raw data points as initial means
                n_a, n_b = 1, 1  # Initially, each cluster contains one point
                distance = self.compute_distance(mean_a, mean_b, n_a, n_b)
                heapq.heappush(self.priorityQueue, (distance, (i, j)))  # Use indices rather than means

    def fit_predict(self, X, y=None):
        self.data = X
        self.setup()
        while self.priorityQueue and len(self.cluster_assignments) > self.clusters:
            distance, (i, j) = heapq.heappop(self.priorityQueue)  # Correctly pop the smallest distance item

            # Ensure clusters i and j have not already been merged
            if i not in self.inverse_idx or j not in self.inverse_idx:
                continue

            # Find and merge points in clusters i and j
            points_in_a = self.cluster_assignments.pop(self.inverse_idx[i])
            points_in_b = self.cluster_assignments.pop(self.inverse_idx[j])
            n_a, n_b = len(points_in_a), len(points_in_b)
            new_points = points_in_a + points_in_b
            mean_a, mean_b = self.inverse_idx[i], self.inverse_idx[j]
            # Recalculate mean for the new cluster
            new_mean = np.array(mean_a) - ((np.array(mean_b) - np.array(mean_a)) * n_b/(n_a + n_b))

            # Update cluster assignments and inverse index
            for p in new_points:
                self.inverse_idx[p] = tuple(new_mean)
            self.cluster_assignments[tuple(new_mean)] = new_points

            # Recompute distances to the new cluster
            for k in set(self.inverse_idx.values()):
                if k == tuple(new_mean):
                    continue
                # mean_k = np.mean([self.data[p] for p in points_in_k], axis=0)
                n_k = len(self.cluster_assignments[k])
                n_new = len(new_points)
                distance = self.compute_distance(new_mean, k, n_new, n_k)
                heapq.heappush(self.priorityQueue, (distance, (tuple(new_mean), k)))

        return self.cluster_assignments