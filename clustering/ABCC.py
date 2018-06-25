import numpy as np
from optimization.ABC import ABC


class ABCCObjectiveFunction:
    def __init__(self, data, n_clusters, n_attributes):
        self.function = self.evaluate
        self.minf = 0.0
        self.maxf = 1.0
        self.n_clusters = n_clusters
        self.n_attributes = n_attributes
        self.data = data

    def evaluate(self, x):
        centroids = x.reshape((self.n_clusters, self.n_attributes))
        clusters = {}

        for k in range(self.n_clusters):
            clusters[k] = []

        for xi in self.data:
            # dist = [(np.linalg.norm(xi - centroids[c])**2) for c in range(len(centroids))]
            dist = [squared_euclidean_dist(xi, centroids[c]) for c in range(len(centroids))]
            class_ = dist.index(min(dist))
            clusters[class_].append(xi)

        return sse(centroids, clusters)


def sse(centroids, clusters):
    global_intra_cluster_sum = 0.0

    for c in range(len(centroids)):
        partial_intra_cluster_sum = 0.0

        if len(clusters[c]) > 0:
            for point in clusters[c]:
                partial_intra_cluster_sum += (squared_euclidean_dist(point, centroids[c]))

        global_intra_cluster_sum += partial_intra_cluster_sum

    return global_intra_cluster_sum


def squared_euclidean_dist(u, v):
    sed = ((u - v) ** 2).sum()
    return sed


class ABCC(object):
    def __init__(self, n_clusters=2, swarm_size=50, n_iter=20, trials_limit=100):
        self.n_clusters = n_clusters
        self.swarm_size = swarm_size
        self.n_iter = n_iter
        self.trials_limit = trials_limit

    def fit(self, data):

        self.n_attributes = data.shape[1]

        self.abc = ABC(ABCCObjectiveFunction(data, self.n_clusters, self.n_attributes),
                       dim=self.n_clusters * self.n_attributes,
                       colony_size=self.swarm_size,
                       n_iter=self.n_iter,
                       trials_limit=self.trials_limit)

        self.abc.optimize()

        self.centroids = {}
        raw_centroids = self.abc.gbest.pos.reshape((self.n_clusters, self.n_attributes))

        self.convergence = self.abc.optimum_cost_tracking_iter

        for c in range(len(raw_centroids)):
            self.centroids[c] = raw_centroids[c]

        self.clusters = self.get_clusters(self.centroids, data)

        self.number_of_effective_clusters = 0

        for c in range(len(self.centroids)):
            if len(self.clusters[c]) > 0:
                self.number_of_effective_clusters = self.number_of_effective_clusters + 1

    @staticmethod
    def get_clusters(centroids, data):

        clusters = {}
        for c in centroids:
            clusters[c] = []

        for xi in data:
            dist = [np.linalg.norm(xi - centroids[c]) for c in centroids]
            class_ = dist.index(min(dist))
            clusters[class_].append(xi)

        return clusters
