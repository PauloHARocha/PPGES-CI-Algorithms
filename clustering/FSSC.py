import numpy as np
from optimization.FSS import FSS


class FSSCObjectiveFunction:
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


class FSSC(object):
    def __init__(self, n_clusters=2, swarm_size=50, n_iter=50):
        self.n_clusters = n_clusters
        self.swarm_size = swarm_size
        self.n_iter = n_iter
        self.weight_min = 1
        self.step_i_init = 0.1
        self.step_i_end = 0.001
        self.step_v_init = 0.01
        self.step_v_end = 0.001

    def fit(self, data):

        self.n_attributes = data.shape[1]

        self.fss = FSS(FSSCObjectiveFunction(data, self.n_clusters, self.n_attributes),
                       dim=self.n_clusters * self.n_attributes, school_size=self.swarm_size,
                       n_iter=self.n_iter, weight_min=self.weight_min, step_i_init=self.step_i_init,
                       step_i_end=self.step_i_end, step_v_init=self.step_v_init, step_v_end=self.step_v_end
                       )

        self.fss.optimize()

        self.centroids = {}
        raw_centroids = self.fss.gbest.pos.reshape((self.n_clusters, self.n_attributes))

        for c in range(len(raw_centroids)):
            self.centroids[c] = raw_centroids[c]

        self.clusters = self.get_clusters(self.centroids, data)

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
