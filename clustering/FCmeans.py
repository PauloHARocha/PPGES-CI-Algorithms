import numpy as np


class FCMeans(object):
    def __init__(self, k=3, n_iter=300, fuzzy_c=2, tolerance=0.001, data=None):
        self.k = k
        self.n_iter = n_iter
        self.fuzzy_c = fuzzy_c
        self.tolerance = tolerance
        self.data = data
        self.centroids = {}
        self.clusters = {}
        self.degree_of_membership = []

    def init_centroids(self):
        self.centroids = {}
        for k in range(self.k):
            self.centroids[k] = np.random.random(self.data.shape[1])

        self.degree_of_membership = np.zeros((self.data.shape[0], self.k))
        for idx_ in self.centroids:
            for idx, xi in enumerate(self.data):
                norm = np.linalg.norm(xi - self.centroids[idx_])
                all_norms = [norm / np.linalg.norm(xi - self.centroids[c]) for c in self.centroids]
                all_norms = np.power(all_norms, 2 / (self.fuzzy_c - 1))
                updated_degree_of_membership = 1 / sum(all_norms)
                self.degree_of_membership[idx][idx_] = updated_degree_of_membership

    def fit(self, data=None, n_clusters=None):

        if data is not None:
            self.data = data
        if n_clusters is not None:
            self.k = n_clusters

        self.init_centroids()

        for iteration in range(self.n_iter):
            powers = np.power(self.degree_of_membership, self.fuzzy_c)
            for idx_ in self.centroids:
                centroid = []
                sum_membeship = 0
                for idx, xi in enumerate(self.data):
                    centroid.append(powers[idx][idx_] * np.array(xi))
                    sum_membeship += powers[idx][idx_]
                centroid = np.sum(centroid, axis=0)
                centroid = centroid / sum_membeship
                self.centroids[idx_] = centroid

            max_episilon = 0.0
            for idx_ in self.centroids:
                for idx, xi in enumerate(self.data):
                    norm = np.linalg.norm(xi - self.centroids[idx_])
                    all_norms = [norm / np.linalg.norm(xi - self.centroids[c]) for c in self.centroids]
                    all_norms = np.power(all_norms, 2 / (self.fuzzy_c - 1))
                    updated_degree_of_membership = 1 / sum(all_norms)
                    diff = updated_degree_of_membership - self.degree_of_membership[idx][idx_]
                    self.degree_of_membership[idx][idx_] = updated_degree_of_membership

                    if diff > max_episilon:
                        max_episilon = diff
            if max_episilon <= self.tolerance:
                break
        self.get_clusters()

    def predict(self, x):
        if len(x.shape) > 1:
            class_ = []
            for c in self.centroids:
                class_.append(np.sum((x - self.centroids[c]) ** 2, axis=1))
            return np.argmin(np.array(class_).T, axis=1)
        else:
            dist = [np.linalg.norm(x - self.centroids[c]) for c in self.centroids]
            class_ = dist.index(min(dist))
            return class_

    def get_clusters(self):
        _class = np.argmax(self.degree_of_membership, axis=1)

        self.clusters = {}  # initialize the clusters
        for k in range(self.k):
            self.clusters[k] = []  # For each k, create an array

        for data, cluster in zip(self.data, _class):
            self.clusters[cluster].append(data)
