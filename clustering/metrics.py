import numpy as np
from scipy.spatial import distance
from operator import truediv
from numpy import inf
from sklearn.metrics import silhouette_score
import math


class Metrics:


    @staticmethod
    def gap_statistic(clusters, random_clusters):
        Wk = Metrics.compute_Wk(clusters)
        En_Wk = Metrics.compute_Wk(random_clusters)
        return np.log(En_Wk) - np.log(Wk)

    @staticmethod
    def compute_Wk(clusters):
        # Funcao do Gap Statistic
        wk_ = 0.0
        for r in range(len(clusters)):
            nr = len(clusters[r])
            if nr > 1:
                dr = distance.pdist(clusters[r], metric='sqeuclidean')
                dr = distance.squareform(dr)
                dr = sum(np.array(dr, dtype=np.float64).ravel())
                wk_ += (1.0 / (2.0 * nr)) * dr
        return wk_

    @staticmethod
    def silhouette(clusters, len_data):
        sil = 0.0
        for k in range(len(clusters)):
            for d_out in range(len(clusters[k])):
                ai = Metrics.silhouette_a(clusters[k][d_out], k, clusters)
                bi = Metrics.minimum_data_data(clusters[k][d_out], k, clusters)
                max_a_b = bi
                if ai > bi:
                    max_a_b = ai
                sil += truediv(bi - ai, max_a_b)
        return truediv(sil, len_data)

    @staticmethod
    def silhouette_a(datum, cluster_in, clusters):
        sum_d = 0.0
        for d_out in range(len(clusters[cluster_in])):
            sum_d += distance.euclidean(datum, clusters[cluster_in][d_out])
        sum_d = truediv(sum_d, len(clusters[cluster_in]))
        return sum_d

    @staticmethod
    def minimum_data_data(datum, cluster_in, clusters):
        min_D = float('inf')
        for k in range(len(clusters)):
            if cluster_in != k:
                x = 0.0
                for d_out in range(len(clusters[k])):
                    x += distance.euclidean(datum, clusters[k][d_out])
                x = truediv(x, len(clusters[k]))
                if min_D > x:
                    min_D = x
        return min_D