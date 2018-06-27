import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from sklearn import datasets
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm
from clustering.metrics import Metrics
from clustering.kmeans import KMeans
from clustering.FCmeans import FCMeans
from clustering.PSOC import PSOC
from clustering.ABCC import ABCC
from clustering.FSSC import FSSC


def main():
    iris = datasets.load_iris()
    X = iris.data[:, :]
    species = iris.target

    # Normalize the dimension value to a float value with range 0 - 1
    std = MinMaxScaler()
    X = std.fit_transform(X)

    # techniques = ['k_means', 'FC_means', 'PSOC', 'ABCC', 'FSSC']
    techniques = ['ABCC']#, 'PSOC', 'FSSC', 'FSSC-2', 'FSSC-3']
    metrics = ['silhouete', 'calinskiHarabaszIndex']#, 'gap']
    num_exec = 30 #30

    for tec in techniques:
        rng = range(3, 4) #2-16

        mean = {}
        std = {}
        dff = {}
        met_eval = {}
        convergence = []
        for met in metrics: #initialize
            mean[met] = list()
            std[met] = list()
            dff[met] = list()
            met_eval[met] = list()

        for k in rng:
            for j in tqdm(range(num_exec), desc='{} - k: {}'.format(tec, k)):
                if tec == 'k_means':
                    clf = KMeans(k=k)
                elif tec == 'FC_means':
                    clf = FCMeans(k=k)
                elif tec == 'PSOC':
                    clf = PSOC(n_clusters=k, n_iter=100, swarm_size=50)
                elif tec == 'ABCC':
                    clf = ABCC(n_clusters=k, n_iter=100, swarm_size=50, trials_limit=10)
                elif tec == 'FSSC':
                    clf = FSSC(n_clusters=k, n_iter=100, swarm_size=50)
                elif tec == 'FSSC-2':
                    clf = FSSC(n_clusters=k, n_iter=100, swarm_size=50,
                               step_i_init=0.1, step_i_end=0.001, step_v_init=0.02, step_v_end=0.002)
                elif tec == 'FSSC-3':
                    clf = FSSC(n_clusters=k, n_iter=100, swarm_size=50, wheight_scale=10)

                clf.fit(data=X) #run technique
                if tec in ['ABCC', 'FSSC', 'PSOC']:
                    convergence.append(np.array(clf.convergence))

                for met in metrics:

                    out_dir = "results_Iris/booking/algorithm_{}/metric_{}/".format(tec, met) #create folder
                    if not os.path.exists(out_dir):
                        os.makedirs(out_dir)

                    file_name = out_dir + "{}_k_{}_exec_{}.csv".format('centroids', k, j)#save centroids
                    save_centroids = pd.DataFrame(clf.centroids)
                    save_centroids = save_centroids.transpose()
                    save_centroids.to_csv(file_name)

                    clusters = {}
                    for c in clf.centroids:
                        clusters[c] = []

                    for xi in range(len(X)):
                        dist = [np.linalg.norm(X[xi] - clf.centroids[c]) for c in clf.centroids]
                        class_ = dist.index(min(dist))
                        data = np.append(X[xi], species[xi])
                        clusters[class_].append(data)

                    clusters_file = open(out_dir + "{}_k_{}_exec_{}.csv".format('clusters', k, j), 'w')  # save clusters
                    clusters_file.write(str(len(clf.centroids)) + '\n')
                    for c in range(len(clf.centroids)):
                        clusters_file.write(str(len(clusters[c])) + '\n')
                        for xi in range(len(clusters[c])):
                            clusters_file.write(str(clusters[c][xi][0]))
                            for xij in range(1, len(clusters[c][xi])):
                                clusters_file.write(' ' + str(clusters[c][xi][xij]))
                            clusters_file.write('\n')
                    clusters_file.close()

                    # clusters_file = open(out_dir + "{}_k_{}_exec_{}.csv".format('clusters', k, j), 'w')#save clusters
                    # clusters_file.write(str(len(clf.centroids)) + '\n')
                    # for c in range(len(clf.centroids)):
                    #     clusters_file.write(str(len(clf.clusters[c])) + '\n')
                    #     for xi in range(len(clf.clusters[c])):
                    #         clusters_file.write(str(clf.clusters[c][xi][0]))
                    #         for xij in range(1, len(clf.clusters[c][xi])):
                    #             clusters_file.write(' ' + str(clf.clusters[c][xi][xij]))
                    #         clusters_file.write('\n')
                    # clusters_file.close()

                    if met == 'silhouete':
                        met_eval[met].append(Metrics.silhouette(clf.clusters, len(X)))
                    elif met == 'calinskiHarabaszIndex':
                        met_eval[met].append(Metrics.variance_based_ch(X, clf.centroids))
                    elif met == 'gap':
                        clusters = clf.clusters

                        random_data = np.random.uniform(0, 1, X.shape)
                        clf.fit(data=random_data)
                        random_clusters = clf.clusters

                        met_eval[met].append(Metrics.gap_statistic(clusters, random_clusters))

            for met in metrics:
                mean[met].append(np.mean(met_eval[met]))
                std[met].append(np.std(met_eval[met]))
                dff[met].append([tec, k, np.mean(met_eval[met]), np.std(met_eval[met])])
                met_eval[met] = list()

            # convergence
            if tec in ['ABCC', 'FSSC', 'PSOC']:
                out_dir = "results_Iris/booking/algorithm_{}/convergence/".format(tec)  # create folder
                if not os.path.exists(out_dir):
                    os.makedirs(out_dir)
                plt.figure()
                figure_name = out_dir + "convergence_k_{}.png".format(k)
                plt.title('{} - Convergence'.format(tec))
                plt.plot(np.mean(convergence, axis=0))
                plt.xlabel('Iterations')
                plt.ylabel('Fitness')
                plt.tight_layout()
                plt.savefig(figure_name)

                save_name = out_dir + "convergence_k_{}.csv".format(k)
                pd.DataFrame(convergence).to_csv(save_name)
                convergence = []

        for met in metrics:
            plt.figure()

            figure_name = "results_Iris/booking/algorithm_{}/metric_{}/plot.png".format(tec, met)
            plt.title('{} - Metric {}'.format(tec, met))
            plt.errorbar(rng, mean[met], yerr=std[met], marker='o', ecolor='b', capthick=2, barsabove=True)
            plt.xlabel('Clusters')
            plt.ylabel('Metric')
            plt.tight_layout()
            plt.savefig(figure_name)

            save_name = "results_Iris/booking/algorithm_{}/metric_{}/output.csv".format(tec, met)
            dff[met] = pd.DataFrame(dff[met])
            dff[met].columns = ['ALGORITHM', 'CLUSTERS', 'MEAN', 'STD']
            dff[met].to_csv(save_name)

if __name__ == '__main__':
    main()
