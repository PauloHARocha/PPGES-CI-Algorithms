import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm
from clustering.metrics import Metrics
from clustering.kmeans import KMeans
from clustering.FCmeans import FCMeans


def main():
    # Importing dataset
    dataset = pd.read_csv("500_Cities_CDC.csv")

    # Select lines and columns
    X = dataset.iloc[:, [4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 64, 68, 72, 76, 80, 84, 88, 92, 96,
                         100, 104, 108, 112]].values

    # Normalize the dimension value to a float value with range 0 - 1
    std = MinMaxScaler()
    X = std.fit_transform(X)

    techniques = ['k_means', 'FC_means'] #PSOC, ABCC
    metrics = ['gap', 'silhouete', 'calinskiHarabaszIndex']
    num_exec = 2

    for tec in techniques:
        rng = range(2, 4)
        met_eval = []
        for met in metrics:
            mean = []
            std = []
            dff = []
            for k in rng:
                for j in tqdm(range(num_exec), desc='{} - {} - k: {}'.format(tec, met, k)):
                    if tec == 'k_means':
                        clf = KMeans(k=k)
                    elif tec == 'FC_means':
                        clf = FCMeans(k=k)

                    clf.fit(data=X) #run technique

                    out_dir = "results/booking/algorithm_{}/metric_{}/".format(tec, met) #create folder
                    if not os.path.exists(out_dir):
                        os.makedirs(out_dir)

                    file_name = out_dir + "{}_k_{}_exec_{}.csv".format('centroids', k, j)#save centroids
                    save_centroids = pd.DataFrame(clf.centroids)
                    save_centroids = save_centroids.transpose()
                    save_centroids.to_csv(file_name)

                    clusters_file = open(out_dir + "{}_k_{}_exec_{}.csv".format('clusters', k, j), 'w')#save clusters
                    clusters_file.write(str(len(clf.centroids)) + '\n')
                    for c in range(len(clf.centroids)):
                        clusters_file.write(str(len(clf.clusters[c])) + '\n')
                        for xi in range(len(clf.clusters[c])):
                            clusters_file.write(str(clf.clusters[c][xi][0]))
                            for xij in range(1, len(clf.clusters[c][xi])):
                                clusters_file.write(' ' + str(clf.clusters[c][xi][xij]))
                            clusters_file.write('\n')
                    clusters_file.close()

                    if met == 'gap':
                        clusters = clf.clusters

                        random_data = np.random.uniform(0, 1, X.shape)
                        clf.fit(data=random_data)
                        random_clusters = clf.clusters

                        met_eval.append(Metrics.gap_statistic(clusters, random_clusters))
                    elif met == 'silhouete':
                        met_eval.append(Metrics.silhouette(clf.clusters, len(X)))
                    elif met == 'calinskiHarabaszIndex':
                        met_eval.append(Metrics.variance_based_ch(X, clf.centroids))

                mean.append(np.mean(met_eval))
                std.append(np.std(met_eval))
                dff.append([tec, k, np.mean(met_eval), np.std(met_eval)])
                met_eval = []

                plt.figure()

            figure_name = "results/booking/algorithm_{}/metric_{}/plot.png".format(tec, met)
            plt.title('{} - Metric {}'.format(tec, met))
            plt.errorbar(rng, mean, yerr=std, marker='o', ecolor='b', capthick=2, barsabove=True)
            plt.xlabel('Clusters')
            plt.ylabel('Metric')
            plt.tight_layout()
            plt.savefig(figure_name)

            save_name = "results/booking/algorithm_{}/metric_{}/output.csv".format(tec, met)
            dff = pd.DataFrame(dff)
            dff.columns = ['ALGORITHM', 'CLUSTERS', 'MEAN', 'STD']
            dff.to_csv(save_name)

if __name__ == '__main__':
    main()
