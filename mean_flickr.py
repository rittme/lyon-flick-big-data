import numpy as np
from sklearn.cluster import MeanShift, estimate_bandwidth
import csv

quantiles = (0.2, 0.1, 0.05, 0.01)

X = []
ID = []
with open('output.csv', 'rb') as csvfile:
    flickreader = csv.reader(csvfile, delimiter=',', quotechar='"')
    for row in flickreader:
        ID.append(row[0])
        X.append((np.float64(row[1]), np.float64(row[2])))

X = np.array(X)

iteration = 1
for quant in quantiles:
    bandwidth = estimate_bandwidth(X, quantile=quant, n_samples=20000)

    ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
    ms.fit(X)
    labels = ms.labels_
    cluster_centers = ms.cluster_centers_

    labels_unique = np.unique(labels)
    n_clusters_ = len(labels_unique)

    print("Iteration " + str(iteration) +
          " number of estimated clusters : %d" % n_clusters_)

    ###########################################################################
    # Export result

    f = open('mean_q' + str(quant) + '_' + 'Clusters.csv', 'w')

    for k in range(n_clusters_):
        my_members = labels == k
        cluster_center = cluster_centers[k]
        f.write('id,cluster\n')
        for i in range(len(my_members)):
            if my_members[i]:
                f.write(ID[i] + ", " +
                        # X[i][0].astype('str') + ", " +
                        # X[i][1].astype('str') + ", " +
                        str(k) + '\n')

    f.close()

    ###########################################################################
    # Plot result
    '''
    import matplotlib.pyplot as plt
    from itertools import cycle

    plt.figure(iteration)


    colors = cycle('bgrcmykbgrcmykbgrcmykbgrcmyk')
    for k, col in zip(range(n_clusters_), colors):
        my_members = labels == k
        cluster_center = cluster_centers[k]
        plt.plot(X[my_members, 0], X[my_members, 1], col + '.')
        # plt.plot(cluster_center[0], cluster_center[1], 'o',
        #          markerfacecolor=col, markeredgecolor='k', markersize=14)
    plt.title('Estimated number of clusters: %d' % n_clusters_)
    plt.show()
    '''
    iteration += 1
