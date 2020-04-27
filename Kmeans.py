import numpy as np
from sklearn.base import BaseEstimator, ClusterMixin

from cluster import Cluster


class KMEANSClustering(BaseEstimator,ClusterMixin):

    def __init__(self,k=3,debug=False): ## add parameters here
        """
        Args:
            k = how many final clusters to have
            debug = if debug is true use the first k instances as the initial centroids otherwise choose random points as the initial centroids.
        """
        self.k = k
        self.debug = debug

    def fit(self,X,y=None):
        """ Fit the data; In this lab this will make the K clusters :D
        Args:
            X (array-like): A 2D numpy array with the training data
            y (array-like): An optional argument. Clustering is usually unsupervised so you don't need labels
        Returns:
            self: this allows this to be chained, e.g. model.fit(X,y).predict(X_test)
        """
        self.all_data = X
        if self.debug == True:
            self.centroid = np.copy(self.all_data[0:self.k,:])
        else:
            rand = np.random.permutation(X.shape[0])
            rand_x = X[rand]
            self.centroid = np.copy(rand_x[0:self.k,:])
        self.all_cluster = []
        for i in range(self.k):
            one_cluster = Cluster()
            one_cluster.init_centroid(self.centroid[i,:])
            self.all_cluster.append(one_cluster)
        self.prev_centroid = np.zeros((1,self.k))
        while not np.array_equal(self.prev_centroid, self.centroid):
            for i in range(len(self.all_cluster)):
                self.all_cluster[i].cluster_member_index = []
            for i in range(self.all_data.shape[0]):
                a = self.getDistance(self.all_data[i, :], self.centroid)
                row, col = np.unravel_index(np.argmin(a, axis=None), a.shape)
                self.all_cluster[col].cluster_member_index.append(i)
            # print(self.centroid)
            # print("__________")
            for i in range(len(self.all_cluster)):
                self.all_cluster[i].cal_centroid(self.all_data)
                # print(self.all_cluster[i].cluster_member_index)
            # print("===============")
            self.prev_centroid = np.copy(self.centroid)
            for i in range(self.centroid.shape[0]):
                self.centroid[i, :] = self.all_cluster[i].centroid

        sum = 0
        for i in range(len(self.all_cluster)):
            sum = sum + self.all_cluster[i].cal_SSE()
        self.totalSSE = sum
        return self

    def getDistance(self, a, centroid):
        dist = np.linalg.norm((a - centroid), axis=1)
        return dist.reshape(1, -1)
    def save_clusters(self,filename):
        """
            f = open(filename,"w+") 
            Used for grading.
            write("{:d}\n".format(k))
            write("{:.4f}\n\n".format(total SSE))
            for each cluster and centroid:
                write(np.array2string(centroid,precision=4,separator=","))
                write("\n")
                write("{:d}\n".format(size of cluster))
                write("{:.4f}\n\n".format(SSE of cluster))
            f.close()
        """

        f = open(filename, "w+")
        f.write("{:d}\n".format(self.k))
        f.write("{:.4f}\n\n".format(self.totalSSE))
        for i in range(len(self.all_cluster)):
            f.write(np.array2string(self.all_cluster[i].centroid, precision=4, separator=","))
            f.write("\n")
            f.write("{:d}\n".format(len(self.all_cluster[i].cluster_member_index)))
            f.write("{:.4f}\n\n".format(self.all_cluster[i].SSE))
        f.close()
    def printSSE(self):
        print(self.totalSSE)