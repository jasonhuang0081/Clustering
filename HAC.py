import numpy as np
from sklearn.base import BaseEstimator, ClusterMixin

from cluster import Cluster


class HACClustering(BaseEstimator,ClusterMixin):

    def __init__(self,k=3,link_type='single'): ## add parameters here
        """
        Args:
            k = how many final clusters to have
            link_type = single or complete. when combining two clusters use complete link or single link
        """
        self.link_type = link_type
        self.k = k

    def getDistance(self,a):
        dist = np.linalg.norm((a-self.all_data),axis=1)
        return dist.reshape(1,-1)

    def mergeCluster(self,a,b):
        one_cluster = Cluster()
        id_a = a.cluster_member_index[0]
        id_b = b.cluster_member_index[0]
        while a.parent is not None:
            a = a.parent
        while b.parent is not None:
            b = b.parent
        if id_a not in b.cluster_member_index and id_b not in a.cluster_member_index:
            one_cluster.cluster_member_index.extend(b.cluster_member_index)
            one_cluster.cluster_member_index.extend(a.cluster_member_index)
            a.parent = one_cluster
            b.parent = one_cluster
            self.totalCluster = self.totalCluster - 1
        return one_cluster

    def fit(self,X,y=None):
        """ Fit the data; In this lab this will make the K clusters :D
        Args:
            X (array-like): A 2D numpy array with the training data
            y (array-like): An optional argument. Clustering is usually unsupervised so you don't need labels
        Returns:
            self: this allows this to be chained, e.g. model.fit(X,y).predict(X_test)
        """
        self.all_data = X
        self.matrix = np.zeros((X.shape[0],X.shape[0]))
        self.all_cluster = []

        for i in range(X.shape[0]):
            self.matrix[i,:] = self.getDistance(X[i,:].reshape(1,-1))
            one_cluster = Cluster()
            one_cluster.cluster_member_index.append(i)
            self.all_cluster.append(one_cluster)

        il = np.tril_indices(X.shape[0])
        self.totalCluster = len(self.all_cluster)
        if self.link_type == 'single':
            self.matrix[il] = np.inf
            while self.totalCluster > self.k:
                row, col = np.unravel_index(self.matrix.argmin(), self.matrix.shape)
                self.mergeCluster(self.all_cluster[row],self.all_cluster[col])
                self.matrix[row,col] = np.inf
            self.cluster_head = self.pickClusterHead()


        if self.link_type == 'complete':
            # self.matrix[il] = np.inf
            while len(self.all_cluster) > self.k:
                matrix = self.updateDistMatrix()
                row, col = np.unravel_index(matrix.argmin(), matrix.shape)
                self.all_cluster[row].cluster_member_index.extend(self.all_cluster[col].cluster_member_index)
                del self.all_cluster[col]
            self.cluster_head = self.all_cluster
            self.totalCluster = len(self.all_cluster)

        sum = 0
        for i in range(len(self.cluster_head)):
            self.cluster_head[i].cal_centroid(self.all_data)
            sum = sum + self.cluster_head[i].cal_SSE()
        self.totalSSE = sum
        return self
                # cluster_head = self.pickClusterHead()
                # for i in range(len(cluster_head)):
                #     if len(cluster_head[i].cluster_member_index) > 1:
                #         for j in range(i + 1, len(cluster_head)):
                #             self.updateLongDist(cluster_head[i].cluster_member_index,
                #                                 cluster_head[j].cluster_member_index)
                # row, col = np.unravel_index(self.matrix.argmin(), self.matrix.shape)
                # parent_node = self.mergeCluster(self.all_cluster[row],self.all_cluster[col])
                # for i in range(len(parent_node.cluster_member_index)):
                #     for j in range(i + 1, len(parent_node.cluster_member_index)):
                #         self.matrix[i,j] = np.inf
                # self.matrix[row,col] = np.inf

    def updateDistMatrix(self):
        matrix = np.ones((len(self.all_cluster),len(self.all_cluster)))*np.inf
        for i in range(len(self.all_cluster)):
            for j in range(i + 1, len(self.all_cluster)):
                matrix[i,j] = self.getLongDist(self.all_cluster[i],self.all_cluster[j])
        return matrix

    def getLongDist(self, cluster1, cluster2):
        current_max = 0
        for i in cluster1.cluster_member_index:
            for j in cluster2.cluster_member_index:
                current = self.matrix[i,j]
                if current_max < current:
                    current_max = current
        return current_max
    # def updateLongDist(self, cluster1, cluster2):
    #     current_max = 0
    #     for i in cluster1:
    #         for j in cluster2:
    #             current = self.matrix[i][j]
    #             if current != np.inf and current > current_max:
    #                 current_max = current
    #     for i in cluster1:
    #         for j in cluster2:
    #             if j > i:
    #                 self.matrix[i][j] = current_max

    def pickClusterHead(self):
        cluster_head = []
        exist_node = []
        for i in range(self.all_data.shape[0]):
            if i not in exist_node:
                current = self.all_cluster[i]
                while current.parent is not None:
                    current = current.parent
                cluster_head.append(current)
                exist_node.extend(current.cluster_member_index)
        return cluster_head

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
        f.write("{:d}\n".format(self.totalCluster))
        f.write("{:.4f}\n\n".format(self.totalSSE))
        for i in range(len(self.cluster_head)):
            f.write(np.array2string(self.cluster_head[i].centroid,precision=4,separator=","))
            f.write("\n")
            f.write("{:d}\n".format(len(self.cluster_head[i].cluster_member_index)))
            f.write("{:.4f}\n\n".format(self.cluster_head[i].SSE))
        f.close()

    def printSSE(self):
        print(self.totalSSE)

