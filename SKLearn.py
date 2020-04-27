from sklearn import metrics
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import davies_bouldin_score
from sklearn.preprocessing import MinMaxScaler

from HAC import HACClustering
from Kmeans import KMEANSClustering
from arff import Arff
from sklearn.model_selection import KFold
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from cluster import Cluster


def calSSE(label,norm_data,k):
    all_cluster = []
    totalSSE = 0
    for i in range(k):
        one_cluster = Cluster()
        all_cluster.append(one_cluster)
    for i in range(norm_data.shape[0]):
        ind = label[i]
        all_cluster[ind].cluster_member_index.append(i)
    for i in range(k):
        all_cluster[i].cal_centroid(norm_data)
        totalSSE = totalSSE + all_cluster[i].cal_SSE()
    return totalSSE

def getDistance(a, b):
        dist = np.linalg.norm((a - b), axis=1)
        return dist.reshape(1, -1)

def getMeanDist(a, cluster, norm_data):
    b = norm_data[cluster.cluster_member_index,:]
    sum = np.sum(getDistance(a,b))
    return sum/b.shape[0]

def mySilhouette(norm_data, labels, k):
    all_cluster = []
    for i in range(k):
        one_cluster = Cluster()
        all_cluster.append(one_cluster)
    for i in range(norm_data.shape[0]):
        ind = labels[i]
        all_cluster[ind].cluster_member_index.append(i)
    s = 0
    for i in range(norm_data.shape[0]):
        own_cluster = Cluster()
        for j in range(len(all_cluster[labels[i]].cluster_member_index)):
            if all_cluster[labels[i]].cluster_member_index[j] != i:
                own_cluster.cluster_member_index.append(all_cluster[labels[i]].cluster_member_index[j])
        a = getMeanDist(norm_data[i,:],own_cluster,norm_data)
        dist = []
        for j in range(len(all_cluster)):
            if j != labels[i]:
                dist.append( getMeanDist(norm_data[i,:],all_cluster[j],norm_data))
            else:
                dist.append(np.inf)
        b = min(dist)
        s = s + (b-a)/max(a,b)
    return s/norm_data.shape[0]

if __name__ == "__main__":
    # mat = Arff("seismic-bumps_train.arff", label_count=0)
    # mat = Arff("iris.arff", label_count=0)
    mat = Arff("lenses.arff", label_count=0)
    raw_data = mat.data
    data = raw_data

    ### Normalize the data ###
    scaler = MinMaxScaler()
    scaler.fit(data)
    norm_data = scaler.transform(data)

    k = 7

    ## SK learn K mean
    kmeans = KMeans(n_clusters=k, random_state=0).fit(norm_data)

    # print("SSE for k mean")
    # print(kmeans.inertia_)
    s_score = metrics.silhouette_score(norm_data, kmeans.labels_, metric='euclidean')
    print("silhouette_score")
    print(s_score)
    my_s_score = mySilhouette(norm_data,kmeans.labels_, k)
    print(my_s_score)
    # d_score = davies_bouldin_score(norm_data, kmeans.labels_)
    # print("davies_bouldin_score")
    # print(d_score)

    print("-----------------")
    ## HAV from SK learn
    clustering = AgglomerativeClustering(n_clusters=k,linkage='complete').fit(norm_data)
    # print(calSSE(clustering.labels_,norm_data,k))
    s_score2 = metrics.silhouette_score(norm_data, clustering.labels_, metric='euclidean')
    print("silhouette_score")
    print(s_score2)
    # d_score2 = davies_bouldin_score(norm_data, clustering.labels_)
    # print("davies_bouldin_score")
    # print(d_score2)
