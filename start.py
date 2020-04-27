from sklearn.preprocessing import MinMaxScaler

from HAC import HACClustering
from Kmeans import KMEANSClustering
from arff import Arff
from sklearn.model_selection import KFold
import numpy as np
from sklearn.model_selection import train_test_split


if __name__ == "__main__":
    # mat = Arff("abalone.arff", label_count=0)  ## label_count = 0 because clustering is unsupervised.
    # mat = Arff("seismic-bumps_train.arff", label_count=0)
    mat = Arff("iris.arff", label_count=0)
    raw_data = mat.data
    data = raw_data

    ### not include last column
    # data = data[:, 0:-1]

    ### Normalize the data ###
    scaler = MinMaxScaler()
    scaler.fit(data)
    norm_data = scaler.transform(data)

    # ### KMEANS ###
    # KMEANS = KMEANSClustering(k=4, debug=False)
    # KMEANS.fit(norm_data)
    # KMEANS.save_clusters("my_kmeans.txt")
    # KMEANS.printSSE()
    # # KMEANS.save_clusters("evaluation_kmeans.txt")
    #
    # ### HAC SINGLE LINK ###
    HAC_single = HACClustering(k=5, link_type='single')
    HAC_single.fit(norm_data)
    HAC_single.save_clusters("my_hac_single.txt")
    # # HAC_single.save_clusters("evaluation_hac_single.txt")
    #
    # ### HAC Complete LINK ###
    # HAC_complete = HACClustering(k=5, link_type='complete')
    # HAC_complete.fit(norm_data)
    # HAC_complete.save_clusters("my_hac_complete.txt")
    # HAC_complete.save_clusters("evaluation_hac_complete.txt")


