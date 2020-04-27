
import numpy as np
class Cluster:
    def __init__(self):
        self.parent = None
        self.cluster_member_index = []
        self.SSE = 0

    def init_centroid(self,centroid):
        self.centroid = centroid

    def cal_centroid(self,x):
        selected = np.zeros((len(self.cluster_member_index),x.shape[1]))
        for i in range(len(self.cluster_member_index)):
            selected[i,:] = x[self.cluster_member_index[i],:]
        self.member_points = selected
        self.centroid = np.average(selected,axis=0)

    def cal_SSE(self):
        # dist = np.linalg.norm((self.centroid-self.member_points),axis=1)
        # self.SSE = np.sum(self.SSE)
        self.SSE = np.sum(np.square(self.centroid-self.member_points))
        return self.SSE