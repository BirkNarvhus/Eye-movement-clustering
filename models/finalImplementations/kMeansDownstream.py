"""
Classes for downstream kmeans
"""

from pathlib import Path

import numpy as np
from sklearn.cluster import KMeans

import sys
path_root = Path(__file__).parents[2]
sys.path.append(str(path_root))

Out_folder = 'content/saved_outputs/'
Output_file = Out_folder + 'kmeans_output.npy'
input_file = 'content/saved_outputs/model_1-feats.npy'
do_plot = True


class KMeansDownstream:
    """
    This class contains the downstream kmeans functionality

    """
    def __init__(self, num_clusters):
        """
        Creates the kmeans method on init
        :param num_clusters: int Number of clusters for kmeans
        """
        self.num_clusters = num_clusters
        self.kmeans = KMeans(n_clusters=num_clusters, random_state=0)

    def fit(self, data):
        """
        Trains the k-means model
        :param data: data to fit
        """
        self.kmeans = self.kmeans.fit(data)

    def predict(self, data):
        """
        get predictions from the model
        :param data: data to predict
        :return: indexes of each cluster data belongs to
        """
        return self.kmeans.predict(data)

    def save(self, path):
        """
        save cluster centers
        :param path: path to save
        """
        with open(path, 'wb') as f:
            np.save(f, self.kmeans.cluster_centers_)

    def load(self, path):
        """
        load cluster centers
        :param path: path of load file
        """
        with open(path, 'rb') as f:
            self.kmeans.cluster_centers_ = np.load(f)

    def get_cluster_centers(self):
        """
        :return: Cluster centers
        """
        return self.kmeans.cluster_centers_

    def get_labels(self):
        """
        get labels of fitted data
        :return: fitted labels
        """
        return self.kmeans.labels_


def test():
    """
    testing downstream kmeans
    """
    data = np.load(input_file)
    num_clusters = 4
    kmeans = KMeansDownstream(num_clusters)
    kmeans.fit(data)
    kmeans.save(Output_file)
    print("KMeans Downstream finished and saved to ", Output_file)


if __name__ == '__main__':
    test()

