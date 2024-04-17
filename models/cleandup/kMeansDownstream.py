
import numpy as np
from sklearn.cluster import KMeans
import sys
sys.path.append('C:\\Users\\vizlab_stud\\Documents\\pythonProjects\\eye-movement-classification')
print(sys.path)
from util.plot_tsne import PlotUtil

Out_folder = 'content/saved_outputs/'
Output_file = Out_folder + 'kmeans_output.npy'
input_file = 'content/saved_outputs/model_1-feats.npy'
do_plot = True

class KMeansDownstream:
    def __init__(self, num_clusters):
        self.num_clusters = num_clusters
        self.kmeans = KMeans(n_clusters=num_clusters, random_state=0)

    def fit(self, data):
        self.kmeans = self.kmeans.fit(data)

    def predict(self, data):
        return self.kmeans.predict(data)

    def save(self, path):
        with open(path, 'wb') as f:
            np.save(f, self.kmeans.cluster_centers_)

    def load(self, path):
        with open(path, 'rb') as f:
            self.kmeans.cluster_centers_ = np.load(f)

    def get_cluster_centers(self):
        return self.kmeans.cluster_centers_

    def get_labels(self):
        return self.kmeans.labels_


def main():
    data = np.load(input_file)
    num_clusters = 4
    kmeans = KMeansDownstream(num_clusters)
    kmeans.fit(data)
    kmeans.save(Output_file)
    print("KMeans Downstream finished and saved to ", Output_file)
    if do_plot:
        PlotUtil(data, "KMeans Downstream", mode="PCA", root=Out_folder).plot_tsne_centers(kmeans.get_cluster_centers())


if __name__ == '__main__':
    main()

