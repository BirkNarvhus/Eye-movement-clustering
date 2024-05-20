"""
This file contains the PlotUtil class which is used to reduce the dimension of the data and plot it
Used in the test_model.py file extensively
"""
import numpy as np
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE, Isomap, MDS
from sklearn.decomposition import KernelPCA, PCA

import sys
from pathlib import Path
path_root = Path(__file__).parents[2]
sys.path.append(str(path_root))


from models.finalImplementations.kMeansDownstream import KMeansDownstream


class PlotUtil:
    """
    Class to reduce the dimension of the data and plot it
    Can do a lot of different dim reductions and generate different plots depending on params
    """
    def __init__(self, data, title, mode="tsne", root="", show=False, dims=2, kmeans_after=False, kmeans_classes=3):
        """
        Reduces the dimension of the data and plots it
        can do kmeans after the dim reduction and plot the centers

        If kmeans_after is True, the centers will be plotted as well

        modes of dim reduction:
        - tsne
        - pca
        - isomap
        - mds
        - kernelPCA

        :param data: data to dim reduce and plot
        :param title: title of the plot
        :param mode: mode of the dim reduction
        :param root: root to save the plot
        :param show: show the plot
        :param dims: number of dims to reduce to
        :param kmeans_after: run kmeans after the dim reduction
        """
        self.data = data
        self.title = title
        self.tsne = None
        self.num_centers = 0
        self.colors = Colors()
        self.dims = dims
        if mode.upper() == "tsne".upper():
            self.mode = TSNE(n_components=dims, random_state=0)
        elif mode.upper() == "pca".upper():
            self.mode = PCA(n_components=40)
        elif mode.upper() == "isomap".upper():
            self.mode = Isomap(n_components=40)
        elif mode.upper() == "mds".upper():
            self.mode = MDS(n_components=40)
        else:
            self.mode = KernelPCA(n_components=None, kernel="rbf", gamma=10, alpha=0.1)
        self.mode_name = mode
        self.root = root
        self.show = show
        self.kmeans_after = kmeans_after
        self.kmeans = None
        self.kmeans_dims = kmeans_classes

    def plot_dim_reduced(self, with_centers=False, targets=None, targets_reg_alg=None):
        """
        Plot the data after dim reduction
        can do kmeans after the dim reduction and plot the centers
        if centers are given as input, the centers will be plotted as well
        targets are the labels of the data, will be plotted if provided
        If kmeans_after is True, then targets and centers will be decided by the kmeans algorithm

        Targets_reg_alg is the labels of the data from the IVVT algorithm, will be plotted if provided

        All plots differ in color based on label

        If Targets_reg_alg is provided, the scrip will run metrics on it to check distributions and accuracy
        Keep in mind accuracy is not a good metric for this kind of data, as the data is clustered not classified

        :param with_centers: plot the centers
        :param targets: the labels of the data
        :param targets_reg_alg: the labels of the data from the IVVT algorithm
        """
        print("Running - ", self.mode_name)
        self.data = self.data.numpy()
        print(self.data.shape)

        # dim reduction
        X_2d = self.mode.fit_transform(self.data)

        # If kmeans after is True, run kmeans and plot the centers
        if self.kmeans_after:
            print("Running kmeans after")
            self.kmeans = KMeansDownstream(self.kmeans_dims)
            self.kmeans.fit(X_2d)
            centers_ = self.kmeans.get_cluster_centers()
            targets = self.kmeans.get_labels()
            X_2d = np.concatenate((X_2d, centers_), axis=0)
            self.num_centers = len(centers_)
            with_centers = True

        # generate the base fig
        fig = plt.figure(figsize=(6, 5))
        ax = fig.add_subplot(111 if targets_reg_alg is None else 121, projection='3d' if self.dims == 3 else None)

        # retrieve the centers from the data (if centers are appended to data in test_model.py)
        # The centers should probably have their own data array, but this made it easier to implement
        if with_centers:
            centers = X_2d[-self.num_centers:]
            X_2d = X_2d[:-self.num_centers]

        # plot the targe/prediction data
        if targets is not None:
            for i in range(3):
                permute_indexes = (targets == i)
                if self.dims == 3:
                    ax.scatter(X_2d[permute_indexes, 0], X_2d[permute_indexes, 1], X_2d[permute_indexes, 2], c=["red", "blue", "green", "yellow", "black", "purple", "orange", "pink", "red", "red"][i])
                else:
                    ax.scatter(X_2d[permute_indexes, 0], X_2d[permute_indexes, 1], c=["red", "blue", "green", "yellow", "black", "purple", "orange", "pink", "red", "red"][i])
        else:
            print("plotting, points")
            if self.dims == 3:
                ax.scatter(X_2d[:, 0], X_2d[:, 1], X_2d[:, 2], c="red")
            else:
                ax.scatter(X_2d[:, 0], X_2d[:, 1], c="red")
        ax.set_title(self.mode_name)

        # plot the IVVT data if provided
        if targets_reg_alg is not None:
            print("plotting, IVVT")
            ax2 = fig.add_subplot(122, projection='3d' if self.dims == 3 else None)
            scatters = []
            for i in range(4):
                permute_indexes = (np.array(targets_reg_alg) == i)
                if self.dims == 3:
                    scatter = ax2.scatter(X_2d[permute_indexes, 0], X_2d[permute_indexes, 1], X_2d[permute_indexes, 2], c=["red", "blue", "green", "yellow", "black", "purple", "orange", "pink", "red", "red"][i])
                else:
                    scatter = ax2.scatter(X_2d[permute_indexes, 0], X_2d[permute_indexes, 1], c=["red", "blue", "green", "yellow", "black", "purple", "orange", "pink", "red", "red"][i])
                scatters.append(scatter)

            ax2.legend(scatters, ['Fixation', 'saccade', 'smooth persuite', 'No class'], loc="upper right", title="classifications")
            ax2.set_title("IVVT algorithm")

            # runs metrics
            if targets is not None:
                counts = [[], []]
                avg_pred = 0
                preds = []
                for i in range(3):
                    count = list(targets).count(i)
                    count2 = targets_reg_alg.count(i)
                    counts[0].append(count)
                    counts[1].append(count2)
                    pred = ((targets == i) == (np.array(targets_reg_alg) == i)).sum() / len(targets)
                    avg_pred += pred
                    preds.append(pred)

                print("avg Accuracy ", avg_pred/3)
                print("Accuracy per class ", preds)
                contributions = [[], []]
                for x, c in enumerate(counts):
                    for i in c:
                        contributions[x].append(i / sum(c))

                print("Contributions of classes")
                print("KMeans: ", contributions[0])
                print("IVVT: ", contributions[1])
        self.title = self.title.replace(" ", "_")

        # plot the centers if with_centers is True
        if with_centers:
            print("plotting, centers")
            for center in centers:
                if self.dims == 3:
                    ax.scatter(center[0], center[1], center[2], c=self.colors.__next__(), marker="x")
                else:
                    ax.scatter(center[0], center[1], c=self.colors.__next__(), marker="x")
        print("saving to ", self.root + "{}_{}.png".format(self.mode_name, self.title))
        plt.savefig(self.root + "{}_{}.png".format(self.mode_name, self.title))
        if self.show:
            plt.show()

    def plot_tsne_centers(self, centers):
        """
        Plot the data after dim reduction
        :param centers:  the centers to plot
        :return:  the plot
        """
        self.num_centers = len(centers)
        self.data = np.concatenate((self.data, centers), axis=0)
        self.plot_dim_reduced(with_centers=True)
        plt.title(self.title)

    def downstream(self, data):
        """
        predicts data with kmeans and dim reduction.
        used for later implementations
        :param data: the data to predict
        :return: the prediction
        """
        data = self.mode.transform(data)
        return self.kmeans.predict(data)


class Colors:
    """
    Helperclass to generate colors for the plots
    """
    def __init__(self):
        self.colors = ["red", "blue", "green", "yellow", "black", "purple", "orange", "pink"]

    def __next__(self):
        popped = self.colors.pop()
        if len(self.colors) == 0:
            self.colors = ["red", "blue", "green", "yellow", "black", "purple", "orange", "pink"]
        return popped
