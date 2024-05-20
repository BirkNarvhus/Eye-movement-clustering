import sys

import numpy as np
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE, Isomap, MDS
from sklearn.decomposition import KernelPCA, PCA

sys.path.append('C:\\Users\\birkn\Documents\\bachlor\\eye-movement-classification')


from models.finalImplementations.kMeansDownstream import KMeansDownstream


class PlotUtil:
    def __init__(self, data, title, mode="tsne", root="", show=False, dims=2, kmeans_after=False):
        """
        Reduces the dimension of the data and plots it
        can do kmeans after the dim reduction and plot the centers

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

    def plot_tsne(self, with_centers=False, targets=None, targets_reg_alg=None):
        print("Running - ", self.mode_name)
        self.data = self.data.numpy()
        print(self.data.shape)
        X_2d = self.mode.fit_transform(self.data)
        if self.kmeans_after:
            print("Running kmeans after")
            self.kmeans = KMeansDownstream(3)
            self.kmeans.fit(X_2d)
            centers_ = self.kmeans.get_cluster_centers()
            targets = self.kmeans.get_labels()
            X_2d = np.concatenate((X_2d, centers_), axis=0)
            self.num_centers = len(centers_)
            with_centers = True

        fig = plt.figure(figsize=(6, 5))
        ax = fig.add_subplot(111 if targets_reg_alg is None else 121, projection='3d' if self.dims == 3 else None)

        if with_centers:
            centers = X_2d[-self.num_centers:]
            X_2d = X_2d[:-self.num_centers]

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
        self.num_centers = len(centers)
        self.data = np.concatenate((self.data, centers), axis=0)
        self.plot_tsne(with_centers=True)
        plt.title(self.title)

    def downstream(self, data):
        data = self.mode.transform(data)
        return self.kmeans.predict(data)



class Colors:
    def __init__(self):
        self.colors = ["red", "blue", "green", "yellow", "black", "purple", "orange", "pink"]

    def __next__(self):
        popped = self.colors.pop()
        if len(self.colors) == 0:
            self.colors = ["red", "blue", "green", "yellow", "black", "purple", "orange", "pink"]
        return popped
