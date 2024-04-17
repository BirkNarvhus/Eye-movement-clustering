import numpy as np
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA


class PlotUtil:
    def __init__(self, data, title, mode="tsne", root="", show=False, dims=2):
        self.data = data
        self.title = title
        self.tsne = None
        self.num_centers = 0
        self.colors = Colors()
        self.dims = dims
        self.mode = TSNE(n_components=dims, random_state=0) if mode.upper() == "tsne".upper() else PCA(n_components=dims)
        self.mode_name = mode
        self.root = root
        self.show = show

    def plot_tsne(self, with_centers=False):
        X_2d = self.mode.fit_transform(self.data)
        fig = plt.figure(figsize=(6, 5))
        ax = fig.add_subplot(projection='3d' if self.dims == 3 else None)
        if with_centers:
            centers = X_2d[-self.num_centers:]
            X_2d = X_2d[:-self.num_centers]
        if self.dims == 3:
            ax.scatter(X_2d[:, 0], X_2d[:, 1], X_2d[:, 2], c="red")
        else:
            ax.scatter(X_2d[:, 0], X_2d[:, 1], c="red")
        plt.title(self.title)

        if with_centers:
            for center in centers:
                if self.dims == 3:
                    ax.scatter(center[0], center[1], center[2], c=self.colors.__next__(), marker="x")
                else:
                    ax.scatter(center[0], center[1], c=self.colors.__next__(), marker="x")
        plt.savefig(self.root + "{}_{}.png".format(self.mode_name, self.title))
        if self.show:
            plt.show()

    def plot_tsne_centers(self, centers):
        self.num_centers = len(centers)
        np.concatenate((self.data, centers), axis=0)
        self.plot_tsne(with_centers=True)
        plt.title(self.title)


class Colors:
    def __init__(self):
        self.colors = ["red", "blue", "green", "yellow", "black", "purple", "orange", "pink"]

    def __next__(self):
        popped = self.colors.pop()
        if len(self.colors) == 0:
            self.colors = ["red", "blue", "green", "yellow", "black", "purple", "orange", "pink"]
        return popped
