import numpy as np
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE


class PlotUtil:
    def __init__(self, data, title):
        self.data = data
        self.title = title

    def plot_tsne(self):
        tsne = TSNE(n_components=2, random_state=0)
        X_2d = tsne.fit_transform(self.data)
        plt.figure(figsize=(6, 5))
        plt.scatter(X_2d[:, 0], X_2d[:, 1], c="red")
        plt.title(self.title)
        plt.savefig("tsne.png")

