import numpy as np
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE


class PlotUtil:
    def __init__(self, data, targets, labels, title):
        self.data = data
        self.target = targets.squeeze()
        self.labels = labels
        self.title = title

    def plot_tsne(self):
        tsne = TSNE(n_components=2, random_state=0)
        X_2d = tsne.fit_transform(self.data)
        target_ids = range(len(self.labels))
        plt.figure(figsize=(6, 5))
        colors = 'r', 'g', 'b', 'c', 'm', 'y', 'k', 'orange', 'purple', 'brown'
        print(self.target.shape)
        for i, c, label in zip(target_ids, colors, self.labels):
            plt.scatter(X_2d[self.target == i, 0], X_2d[self.target == i, 1], c=c, label=label)
        plt.title(self.title)
        plt.legend()
        plt.show()