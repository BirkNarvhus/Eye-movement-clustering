import torch
from sklearn.manifold import TSNE
import numpy as np
import matplotlib.pyplot as plt
from util.data import data_generator

def model(x):
    return torch.randn((x.shape[0], 128))

def main():
    num_classes = 10
    batch_size = 128

    feats = np.array([]).reshape((0, 128))
    targets = np.array([]).reshape((0, 1))

    train_loader, _ = data_generator("../data/mnist", batch_size=batch_size, clr=True)

    for idx, (x1, x2, target) in enumerate(train_loader):
        if idx > 5:
            break
        x1 = x1.to(device='cpu', dtype=torch.float)
        out = model(x1)
        if len(out) != batch_size:
            break
        out = out.cpu().data.numpy()
        feats = np.append(feats, out, axis=0)
        target = target.cpu().data.numpy().reshape((-1, 1))
        targets = np.append(targets, target, axis=0)

    tsne = TSNE(n_components=3, perplexity=50)
    x_feats = tsne.fit_transform(feats)
    print(x_feats.shape)

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    num_samples = int(batch_size * (targets.shape[0] // batch_size))  # (len(val_df)
    for i in range(num_classes):

        test = np.array([targets[:num_samples] == i]).squeeze()
        ax.scatter(x_feats[test, 2], x_feats[test, 1], x_feats[test, 0])

    ax.legend([str(i) for i in range(num_classes)])
    plt.show()


if __name__ == "__main__":
    main()
