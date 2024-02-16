import numpy as np
import torch
from sklearn.manifold import TSNE
from tqdm import tqdm

from models.autoEncoder import AutoEncoder
from util.data import data_generator
from matplotlib import pyplot as plt

from util.plot_tsne import PlotUtil

root = '../data/mnist'

max_batches = 0

device = "cpu"
if torch.cuda.is_available():
    device = "cuda:0"


def plot_features(model, num_classes, num_feats, batch_size, test_loader):
    preds = np.array([]).reshape((0, 1))
    gt = np.array([]).reshape((0, 1))
    feats = np.array([]).reshape((0, num_feats))
    targets = np.array([]).reshape((0, 1))
    model.eval()
    with torch.no_grad():
        for idx, (x1, target) in tqdm(enumerate(test_loader), position=0, leave=True, desc="Eval model"):
            if 0 < max_batches < idx:
                break
            x1 = x1.to(device=device, dtype=torch.float)
            out = model(x1)
            if len(out) != batch_size:
                break
            out = out.cpu().data.numpy()
            out = out.reshape((batch_size, num_feats))
            feats = np.append(feats, out, axis=0)
            target = target.cpu().data.numpy().reshape((-1, 1))
            targets = np.append(targets, target, axis=0)
    print("Eval done starting t-SNE")

    plt_util = PlotUtil(feats, targets, range(num_classes), "t-SNE")
    plt_util.plot_tsne()


def main():
    model = AutoEncoder(1, 3, 1).to(device=device)
    model.load_state_dict(torch.load("../content/saved_models/auto_encoder.pth"))
    model.eval()

    _, test_loader = data_generator(root, 128, clr=False, shuffle=True)

    plot_features(model, 10, 28*28, 128, test_loader)


if __name__ == '__main__':
    main()