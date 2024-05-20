"""
Usage:
    python ./ClusterEncodedData.py <checkpoint_path>

Description:
    Test the model and cluster the encoded data
    The labels decide with color on the plot
"""
import os
import sys

import numpy as np
import scipy
import torch
from tqdm import tqdm

from models.pre_tests.autoEncoder import AutoEncoder
from util.dataUtils.data import data_generator

from util.plot_dim_reduced import PlotUtil

root = '../data/mnist'

max_batches = 0

device = "cpu"
if torch.cuda.is_available():
    device = "cuda:0"


def plot_features(model, num_classes, num_feats, batch_size, test_loader):
    """
    Uses the model to generate encoded features and plots them
    Runs on provided dataloader
    Uses PlotUtil to plot the data
    and kmeans after in plotutil

    :param model: the model
    :param num_classes: number of classes
    :param num_feats: number of classes for kmeans
    :param batch_size: batch size
    :param test_loader: test loader
    """
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
            target = target.cpu().data.numpy().reshape((-1, 1))
            targets = np.append(targets, target, axis=0)
            feats = np.append(feats, out, axis=0)

    plt_util = PlotUtil(feats, "test encoder", mode="pcak", show=True, dims=2, kmeans_after=True, kmeans_classes=num_classes)
    plt_util.plot_dim_reduced(targets=targets.squeeze())

def main():
    """
    Function for retrieving sys args and running plot method
    """

    if len(sys.argv) < 2:
        print("Usage: python ClusterEncodedData.py <checkpoint_path>")
        sys.exit(1)

    model_file = sys.argv[1]

    if not os.path.exists(model_file):
        print("The file does not exist")
        sys.exit(1)

    model = AutoEncoder(1, 3, 1).to(device=device)
    model.load_state_dict(torch.load(model_file))

    # better method of loading encoder. Dose not work the same if the models uses bottlneck
    model = model.encoder
    model.eval()

    _, test_loader = data_generator(root, 128, clr=False, shuffle=True)

    plot_features(model, 10, 3*7*7, 128, test_loader)


if __name__ == '__main__':
    main()