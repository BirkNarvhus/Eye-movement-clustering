"""
Usage:
    python ./showEncoderImages.py <checkpoint_path>

Description:
    Test the model and show the images before and after autoencoder
"""
import os
import sys

import torch

from models.pre_tests.autoEncoder import AutoEncoder
from util.dataUtils.data import data_generator
from matplotlib import pyplot as plt

root = '../data/mnist'


def main():
    if len(sys.argv) < 2:
        print("Usage: python showEncoderImages.py <checkpoint_path>")
        return

    filepaths = sys.argv[1]

    if not os.path.exists(filepaths):
        print("File does not exist")
        return

    auto_encoder = AutoEncoder(1, 3, 1)
    auto_encoder.load_state_dict(torch.load(filepaths))
    auto_encoder.eval()
    _, test_loader = data_generator(root, 5, clr=False, shuffle=True)

    with torch.no_grad():
        x = next(iter(test_loader))[0]
        z = auto_encoder(x)

    fig, ax = plt.subplots(2, 5)
    for i in range(5):
        ax[0, i].imshow(x[i].squeeze().numpy(), cmap='gray')
        ax[1, i].imshow(z[i].squeeze().numpy(), cmap='gray')
    plt.show()


if __name__ == '__main__':
    main()
