import torch

from models.autoEncoder import AutoEncoder
from util.data import data_generator
from matplotlib import pyplot as plt

root = '../data/mnist'


def main():
    auto_encoder = AutoEncoder(1, 3, 1)
    auto_encoder.load_state_dict(torch.load("../content/saved_models/auto_encoder.pth"))
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
