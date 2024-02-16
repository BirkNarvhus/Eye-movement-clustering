import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from util.clr_dataset import ClrDataset


def data_generator(root, batch_size, clr=False, shuffle=False):
    """
    Generator function for data loading of mnist datasett (for now)
    :param clr: if True, returns a dataset for contrastive learning
    :param root: root directory
    :param batch_size: batch size
    :return: data loader
    """
    if clr:
        train_set = ClrDataset(root=root, train=True, download=True,
                                transform=transforms.Compose([
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.1307,), (0.3081,)),
                                    transforms.RandomGrayscale(p=0.2)
                                ]))
        test_set = ClrDataset(root=root, train=False, download=True,
                              transform=transforms.Compose([
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.1307,), (0.3081,)),
                                   transforms.RandomGrayscale(p=0.3)
                               ]))
    else:
        train_set = datasets.MNIST(root=root, train=True, download=True,
                                   transform=transforms.Compose([
                                       transforms.ToTensor(),
                                       transforms.Normalize((0.1307,), (0.3081,))
                                   ]))
        test_set = datasets.MNIST(root=root, train=False, download=True,
                                  transform=transforms.Compose([
                                      transforms.ToTensor(),
                                      transforms.Normalize((0.1307,), (0.3081,))
                                  ]))

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=shuffle)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=shuffle)
    return train_loader, test_loader


def main():
    root = '../data/mnist'
    train_loader, test_loader = data_generator(root, batch_size=128)


if __name__ == '__main__':
    main()
