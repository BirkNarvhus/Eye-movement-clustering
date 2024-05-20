"""
Usage:
    python ./Train_auto_encoder.py

Description:
    Trains an autoencoder model on the mnist dataset
"""
import torch
from torch import nn

from models.pre_tests.autoEncoder import AutoEncoder
from util.dataUtils.data import data_generator

epochs = 5
batch_size = 128
root = '../data/mnist'
num_classes = 10
input_channels = 1
num_feats = 64
lr = 0.001


device = "cpu"
if torch.cuda.is_available():
    device = "cuda:0"
log_interval = 5


def train(model, train_loader, optimizer, loss_function, device, log_interval, epoch):
    """
    Train the model on the train loader
    :param model:  the model
    :param train_loader:  the train loader
    :param optimizer:  the optimizer
    :param loss_function:  the loss function
    :param device:  the device
    :param log_interval:  the log interval
    :param epoch:  the epoch
    """
    model.train()
    train_loss = 0
    for batch_idx, (x, y) in enumerate(train_loader):
        x = x.to(device=device, dtype=torch.float)
        optimizer.zero_grad()
        z = model(x)
        loss = loss_function(z, x)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(x), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader),
                       loss.item() / len(x)))
    train_loss /= len(train_loader.dataset)
    print('====> Epoch: {} Average loss: {:.4f}'.format(
        epoch, train_loss))


def main():
    """
    Main function for training the model
    """
    train_loader, test_loader = data_generator(root, batch_size, clr=False)

    model = AutoEncoder(input_channels, 3, 1)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_function = nn.HuberLoss()

    for epoch in range(1, epochs + 1):
        train(model, train_loader, optimizer, loss_function, device, log_interval, epoch)

    torch.save(model.state_dict(), "../content/saved_models/auto_encoder.pth")


if __name__ == '__main__':
    main()