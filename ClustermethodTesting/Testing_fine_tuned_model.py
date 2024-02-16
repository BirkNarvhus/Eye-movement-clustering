import numpy as np
import torch
from sklearn.manifold import TSNE
from tqdm import tqdm

from models.autoEncoder import AutoEncoder
from models.Pretrained_classifier import finetuned_encoder_classifier
from util.data import data_generator
from matplotlib import pyplot as plt

from util.plot_tsne import PlotUtil

root = '../data/mnist'

max_batches = 0
epochs = 5

device = "cpu"
if torch.cuda.is_available():
    device = "cuda:0"


def test(model, test_loader, loss_function, device, epoch):
    model.eval()
    with torch.no_grad():
        test_loss = 0
        correct = 0
        for x, y in test_loader:
            x = x.to(device=device, dtype=torch.float)
            y = y.to(device=device, dtype=torch.long)
            z = model(x)
            test_loss += loss_function(z, y).item()
            pred = z.argmax(dim=1, keepdim=True)
            correct += pred.eq(y.view_as(pred)).sum().item()
        test_loss /= len(test_loader.dataset)
        print('\nTest Epoch: {} Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(epoch,
            test_loss, correct, len(test_loader.dataset),
            100. * correct / len(test_loader.dataset)))


def train(model, train_loader, optimizer, loss_function, device, log_interval, epoch):
    model.train()
    train_loss = 0
    for batch_idx, (x, y) in enumerate(train_loader):
        x = x.to(device=device, dtype=torch.float)
        optimizer.zero_grad()
        z = model(x)
        loss = loss_function(z, y)
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
    autoEncoder = AutoEncoder(1, 3, 1).to(device=device)
    autoEncoder.load_state_dict(torch.load("../content/saved_models/auto_encoder.pth"))
    dsmodel = finetuned_encoder_classifier(autoEncoder.encoder, 3, 10).to(device=device)

    train_loader, test_loader = data_generator(root, 128, clr=False, shuffle=True)

    optimizer = torch.optim.Adam(dsmodel.parameters(), lr=0.001)

    loss_function = torch.nn.CrossEntropyLoss()

    for epoch in range(epochs):
        train(dsmodel, train_loader, optimizer, loss_function, device, 5, epoch)
        test(dsmodel, test_loader, loss_function, device, epoch)


if __name__ == '__main__':
    main()