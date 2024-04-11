import torch
from torch import nn

from models.autoencoder_3d import AutoEncoder
from util.data import data_generator
from util.dataset_loader import OpenEDSLoader

epochs = 5
batch_size = 32
root = '../data/openEDS/openEDS'
save_path = '../data/openEDS/openEDS.npy'
input_channels = 1
lr = 0.001


device = "cpu"
if torch.cuda.is_available():
    device = "cuda:0"
log_interval = 5


def train(model, train_loader, optimizer, loss_function, device, log_interval, epoch):
    model.train()
    train_loss = 0
    for batch_idx, x in enumerate(train_loader):
        x = x.to(device=device, dtype=torch.float)
        optimizer.zero_grad()
        z = model(x)
        loss = loss_function(x, z)
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
    train_loader, test_loader, val_loader = OpenEDSLoader(root, batch_size, shuffle=True, max_videos=None,
                                                          save_path=save_path, save_anyway=False).get_loaders()

    hidden_size = 64*7*7
    layers = ((16, 32, 2, 1, 1, 5), (32, 64, 1, 1, 1, 5), (64, 64, 1, 1, 1, 5))

    model = AutoEncoder(layers=layers)

    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_function = nn.CrossEntropyLoss()

    for epoch in range(1, epochs + 1):
        train(model, train_loader, optimizer, loss_function, device, log_interval, epoch)

    torch.save(model.state_dict(), "../content/saved_models/auto_encoder3d.pth")


if __name__ == '__main__':
    main()