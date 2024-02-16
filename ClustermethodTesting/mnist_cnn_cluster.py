import os

from sklearn.manifold import TSNE

from models.larsOptim import LARS
from util.data import data_generator
import torch.nn as nn
import numpy as np
import torch
from models.simpleCnn import SimpleCnn
from models.simClrLoss import SimCLR_Loss
import matplotlib.pyplot as plt

from util.plot_tsne import PlotUtil

#mpl.use('Qt5Agg')

device = "cpu"
if torch.cuda.is_available():
    device = "cuda:0"

root = '../data/mnist'
batch_size = 128
log_interval = 5
lr = 0.001
input_size = 1
n_epochs = 5
steps = 0
max_batches = 5 # all if 0
permute = False
num_feats = 64
lossfunction = SimCLR_Loss(batch_size, 0.5)


def test():
    model.eval()
    test_loss = 0
    with torch.no_grad():
        idx = 0
        for x_i, x_j, _ in test_loader:
            idx += 1
            x_i = x_i.to(device=device, dtype=torch.float)
            x_j = x_j.to(device=device, dtype=torch.float)
            #target.to(device)

            z_i = model(x_i)
            if len(z_i) != batch_size:
                break

            z_j = model(x_j)
            loss = lossfunction(z_i, z_j)
            test_loss += loss.item()
            #pred = z_i.data.max(1, keepdim=True)[1]
            #correct += pred.eq(target.data.view_as(pred)).cpu().sum()

        test_loss /= idx
        print('\nTest set: Average loss: {:.4f}\n'.format(
            test_loss))
        return test_loss


def plot_features(model, num_classes, num_feats, batch_size):
    preds = np.array([]).reshape((0, 1))
    gt = np.array([]).reshape((0, 1))
    feats = np.array([]).reshape((0, num_feats))
    targets = np.array([]).reshape((0, 1))
    model.eval()
    with torch.no_grad():
        for idx, (x1, x2, target) in enumerate(train_loader):
            if 0 < max_batches < idx:
                break
            x1 = x1.to(device=device, dtype=torch.float)
            out = model(x1)
            if len(out) != batch_size:
                break
            out = out.cpu().data.numpy()
            feats = np.append(feats, out, axis=0)
            target = target.cpu().data.numpy().reshape((-1, 1))
            targets = np.append(targets, target, axis=0)
    plt_util = PlotUtil(feats, targets, "t-SNE")
    plt_util.plot_tsne()


def save_model(model, optimizer, current_epoch, name):
    out = os.path.join('../content/saved_models/', name.format(current_epoch))

    torch.save({'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()}, out)


if __name__ == '__main__':
    train_loader, test_loader = data_generator(root, batch_size, clr=True)

    model = SimpleCnn(input_size, num_feats)
    model.to(device)

    optimizer = LARS(
        [params for params in model.parameters() if params.requires_grad],
        lr=0.2,
        weight_decay=1e-6,
        exclude_from_weight_decay=["batch_normalization", "bias"],
    )


    loss_fn = lossfunction

    batch_loss = []

    for epoch in range(n_epochs):
        train_loss = 0
        model.train()
        for idx, (x_i, x_j, _) in enumerate(train_loader):
            if 0 < max_batches < idx:
                break

            x_i = x_i.to(device)
            x_j = x_j.to(device)
            #y_batch = y_batch.to(device)

            z_i = model(x_i)
            z_j = model(x_j)

            if len(z_i) != batch_size:
                continue
            loss = loss_fn(z_i, z_j)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # Validation
            train_loss += loss
            steps += 28
            if idx > 0 and idx % log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tSteps: {}'.format(
                    epoch, idx * batch_size, len(train_loader.dataset),
                          100. * idx / len(train_loader), train_loss.item() / log_interval, steps))

                batch_loss.append(train_loss.item() / log_interval)
                train_loss = 0
        test()

    pytorch_total_params = sum(p.numel() for p in model.parameters())
    print('Number of params: {}'.format(pytorch_total_params))
    save_model(model, optimizer, n_epochs, 'simclr_model_{}.pth')
    plot_features(model, 10, num_feats, batch_size)


