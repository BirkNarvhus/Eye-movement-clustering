import os

from tqdm import tqdm

from models.cleandup.encoderClassifer import Encoder_classifier
from models.larsOptim import LARS
import numpy as np
import torch
from models.simClrLoss import SimCLR_Loss
from util.dataset_loader import OpenEDSLoader
from util.layerFactory import LayerFactory

from util.plot_tsne import PlotUtil
from util.transformations import *
import time

device = "cpu"
if torch.cuda.is_available():
    device = "cuda:0"

root = '../../data/openEDS/openEDS'
save_path = '../../data/openEDS/openEDS.npy'

batch_size = 16
log_interval = 5
lr = 0.00001
n_epochs = 5
steps = 0
max_batches = 0 # all if 0
lossfunction = SimCLR_Loss(batch_size, 0.5)


def test(test_loader, model):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        idx = 0
        for x_i, x_j, _ in test_loader:
            idx += 1
            x_i = x_i.to(device=device, dtype=torch.float16)
            x_j = x_j.to(device=device, dtype=torch.float16)

            z_i = model(x_i)
            if len(z_i) != batch_size:
                break

            z_j = model(x_j)
            loss = lossfunction(z_i, z_j)
            test_loss += loss.item()

        test_loss /= idx
        print('\nTest set: Average loss: {:.4f}\n'.format(
            test_loss))
        return test_loss


def plot_features(model, num_classes, num_feats, batch_size):
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
    out = os.path.join('../../content/saved_models/final/', name.format(current_epoch))

    torch.save({'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()}, out)


if __name__ == '__main__':

    layerfac = LayerFactory()

    filename = "Arc/model_3.csv"

    layerfac.read_from_file(filename, full_block_res=True, res_interval=2)

    transformations = [
        TempStride(2),
        Crop_top(50),  # centers the image better
        RandomCrop(20),
        Crop((256, 256)),
        Rotate(20),
        Normalize(0, 1),
        Noise(0.6),
    ]

    loader = OpenEDSLoader(root, batch_size=batch_size, shuffle=True, max_videos=30, save_path=save_path, save_anyway=False,
                           transformations=transformations)

    train_loader, test_loader, _ = loader.get_loaders()

    data_size = 256
    output_size = 600

    model = Encoder_classifier(layerfac, data_size, output_size)
    model.to(device)

    optimizer = torch.optim.Adam(
    [params for params in model.parameters() if params.requires_grad],
    lr=lr,
    weight_decay=1e-6,
    )

    #optimizer = LARS(
    #    [params for params in model.parameters() if params.requires_grad],
    #    lr=0.2,
    #    weight_decay=1e-6,
    #    exclude_from_weight_decay=["batch_normalization", "bias"],
    #)
    loss_fn = lossfunction

    batch_loss = []
    time0 = time.time()
    for epoch in range(n_epochs):
        train_loss = 0
        model.train()
        for idx, (x_i, x_j) in enumerate(train_loader):

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
            if idx > 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tSteps: {}\t{:.2f}s/it'.format(
                    epoch, idx*batch_size, len(train_loader)*batch_size,
                          100. * idx / len(train_loader), train_loss.item() / log_interval, steps), time.time() - time0)

                time0 = time.time()

                batch_loss.append(train_loss.item() / log_interval)
                train_loss = 0
        test(test_loader, model)

    pytorch_total_params = sum(p.numel() for p in model.parameters())
    print('Number of params: {}'.format(pytorch_total_params))
    save_model(model, optimizer, n_epochs, 'simclr_model_{}.pth')
    plot_features(model, 10, 100, batch_size)


