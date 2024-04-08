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
from datetime import datetime

import warnings
warnings.simplefilter("ignore", UserWarning)

device = "cpu"
if torch.cuda.is_available():
    device = "cuda:0"

root = '../../data/openEDS/openEDS'
save_path = '../../data/openEDS/openEDS.npy'

batch_size = 16
log_interval = 5
lr = 0.00001
n_epochs = 10
steps = 0
max_batches = 0  # all if 0
lossfunction = SimCLR_Loss(batch_size, 0.5)

arc_filename = "Arc/model_1.csv"

transformations = [
    TempStride(2),
    Crop_top(20),  # centers the image better
    RandomCrop(20),
    Crop((256, 256)),
    Rotate(40),
    Normalize(0, 1),
    Noise(0.6),
]
data_size = 256
output_size = 744


def test(test_loader, model):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        idx = 0
        for x_i, x_j in test_loader:
            idx += 1
            x_i = x_i.to(device=device, dtype=torch.float)
            x_j = x_j.to(device=device, dtype=torch.float)

            z_i = model(x_i)
            if len(z_i) != batch_size:
                break

            z_j = model(x_j)
            loss = lossfunction(z_i, z_j)
            test_loss += loss.item()
        if idx == 0:
            return test_loss
        test_loss /= idx
        print('\nTest set: Average loss: {:.4f}\n'.format(
            test_loss))
        return test_loss


def plot_features(model, num_feats, batch_size):
    feats = np.array([]).reshape((0, num_feats))
    model.eval()
    print("extracting features...")
    with torch.no_grad():
        pbar = tqdm(enumerate(train_loader), total=len(train_loader))
        for idx, (x1, _) in enumerate(train_loader):
            x1 = x1.to(device=device, dtype=torch.float)
            out = model(x1)
            if len(out) != batch_size:
                break
            out = out.cpu().data.numpy()
            if feats.shape[0] > 0:
                feats = np.append(feats, out, axis=0)
            else:
                feats = out
            pbar.update(batch_size)
        pbar.close()

        with open('../../content/saved_models/final/feats.npy', 'wb') as f:
            np.save(f, feats)
    print("plotting t-SNE plot...")
    plt_util = PlotUtil(feats, "t-SNE")
    plt_util.plot_tsne()


def save_model(model, optimizer, name):
    out = os.path.join('../../content/saved_models/final/', name)

    torch.save({'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()}, out)


if __name__ == '__main__':

    layerfac = LayerFactory()

    model_name = arc_filename.split('/')[1].split('.')[0]

    layerfac.read_from_file(arc_filename, full_block_res=True, res_interval=3)

    loader = OpenEDSLoader(root, batch_size=batch_size, shuffle=True, max_videos=None, save_path=save_path,
                           save_anyway=False,
                           transformations=transformations)

    train_loader, test_loader, _ = loader.get_loaders()

    model = Encoder_classifier(layerfac, data_size, output_size, hidden_encoder_pro=600, hidden_linear_features=1000)
    model.to(device)

    optimizer = LARS(
        [params for params in model.parameters() if params.requires_grad],
        lr=0.2,
        weight_decay=1e-6,
        exclude_from_weight_decay=["batch_normalization", "bias"],
    )
    loss_fn = lossfunction

    batch_loss = []
    start_time = datetime.now().replace(microsecond=0)

    print("Starting training at {}...".format(start_time))
    time0 = datetime.now()
    for epoch in range(n_epochs):
        train_loss = 0
        model.train()
        for idx, (x_i, x_j) in enumerate(train_loader):
            x_i = x_i.to(device, dtype=torch.float)
            x_j = x_j.to(device, dtype=torch.float)
            # y_batch = y_batch.to(device)

            z_i = model(x_i)
            z_j = model(x_j)
            loss = loss_fn(z_i, z_j)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # Validation
            train_loss += loss
            steps += batch_size
            if idx > 0:
                current_time = datetime.now().replace(microsecond=0) - start_time
                delta_time = datetime.now() - time0
                predicted_finish = delta_time * len(train_loader) * n_epochs
                time_left = predicted_finish - current_time

                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tSteps: {}\t{}s/it\tRunning Time: {} - {}\tTime left: {}'.format(
                    epoch, idx * batch_size, len(train_loader) * batch_size,
                           100. * idx / len(train_loader), train_loss.item() / log_interval, batch_size,
                           str(delta_time), str(current_time), str(predicted_finish), str(time_left)))

                time0 = datetime.now()

                batch_loss.append(train_loss.item() / log_interval)
                train_loss = 0
        test(test_loader, model)

    pytorch_total_params = sum(p.numel() for p in model.parameters())
    print('Number of params: {}'.format(pytorch_total_params))
    save_model(model, optimizer, 'simclr_model_{}_{}.pth'.format(n_epochs, model_name))
    plot_features(model, 100, batch_size)
    print("Training finished. at {}".format(datetime.now().replace(microsecond=0)))
