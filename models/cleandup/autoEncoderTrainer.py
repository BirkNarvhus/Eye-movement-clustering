import os

from torch import nn
from tqdm import tqdm
import numpy as np
import torch
from datetime import datetime
import warnings


import sys

from models.cleandup.EncoderDecoder import EncoderDecoder

sys.path.append('C:\\Users\\vizlab_stud\\Documents\\pythonProjects\\eye-movement-classification')

from util.checkpointsLogging import CheckpointUtil
from util.dataset_loader import OpenEDSLoader
from util.layerFactory import LayerFactory

from util.plot_tsne import PlotUtil
from util.transformations import *

warnings.simplefilter("ignore", UserWarning)

device = "cpu"
if torch.cuda.is_available():
    device = "cuda:0"

relative_path = "../../"


root = relative_path + 'data/openEDS/openEDS'
save_path = relative_path + 'data/openEDS/openEDS.npy'



batch_size = 16
log_interval = 2
lr = 0.00001
n_epochs = 1
steps = 0
max_batches = 0  # all if 0
lossfunction = nn.HuberLoss()

arc_filename_enc = relative_path + "content/Arc/model_3.csv"
arc_filename_dec = relative_path + "content/Arc/model_3_reverse.csv"

model_name = arc_filename_enc.split('/')[2].split('.')[0] + "auto_encoder"

checkpoint_dir = relative_path + 'content/saved_models/clr_checkpoints/' + model_name
output_dir = relative_path + 'content/saved_outputs/'
'''
transformations = [
    TempStride(2),
    Crop_top(20),  # centers the image better
    RandomCrop(20),
    Crop((256, 256)),
    Rotate(30),
    Normalize(0, 1),
    Noise(0.3),
]
'''
transformations = [
    TempStride(2),
    Crop_top(20),  # centers the image better
    Crop((256, 256))
]

layerfac = LayerFactory()


layerfac.read_from_file(arc_filename_enc, full_block_res=True, res_interval=2)
layers_enc = layerfac.generate_layer_array()


layerfac.read_from_file(arc_filename_dec, full_block_res=True, res_interval=2)
layers_dec = layerfac.generate_layer_array()

loader = OpenEDSLoader(root, batch_size=batch_size, shuffle=True, max_videos=None, save_path=save_path,
                       save_anyway=False,
                       transformations=transformations, sim_clr=False)

train_loader, test_loader, _ = loader.get_loaders()


model = EncoderDecoder(layers_enc, layers_dec, 216, 216)

optimizer = torch.optim.Adam(
    [params for params in model.parameters() if params.requires_grad],
    lr=lr,
    weight_decay=1e-6,
)

'''
optimizer = LARS(
        [params for params in model.parameters() if params.requires_grad],
        lr=0.2,
        weight_decay=1e-6,
        exclude_from_weight_decay=["batch_normalization", "bias"],
    )

'''

print("running on ", device)


def test(test_loader, model):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        idx = 0
        for x in test_loader:
            idx += 1
            x = x.to(device=device, dtype=torch.float)

            z = model(x)
            if len(z) != batch_size:
                break
            loss = lossfunction(x, z)
            test_loss += loss.item()
        if idx == 0:
            return test_loss
        test_loss /= idx
        print('\nTest set: Average loss: {:.4f}'.format(
            test_loss))
        return test_loss


def plot_features(model, num_feats, batch_size, loader, name="test"):
    feats = np.array([]).reshape((0, num_feats))
    model.eval()
    print("extracting features...")
    with torch.no_grad():
        for idx, x1 in tqdm(enumerate(loader)):
            x1 = x1.to(device=device, dtype=torch.float)
            out = model(x1)
            if len(out) != batch_size:
                break
            out = out.cpu().data.numpy()
            if feats.shape[0] > 0:
                feats = np.append(feats, out, axis=0)
            else:
                feats = out

        with open(output_dir + '{}-feats.npy'.format(model_name), 'wb') as f:
            np.save(f, feats)
    plt_util = PlotUtil(feats, name, mode="PCA", root=output_dir)
    plt_util.plot_tsne()


if __name__ == '__main__':
    model.to(device)

    loss_fn = lossfunction

    start_time = datetime.now().replace(microsecond=0)

    checkpoint_util = CheckpointUtil(checkpoint_dir)

    print("Starting training at {}...".format(start_time))
    time0 = datetime.now()
    best_loss_test = 1000000
    best_loss = 1000000
    for epoch in range(n_epochs):
        print()
        train_loss = 0
        model.train()
        for idx, x_i in enumerate(train_loader):
            x_i = x_i.to(device, dtype=torch.float)

            z_i = model(x_i)
            loss = loss_fn(x_i, z_i)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # Validation
            train_loss = loss
            steps += batch_size
            if idx > 0:

                current_time = datetime.now().replace(microsecond=0) - start_time
                delta_time = datetime.now() - time0
                predicted_finish = delta_time * (len(train_loader)) * (n_epochs - epoch + 1)
                + delta_time * (len(train_loader) - (idx+1)) + current_time
                time_left = predicted_finish - current_time

                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tSteps: {}\t{}s/it\tRunning Time: {} - {}\tTime left: {}'.format(
                    epoch, idx * batch_size, len(train_loader) * batch_size,
                           100. * idx / len(train_loader), train_loss.item(), batch_size,
                           str(delta_time), str(current_time), str(predicted_finish), str(time_left)))

                time0 = datetime.now()

        loss = test(test_loader, model)
        if loss < best_loss_test:
            best_loss_test = loss

        if train_loss < best_loss:
            best_loss = train_loss

        if (loss == best_loss or train_loss == best_loss) and epoch > 0:
            checkpoint_util.save_checkpoint(model, optimizer, epoch, train_loss, loss, best_loss, True)
        elif epoch % log_interval == 0 and epoch > 0 or epoch == n_epochs - 1:
            checkpoint_util.save_checkpoint(model, optimizer, epoch, train_loss, loss, best_loss, False)
    pytorch_total_params = sum(p.numel() for p in model.parameters())
    print('Number of params: {}'.format(pytorch_total_params))
    print("Dim reducing and plotting...")
    plot_features(model, 100, batch_size, loader=test_loader, name="test")
    plot_features(model, 100, batch_size, loader=train_loader, name="train")
    print("Training finished. at {}".format(datetime.now().replace(microsecond=0)))
