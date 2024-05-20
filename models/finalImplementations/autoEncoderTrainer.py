"""
Usage:
    python ./autoEncoderTrainer.py <optional: model_file>

Description:
    This scripts trains the auto-encoder model. The model is trained on the OpenEDS dataset.
    The model is trained using adam optimizer. Can load previous model from a checkpoint file.

"""

import os
from pathlib import Path

import torch
from datetime import datetime
import warnings
import yaml


import sys
path_root = Path(__file__).parents[2]
sys.path.append(str(path_root))

from models.finalImplementations.EncoderDecoder import EncoderDecoder
from util.layerFactory import LayerFactory
from util.testingUtils.checkpointsLogging import CheckpointUtil
from util.dataUtils.dataset_loader import OpenEDSLoader
from util.dataUtils.transformations import *
from models.finalImplementations.customLoss import DiceCrossEntrepy

warnings.simplefilter("ignore", UserWarning)

device = "cpu"
if torch.cuda.is_available():
    device = "cuda:0"

relative_path = ""

configs = {}


def tuple_constructor(loader, node):
    # Load the sequence of values from the YAML node
    values = loader.construct_sequence(node)
    # Return a tuple constructed from the sequence
    return tuple(values)


# Register the constructor with PyYAML
yaml.SafeLoader.add_constructor('tag:yaml.org,2002:python/tuple',
                                tuple_constructor)


with open("configs/modelConfig.yaml") as stream:
    try:
        configs = yaml.load(stream, Loader=yaml.SafeLoader)
    except yaml.YAMLError as exc:
        print(exc)
        exit(1)


root = configs['data_params']['root']  # relative_path + 'data/openEDS/openEDS'
save_path = configs['data_params']['save_path']  # relative_path + 'data/openEDS/openEDSSplit.npy'

training_runs = configs['hyper_params']['training_runs']
batch_size = configs['hyper_params']['batch_size']
log_interval = configs['hyper_params']['log_interval']
lr = configs['hyper_params']['lr']
n_epochs = configs['hyper_params']['n_epochs']
steps = 0
max_batches = configs['hyper_params']['max_batches']  # all if 0
lossfunction = DiceCrossEntrepy()

arc_filename_enc = configs['model']['encoder_layers']  # relative_path + "content/Arc/" + "model_5.csv"
arc_filename_dec = configs['model']['decoder_layers']  # relative_path + "content/Arc/" + "model_5_reverse.csv"

model_name = arc_filename_enc.split('/')[2].split('.')[0]

checkpoint_dir = configs['misc']['checkpoint_dir']  # relative_path + 'content/saved_models/DiceCrossEntrepyAuto_enc/' + model_name
output_dir = configs['misc']['output_dir']  # relative_path + 'content/saved_outputs/autoEnc/'


transformations = [
    Crop_top(20),  # centers the image better
    Crop((256, 256)),
    Normalize(76.3, 41.7)
]
shuffle = configs['data_params']['shuffle'] # True
split_frames = configs['data_params']['split_frames']  # 6
loader = OpenEDSLoader(root, batch_size=batch_size, shuffle=shuffle, max_videos=None, save_path=save_path,
                       save_anyway=False,
                       transformations=transformations, sim_clr=False, split_frames=split_frames)

train_loader, test_loader, _ = loader.get_loaders()

lay_fac = LayerFactory()
res_interval = configs['model']['res_interval'] # 2

lay_fac.read_from_file(arc_filename_dec, full_block_res=True, res_interval=res_interval)
layers_dec = lay_fac.generate_layer_array()

lay_fac.read_from_file(arc_filename_enc, full_block_res=True, res_interval=res_interval)
layers_enc = lay_fac.generate_layer_array()

dil_bottle_neck_in_channels = configs['model']['dil_bottleneck']['bottleneck_input']  # 200
dil_bottle_neck_out_channels = configs['model']['dil_bottleneck']['bottleneck_output']  # 200
dil_factors = configs['model']['dil_bottleneck']['factors']  # (1, 2, 2)

lin_bottleneck = configs['model']['linear_bottleneck']['use']  # True
lin_layers = configs['model']['linear_bottleneck']['layers']  # 3
lin_bottle_in = configs['model']['linear_bottleneck']['bottleneck_input']  # 200 * 8 * 8
lin_bottle_hidden = configs['model']['linear_bottleneck']['hidden']  # 1000
lin_bottle_out = configs['model']['linear_bottleneck']['bottleneck_output']  # 120 * 8 * 8

stream_buffer = configs['model']['stream_buffer']  # False

model = EncoderDecoder(layers_enc, layers_dec, dil_bottle_neck_in_channels, dil_bottle_neck_out_channels,
                       dil_factors=dil_factors, lin_bottleneck=lin_bottleneck, lin_bottleneck_layers=lin_layers,
                       lin_bottleneck_channels=(lin_bottle_in, lin_bottle_hidden, lin_bottle_out), stream_buffer=stream_buffer)

weight_decay = configs['hyper_params']['weight_decay']  # 1e-6

optimizer = torch.optim.Adam(
    [params for params in model.parameters() if params.requires_grad],
    lr=lr,
    weight_decay=weight_decay,
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


def main():
    global model, optimizer, steps, n_epochs, checkpoint_dir, lossfunction, batch_size, log_interval, device
    if len(sys.argv) >= 2:
        load_file_path = sys.argv[1]
        load_file_dir_path = os.path.dirname(load_file_path)
        load_file_name = os.path.basename(load_file_path)
        print("Loading model checkpoint from: ", load_file_path)
        checkpoint_util = CheckpointUtil(load_file_dir_path)
        model, optimizer, start_epoch, _, loss = checkpoint_util.load_checkpoint(model, optimizer, load_file_name,
                                                                    reset_optimizer=True)
        print("Loaded model from epoch: ", start_epoch, " with loss: ", loss)
    model.to(device)

    loss_fn = lossfunction

    start_time = datetime.now().replace(microsecond=0)

    checkpoint_util = CheckpointUtil(checkpoint_dir)
    test_loss_buffer = []
    print("Starting training at {}...".format(start_time))
    best_loss = 1000000
    for epoch in range(n_epochs):
        print("")

        print("Epoch: ", epoch +1, "/", n_epochs, " at ", datetime.now().replace(microsecond=0), "...")
        print("")
        train_loss = 0
        model.train()
        time0 = datetime.now()
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

            time_now = datetime.now()
            delta_time = time_now - time0
            time0 = time_now
            if idx > 0:
                current_time = time_now.replace(microsecond=0) - start_time
                predicted_finish = delta_time * (len(train_loader)) * (n_epochs - epoch - 1) + current_time
                time_left = predicted_finish - current_time

                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tSteps: {}\t{}s/it\tRunning Time: {} - {}\tTime left: {}'.format(
                    epoch + 1, idx * batch_size, len(train_loader) * batch_size,
                           100. * idx / len(train_loader), train_loss.item(), batch_size,
                           str(delta_time), str(current_time), str(predicted_finish), str(time_left)))


        loss = test(test_loader, model)

        if train_loss < best_loss:
            best_loss = train_loss

        if epoch % log_interval == 0 and epoch > 0 or epoch == n_epochs - 1:
            checkpoint_util.save_checkpoint(model, optimizer, epoch, train_loss, loss, best_loss, False)

    print("Training finished. at {}".format(datetime.now().replace(microsecond=0)))


if __name__ == '__main__':
    prev_checkpoint_run = None
    for i in range(training_runs):
        if i > 0:
            index = checkpoint_dir.rfind("_")
            checkpoint_dir = checkpoint_dir[:index] + "_" + str(i)
        else:
            checkpoint_dir = checkpoint_dir + "_" + str(i)
        print("Starting training run ", i + 1)
        print("Checkpoint dir: ", checkpoint_dir)
        main()
        if i + 1 < training_runs:
            print("Resting optimizer for next run.")
            optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-6)
            loader = OpenEDSLoader(root, batch_size=batch_size, shuffle=True, max_videos=None, save_path=save_path,
                                   save_anyway=False,
                                   transformations=transformations, sim_clr=False)

            train_loader, test_loader, _ = loader.get_loaders()

