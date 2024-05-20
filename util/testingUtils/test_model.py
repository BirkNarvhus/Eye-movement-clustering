"""
usage:
    python ./test_model.py <model_file> <optional: mode>
Description:
    Test the model with the given model file
    The model file should be a checkpoint file
    The mode can be None/video, kmeans, save, or svm
    None/video:  shows the video of the autoencoder output
    kmeans:  runs kmeans on the encoded data
    save:  saves the video of the autoencoder output
    svm:  runs svm on the encoded data

    This is the main testing scrip for the later models
    All non-video modes will plot the output with the PlotUtil
    The threshold values used for IVVT are hardcoded and not optimized, but gives an ok result

    modelConfig.yaml is used to load the model configuration
    THIS HAS TO MATCH THE MODEL CONFIGURATION USED IN MODEL_FIlE
"""

import os

import torch
import numpy as np
import cv2
import yaml

from sklearn import svm
from torch import nn
from tqdm import tqdm

import sys
from pathlib import Path
path_root = Path(__file__).parents[2]
sys.path.append(str(path_root))

from util.plot_dim_reduced import PlotUtil
from models.finalImplementations.EncoderDecoder import EncoderDecoder
from util.testingUtils.checkpointsLogging import CheckpointUtil
from util.dataUtils.dataset_loader import OpenEDSLoader
from util.layerFactory import LayerFactory
from util.dataUtils.transformations import Crop_top, Crop, Normalize
from util.ivtUtil.IVVt import IvvtHelper

relative_path = ""  # "../"

batch_size = 32
save_path = relative_path + 'data/openEDS/openEDS.npy'
root = relative_path + 'data/openEDS/openEDS'
Out_folder = 'content/saved_outputs/'

device = "cpu" if not torch.cuda.is_available() else "cuda:0"

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


def load_model(model_file, legacy=False, remove_decoder=False):
    """
    Load the model from the model file
    :param model_file:  the model file
    :param legacy:  if True, load the legacy model
    :param remove_decoder:  if True, remove the decoder
    :return:  the model, optimizer, epoch, best_loss, loss
    """
    model_dir = model_file[:model_file.rfind('/')]
    model_file_name = model_file[model_file.rfind('/') + 1:]

    check_loader = CheckpointUtil(model_dir)

    arc_filename_enc = configs['model']['encoder_layers']  # relative_path + "content/Arc/" + "model_5.csv"
    arc_filename_dec = configs['model']['decoder_layers']  # relative_path + "content/Arc/" + "model_5_reverse.csv"
    res_interval = configs['model']['res_interval']  # 2

    lay_fac = LayerFactory()
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

    model = EncoderDecoder(layers_enc, layers_dec, dil_bottle_neck_in_channels, dil_bottle_neck_out_channels,
                           dil_factors=dil_factors, lin_bottleneck=lin_bottleneck, lin_bottleneck_layers=lin_layers,
                           lin_bottleneck_channels=(lin_bottle_in, lin_bottle_hidden, lin_bottle_out), stream_buffer=False,
                           legacy=legacy, remove_decoder=remove_decoder)

    # does not matter, but required for loading state dict
    optimizer = torch.optim.Adam(
        [params for params in model.parameters() if params.requires_grad],
        lr=0.001,
        weight_decay=1e-6,
    )
    return check_loader.load_checkpoint(model=model, optimizer=optimizer, check_point_name=model_file_name)


def plot_test(model, data_loader, save=False):
    """
    Plot(show) the test videos
    uses cv2, and might have issues with some versions of cv2 and numpy
    :param model:  the model
    :param data_loader:  the data loader
    :param save:  if True, save the video
    """
    model.eval()
    with torch.no_grad():
        batch = data_loader.__iter__().__next__()
        auto_encoded_output = model(batch)
        framebuffer = []
        fps = 30
        slowdown_factor = 40

        for x in range(batch_size):
            batch_one = batch[x].numpy().astype(np.float32)

            auto_encoded_output_one = auto_encoded_output[x].numpy().astype(np.float32)
            for i in range(6):
                frame_a = batch_one[:, i, :, :]
                frame_b = auto_encoded_output_one[:, i, :, :]

                frame_a = np.array(torch.nn.functional.sigmoid(torch.tensor(frame_a)))
                frame_b = np.array(torch.nn.functional.sigmoid(torch.tensor(frame_b)))
                frame_a = np.reshape(frame_a, (256, 256))
                frame_b = np.reshape(frame_b, (256, 256))
                frame = np.concatenate((frame_a, frame_b), axis=1)
                cv2.imshow('Grayscale', frame)
                if save:
                    framebuffer.append(frame)
                # Press Q on keyboard to  exit
                if cv2.waitKey((60 // fps) * slowdown_factor) & 0xFF == ord('q'):
                    break
        if save:
            print("Saving video to: ", Out_folder + '/autoEncVideo/' + "frames" + '.npy')
            os.makedirs(Out_folder + '/autoEncVideo', exist_ok=True)
            np.save(Out_folder + '/autoEncVideo/' + "frames" + '.npy', framebuffer)


def do_kmeans(model, data_loader):
    """
    Run kmeans on the encoded data
    :param model:  the model
    :param data_loader:  the data loader
    """
    model = model.encoder

    model.to(device)

    flatten = nn.Flatten()
    model.eval()
    with (torch.no_grad()):
        print("Encoding with model")
        output_buffer = torch.tensor([])
        target_buffer = []

        ivvtHelper = IvvtHelper(sacadeThreshold=0.45, smoothperThreshold=0.15)
        for batch in tqdm(data_loader):
            batch = batch.to(device)
            target_buffer.append(ivvtHelper.classify_bach(batch * 41.7 + 76.3))
            encoder_output = model(batch)

            if output_buffer.shape[0] == 0:
                output_buffer = flatten(encoder_output).cpu().detach()
            else:
                output_buffer = torch.cat([output_buffer, flatten(encoder_output).cpu().detach()], 0)
        target_buffer = [x for sublist in target_buffer for x in sublist]
        print("Encoded data shape: ", output_buffer.shape)
        print("Running KMeans on the encoded data...")
        pUtil = PlotUtil(output_buffer, "KMeans Downstream Auto enc", mode="mds", root=Out_folder, show=True, dims=2,
                 kmeans_after=True)
        pUtil.plot_dim_reduced(targets_reg_alg=target_buffer)


def do_svm(model, data_loader):
    """
    Run svm on the encoded data
    :param model:  the model
    :param data_loader:  the data loader
    """
    model = model.encoder
    model.to(device)

    flatten = nn.Flatten()
    model.eval()
    with (torch.no_grad()):
        print("Encoding with model")
        output_buffer = torch.tensor([])
        target_buffer = []

        ivvtHelper = IvvtHelper(sacadeThreshold=0.45, smoothperThreshold=0.15)
        for batch in tqdm(data_loader):
            batch = batch.to(device)
            target_buffer.append(ivvtHelper.classify_bach(batch * 41.7 + 76.3))
            encoder_output = model(batch)

            if output_buffer.shape[0] == 0:
                output_buffer = flatten(encoder_output).cpu().detach()
            else:
                output_buffer = torch.cat([output_buffer, flatten(encoder_output).cpu().detach()], 0)
        target_buffer = [x for sublist in target_buffer for x in sublist]

        print("Encoded data shape: ", output_buffer.shape)
        print("Running svm on the encoded data...")

        supportvector = svm.SVC()
        supportvector.fit(output_buffer, target_buffer)
        targets = supportvector.predict(output_buffer)
        pUtil = PlotUtil(output_buffer, "SVM downstream", mode="kpca", root=Out_folder, show=True, dims=2,
                         kmeans_after=False)
        pUtil.plot_dim_reduced(targets=targets, targets_reg_alg=target_buffer)


def main():
    """
    Main function for parsing model file and mode
    calls the correct function based on the mode
    """
    if len(sys.argv) < 2:
        print("Usage: python test_model.py <model_file> <optional: mode>")
        sys.exit(1)
    mode = "autoencoder"
    if len(sys.argv) > 2:
        mode = sys.argv[2].lower()
        if mode == "kmeans":
            print("Running kmeans")
        elif mode == "save":
            print("Running and saving autoencoder output to: ", Out_folder)
        elif mode == "svm":
            print("Running svm")
        else:
            print("Could Not find mode: ", mode)
            sys.exit(1)

    model, _, epoch, best_loss, loss = load_model(sys.argv[1], legacy=True, remove_decoder=(mode == "kmeans"))
    print("Loaded model from file: ", sys.argv[1])
    print("model stats: epoch: ", epoch, " best_loss: ", best_loss, " loss: ", loss)
    transformations = [
        Crop_top(20),  # centers the image better
        Crop((256, 256)),
        Normalize(76.3, 41.7)
    ]

    loader = OpenEDSLoader(root, batch_size=batch_size, shuffle=True, max_videos=None, save_path=save_path,
                           save_anyway=False,
                           transformations=transformations, sim_clr=False, split_frames=6)

    train_loader, test_loader, _ = loader.get_loaders()
    if mode == "kmeans":
        do_kmeans(model, train_loader)
    elif mode == "svm":
        do_svm(model, train_loader)
    else:
        plot_test(model, train_loader, save=(mode == "save"))


if __name__ == '__main__':
    main()
