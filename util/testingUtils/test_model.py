import os

import torch
import numpy as np
import cv2
import sys
from torch import nn
from tqdm import tqdm

sys.path.append('C:\\Users\\vizlab_stud\\Documents\\pythonProjects\\eye-movement-classification')

from models.cleandup.kMeansDownstream import KMeansDownstream
from util.plot_tsne import PlotUtil
from models.cleandup.EncoderDecoder import EncoderDecoder
from util.testingUtils.checkpointsLogging import CheckpointUtil
from util.dataset_loader import OpenEDSLoader
from util.layerFactory import LayerFactory
from util.transformations import Crop_top, TempStride, Crop, Normalize

relative_path = ""  # "../"

batch_size = 32
save_path = relative_path + 'data/openEDS/openEDS.npy'
root = relative_path + 'data/openEDS/openEDS'
Out_folder = 'content/saved_outputs/'

device = "cpu" if not torch.cuda.is_available() else "cuda:0"

def load_model(model_file, legacy=False, remove_decoder=False):
    model_dir = model_file[:model_file.rfind('/')]
    model_file_name = model_file[model_file.rfind('/') + 1:]

    transformations = [
        Crop_top(20),  # centers the image better
        Crop((256, 256)),
        Normalize(76.3, 41.7)
    ]

    check_loader = CheckpointUtil(model_dir)


    '''
        model, optimizer = load_auto_encoder(arc_filename_enc, arc_filename_dec, 216,
                                         216, lr, torch.optim.Adam, False,
                                         2, 1e-6)
    '''

    lay_fac = LayerFactory()
    lay_fac.read_from_file("content/Arc/" + "model_3_v3_reverse.csv", full_block_res=True, res_interval=2)
    layers_dec = lay_fac.generate_layer_array()

    lay_fac.read_from_file("content/Arc/" + "model_4.csv", full_block_res=True, res_interval=2)
    layers_enc = lay_fac.generate_layer_array()

    model = EncoderDecoder(layers_enc, layers_dec, 400, 400,
                           dil_factors=(1, 2, 2), lin_bottleneck=True, lin_bottleneck_layers=3,
                           lin_bottleneck_channels=(400 * 8 * 8, 2000, 64 * 8 * 8), stream_buffer=False, remove_decoder=remove_decoder)

    optimizer = torch.optim.Adam(
        [params for params in model.parameters() if params.requires_grad],
        lr=0.001,
        weight_decay=1e-6,
    )
    return check_loader.load_checkpoint(model=model, optimizer=optimizer, check_point_name=model_file_name)


def plot_test(model, data_loader, save=False):
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
    model.to(device)

    flatten = nn.Flatten()
    model.eval()
    with (torch.no_grad()):
        print("Encoding with model")
        output_buffer = torch.tensor([])
        for batch in tqdm(data_loader):
            batch = batch.to(device)
            encoder_output = model(batch)
            if output_buffer.shape[0] == 0:
                output_buffer = flatten(encoder_output).cpu().detach()
            else:
                output_buffer = torch.cat([output_buffer, flatten(encoder_output).cpu().detach()], 0)

        print("Encoded data shape: ", output_buffer.shape)
        print("Running KMeans on the encoded data...")

        # do kmeans on the output_buffer
        kmds = KMeansDownstream(4)
        kmds.fit(output_buffer)
        print("KMeans Done, plotting...")
        print(output_buffer.shape)

        PlotUtil(output_buffer, "KMeans Downstream Auto enc", mode="TSNE", root=Out_folder, show=True, dims=3
                 ).plot_tsne_centers(kmds.get_cluster_centers())
        kmds.save(Out_folder + 'kmeans_output.npy')


def main():
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
        else:
            print("Could Not find mode: ", mode)
            sys.exit(1)

    model, _, epoch, best_loss, loss = load_model(sys.argv[1], legacy=True, remove_decoder=(mode == "kmeans"))
    print("Loaded model from file: ", sys.argv[1])
    print("model stats: epoch: ", epoch, " best_loss: ", best_loss, " loss: ", loss)
    transformations = [
        TempStride(2),
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
    else:
        plot_test(model, train_loader, save=(mode == "save"))


if __name__ == '__main__':
    main()
