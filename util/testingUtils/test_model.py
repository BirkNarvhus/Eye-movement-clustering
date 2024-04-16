import torch
import numpy as np
import cv2
import sys

sys.path.append('/')

from models.cleandup.EncoderDecoder import EncoderDecoder
from util.testingUtils.checkpointsLogging import CheckpointUtil
from util.dataset_loader import OpenEDSLoader
from util.layerFactory import LayerFactory
from util.transformations import Crop_top, TempStride, Crop, Normalize

relative_path = "" #"../"

batch_size = 8
save_path = relative_path + 'data/openEDS/openEDS.npy'
root = relative_path + 'data/openEDS/openEDS'


def load_model(model_file):
    model_dir = model_file[:model_file.rfind('/')]
    model_file_name = model_file[model_file.rfind('/') + 1:]

    check_loader = CheckpointUtil(model_dir)
    arc_filename_enc = relative_path + "content/Arc/model_3.csv"
    arc_filename_dec = relative_path + "content/Arc/model_3_reverse.csv"

    layerfac = LayerFactory()

    layerfac.read_from_file(arc_filename_enc, full_block_res=True, res_interval=2)
    layers_enc = layerfac.generate_layer_array()

    layerfac.read_from_file(arc_filename_dec, full_block_res=True, res_interval=2)
    layers_dec = layerfac.generate_layer_array()

    model = EncoderDecoder(layers_enc, layers_dec, 216, 216)

    optimizer = torch.optim.Adam(
        [params for params in model.parameters() if params.requires_grad],
        lr=0.0001,
        weight_decay=1e-6,
    )
    return check_loader.load_checkpoint(model=model, optimizer=optimizer, check_point_name=model_file_name)


def plot_test(model, data_loader):
    model.eval()
    with torch.no_grad():
        batch = data_loader.__iter__().__next__()
        auto_encoded_output = model(batch)

        fps = 30
        slowdown_factor = 40

        for x in range(batch_size):
            batch_one = batch[x].numpy().astype(np.float32)

            auto_encoded_output_one = auto_encoded_output[x].numpy().astype(np.float32)
            for i in range(60):
                frame_a = batch_one[:, i, :, :]
                frame_b = auto_encoded_output_one[:, i, :, :]
                frame_a = np.reshape(frame_a, (256, 256))
                frame_b = np.reshape(frame_b, (256, 256))
                frame = np.concatenate((frame_a, frame_b), axis=1)
                cv2.imshow('Grayscale', frame)
                # Press Q on keyboard to  exit
                if cv2.waitKey((60 // fps) * slowdown_factor) & 0xFF == ord('q'):
                    break
        # plot the batch



def main():
    if len(sys.argv) < 2:
        print("Usage: python test_model.py <model_file>")
        sys.exit(1)

    model, _, epoch, best_loss, loss = load_model(sys.argv[1])
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
                           transformations=transformations, sim_clr=False)

    train_loader, test_loader, _ = loader.get_loaders()

    plot_test(model, train_loader)


if __name__ == '__main__':
    main()