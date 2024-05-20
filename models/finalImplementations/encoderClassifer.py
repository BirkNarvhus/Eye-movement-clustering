"""
This file contains the Encoder_classifier class which is a model that combines the Encoder and the Classifier
"""

from pathlib import Path

import torch
from torch import nn
import sys
path_root = Path(__file__).parents[2]
sys.path.append(str(path_root))

from models.finalImplementations.Blocks import DownsampleLayer, Cumulativ_global_pooling, Projection, MultiResLayer
from util.layerFactory import LayerFactory
from util.modelUtils import get_n_params


class Encoder(nn.Module):
    """
    Encoder class that takes a list of layers and creates a model from them
    Uses DownsampleLayer and MultiResLayer depending on the block type
    """
    def __init__(self, layers, stream_buffer=True):
        """

        :param layers: Layers from layer fac
        :param stream_buffer: If true, the model will use a stream buffer to save memory
        """
        super(Encoder, self).__init__()

        self.convLayers = nn.ModuleList()
        self.stream_buffer = stream_buffer
        for i, layer in enumerate(layers):
            layer_type = layer.pop(0)
            if i != 0:
                pro_from = layers[i-1][-1][-1][0][-1]
                pro_to = layer[0][0][0][0]
                if pro_from != pro_to:
                    self.convLayers.append(Projection((pro_from, pro_to)))
            if layer_type == "down":
                self.convLayers.append(DownsampleLayer([[y[0] for y in x] for x in layer],
                                                       kernels_size=[[y[1] for y in x] for x in layer]))
            elif layer_type == "none":
                self.convLayers.append(MultiResLayer([[y[0] for y in x] for x in layer],
                                                       kernels_size=[[y[1] for y in x] for x in layer]))

        self.net = nn.Sequential(*self.convLayers)

    def forward(self, x):
        """
        Forward pass of the model
        Implements the stream buffer
        :param x: input tensor
        :return: output tensor
        """

        if not self.stream_buffer:
            return self.net(x)
        buffer = list(x.split(10, 2))
        stream_buffer = [[] for x in range(len(self.convLayers))]
        output_buffer = None

        for i in range(len(buffer)):
            x = buffer.pop(0)
            for buffer_index, layer in enumerate(self.convLayers):
                if len(stream_buffer[buffer_index]) > 0:
                    x = torch.cat([stream_buffer[buffer_index], x], 2)
                else:
                    x = torch.cat([torch.zeros_like(x[:, :, : 3, :, :]), x], 2)
                stream_buffer[buffer_index] = x[:, :, -3:, :, :]
                if len(buffer) == 0:
                    x = layer(x)[:, :, -10:, :, :]
                else:
                    with torch.no_grad(): # I realy dont know if this is the right way to do it but it saves memory help plz
                        x = layer(x)[:, :, -10:, :, :]
            if output_buffer is None:
                output_buffer = x
            else:
                output_buffer = torch.cat([output_buffer, x], 2)

        return output_buffer


class BottleNeck(nn.Module):
    """
    Linear bottleneck class that takes a flattened input and creates a linear model from it
    Uses two linear layers with a relu activation
    """
    def __init__(self, flattend_out, hidden_features, classes):
        super(BottleNeck, self).__init__()

        self.linear = nn.Linear(flattend_out, hidden_features)
        self.linear2 = nn.Linear(hidden_features, classes)

        self.relu = nn.ReLU()

        self.net = nn.Sequential(self.linear, self.relu, self.linear2)

    def forward(self, x):
        return self.net(x)


class Encoder_classifier(nn.Module):
    """
    Encoder_classifier class that takes a layer fac, data size and output size to create a model
    Uses the Encoder and the BottleNeck class
    Used in contrastiv learning model
    """
    def __init__(self, layer_fac, data_size, output_size, hidden_encoder_pro=134, hidden_linear_features=500):
        """
        :param layer_fac: Layer factory object
        :param data_size: Size of the data
        :param output_size: Size of the output
        :param hidden_encoder_pro: Hidden features of the encoder
        :param hidden_linear_features: Hidden features of the linear model
        """
        super(Encoder_classifier, self).__init__()
        self.encoder = Encoder(layer_fac.generate_layer_array())

        self.init_down_size = nn.Conv3d(1, 16, (1, 1, 1), stride=(1, 2, 2), padding=0)

        data_size = data_size // 2 # init downsize

        downscale_factor, last_feature_size = layer_fac.get_last_size()

        self.projection_2d = nn.Conv2d(last_feature_size, hidden_encoder_pro, 1, stride=1, padding=0)

        class_input_size = hidden_encoder_pro * (data_size // downscale_factor)**2  # ONLY WORKS IF THE DATA IS SQUARE
        self.flatten = nn.Flatten()
        self.bottle_neck = BottleNeck(class_input_size, hidden_linear_features, output_size)

        self.cgp = Cumulativ_global_pooling()

        self.net = nn.Sequential(self.init_down_size, self.encoder, self.cgp, self.projection_2d, self.flatten,
                                 self.bottle_neck)

    def forward(self, x):

        return self.net(x)


def test():
    """
    Test function for the Encoder_classifier class
    """
    layerfac = LayerFactory()

    filename = "Arc/model_1.csv"
    relative_path = "../../content/"
    layerfac.read_from_file(relative_path + filename, full_block_res=True, res_interval=2)

    #name = layerfac.add_layer("down", pool_type="down")
    #layerfac.add_residual_block(name, ((16, 128), (128, 64), (64, 32)), ((3, 3, 3), (3, 3, 3), (3, 3, 3)))
    #name = layerfac.add_layer("down", pool_type="none")
    #layerfac.add_residual_block(name, ((64, 128), (128, 64), (64, 32)), ((3, 3, 3), (3, 3, 3), (3, 3, 3)))
    data_size = 256
    output_size = 600

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print("running on ",  device)

    model = Encoder_classifier(layerfac, data_size, output_size)
    x = torch.randn(16, 1, 60, data_size, data_size)
    x = model(x)
    print(x.shape)
    print("num params ", get_n_params(model))


if __name__ == '__main__':
    test()