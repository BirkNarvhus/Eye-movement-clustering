import numpy as np
import torch
from torch import nn
from Blocks import DownsampleLayer, Cumulativ_global_pooling, Projection, MultiResLayer
from util.layerFactory import LayerFactory
from torchinfo import summary
from util.modelUtils import get_n_params


class Encoder(nn.Module):
    def __init__(self, layers, global_pooling=False):
        super(Encoder, self).__init__()

        self.convLayers = nn.ModuleList()

        for i, layer in enumerate(layers):
            print(layer)
            layer_type = layer.pop(0)
            if i != 0:
                print(layer)
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
        self.global_pooling = Cumulativ_global_pooling() if global_pooling else None

    def forward(self, x):
        buffer = x.split(12, 2)
        stream_buffer = [[] for x in range(len(self.convLayers))]
        output_buffer = []
        for i in range(len(buffer)):
            x = buffer[i]
            print("Running on slice: ", i + 1)
            for buffer_index, layer in enumerate(self.convLayers):
                if len(stream_buffer[buffer_index]) > 0:
                    x = torch.cat([stream_buffer[buffer_index], x], 2)
                stream_buffer[buffer_index] = x[:, :, -3:, :, :]
                x = layer(x)[:, :, -12:, :, :]  # dont care about the first 3 because they are already in the buffer
            output_buffer.append(x)
        if self.global_pooling is not None:
            x = self.global_pooling(torch.cat(output_buffer, dim=2))
        else:
            x = x[:, :, -1:, :, :]  # take the last time slize witch hopefully contains all the information
        return x


class BottleNeck(nn.Module):
    def __init__(self, flattend_out, hidden_features, classes):
        super(BottleNeck, self).__init__()

        self.linear = nn.Linear(flattend_out, hidden_features)
        self.linear2 = nn.Linear(hidden_features, classes)

        self.relu = nn.ReLU()

        self.net = nn.Sequential(self.linear, self.relu, self.linear2)

    def forward(self, x):
        return self.net(x)


class Encoder_classifier(nn.Module):
    def __init__(self, layer_fac, data_size, output_size, hidden_encoder_pro=744, hidden_linear_features=2048):
        super(Encoder_classifier, self).__init__()
        self.encoder = Encoder(layer_fac.generate_layer_array(), global_pooling=True)

        self.init_down_size = nn.Conv3d(1, 16, (1, 1, 1), stride=(1, 2, 2), padding=0)

        data_size = data_size // 2 # init downsize

        downscale_factor, last_feature_size = layer_fac.get_last_size()

        self.project_encoder_out = Projection((last_feature_size, hidden_encoder_pro))

        class_input_size = hidden_encoder_pro * (data_size // downscale_factor)**2  # ONLY WORKS IF THE DATA IS SQUARE

        self.flatten = nn.Flatten()
        self.bottle_neck = BottleNeck(class_input_size, hidden_linear_features, output_size)

        self.net = nn.Sequential(self.init_down_size, self.encoder, self.project_encoder_out, self.flatten, self.bottle_neck)

    def forward(self, x):

        return self.net(x)


def test():
    layerfac = LayerFactory()

    filename = "Arc/model_1.csv"

    layerfac.read_from_file(filename, full_block_res=True, res_interval=3)
    print(len(layerfac.generate_layer_array()))
    data_size = 256
    output_size = 600

    model = Encoder_classifier(layerfac, data_size, output_size)
    #x = torch.randn(1, 1, 120, data_size, data_size)
    #model(x)
    print(summary(model, input_size=(32, 1, 120, data_size, data_size)))
    print("num params ", get_n_params(model))

if __name__ == '__main__':
    test()