import torch
from torch import nn
from Blocks import DownsampleLayer, Cumulativ_global_pooling, Projection, MultiResLayer
from util.layerFactory import LayerFactory
from util.modelUtils import get_n_params


class Encoder(nn.Module):
    def __init__(self, layers):
        super(Encoder, self).__init__()

        self.convLayers = nn.ModuleList()

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
    def __init__(self, flattend_out, hidden_features, classes):
        super(BottleNeck, self).__init__()

        self.linear = nn.Linear(flattend_out, hidden_features)
        self.linear2 = nn.Linear(hidden_features, classes)

        self.relu = nn.ReLU()

        self.net = nn.Sequential(self.linear, self.relu, self.linear2)

    def forward(self, x):
        return self.net(x)


class Encoder_classifier(nn.Module):
    def __init__(self, layer_fac, data_size, output_size, hidden_encoder_pro=134, hidden_linear_features=500):
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
    layerfac = LayerFactory()

    filename = "Arc/model_3.csv"

    layerfac.read_from_file(filename, full_block_res=True, res_interval=2)

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