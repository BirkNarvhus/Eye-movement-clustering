import torch
from torch import nn
from encoderClassifer import Encoder
from Blocks import TempConvBlock, UpsampleLayer, MultiResLayer, Projection, Cumulativ_global_pooling
from util.layerFactory import LayerFactory
from util.modelUtils import get_n_params


class DilationBottleneck(nn.Module):
    def __init__(self, dil_factors=(1, 2, 4, 8), kernel=(3, 3, 3), in_channels=1, out_channels=1):
        super(DilationBottleneck, self).__init__()
        self.conv_layers = nn.ModuleList()
        for i, dil in enumerate(dil_factors):
            if i == 0:
                self.conv_layers.append(TempConvBlock(in_channels, out_channels, kernel, dilation_size=dil))
            elif i == len(dil_factors) - 1:
                self.conv_layers.append(TempConvBlock(out_channels, out_channels, kernel, dilation_size=dil))
            else:
                self.conv_layers.append(TempConvBlock(out_channels, out_channels, kernel, dilation_size=dil))
        self.net = nn.Sequential(*self.conv_layers)

    def forward(self, x):
        sum_output = None
        for layer in self.net:
            if sum_output is None:
                sum_output = x
            else:
                sum_output += x
            x = layer(x)
        x = sum_output + x
        return x


class Decoder(nn.Module):
    def __init__(self, layers):
        super(Decoder, self).__init__()

        self.convLayers = nn.ModuleList()

        for i, layer in enumerate(layers):
            layer_type = layer.pop(0)
            if i != 0:
                pro_from = layers[i - 1][-1][-1][0][-1]
                pro_to = layer[0][0][0][0]
                if pro_from != pro_to:
                    self.convLayers.append(Projection((pro_from, pro_to)))
            if i == 0:
                layer_type = "up"
            if layer_type == "up":
                self.convLayers.append(UpsampleLayer([[y[0] for y in x] for x in layer],
                                                     kernels_size=[[y[1] for y in x] for x in layer], causel=False))
            if layer_type == "temp_up":
                temp_modlist = nn.ModuleList()
                temp_modlist.append(nn.Upsample(scale_factor=(2, 1, 1), mode='trilinear'))
                temp_modlist.append(MultiResLayer([[y[0] for y in x] for x in layer],
                                                     kernels_size=[[y[1] for y in x] for x in layer], causel=False))
                temp_modlist = nn.Sequential(*temp_modlist)
                self.convLayers.append(temp_modlist)
            elif layer_type == "none":
                self.convLayers.append(MultiResLayer([[y[0] for y in x] for x in layer],
                                                     kernels_size=[[y[1] for y in x] for x in layer], causel=False))

            print("Layer ", i, " ", layer_type, " ", len(self.convLayers))

        self.net = nn.Sequential(*self.convLayers)

    def forward(self, x):
        return self.net(x)[..., 2:-2, :, :]


class Unsqeeze(nn.Module):
    def __init__(self, dim):
        super(Unsqeeze, self).__init__()
        self.dim = dim

    def forward(self, x):
        return x.unsqueeze(self.dim)


class EncoderDecoder(nn.Module):
    def __init__(self, encoder_layers, decoder_layers, bottleneck_input_channels, bottleneck_output_channels):
        super(EncoderDecoder, self).__init__()
        self.init_down_size = nn.Conv3d(1, 16, (1, 1, 1), stride=(1, 2, 2), padding=0)
        self.encoder = Encoder(encoder_layers)
        self.bottleneck = DilationBottleneck(in_channels=bottleneck_input_channels,
                                             out_channels=bottleneck_output_channels)
        self.cgp = Cumulativ_global_pooling()
        self.decoder = Decoder(decoder_layers)
        self.unsq = Unsqeeze(2)

        self.net = nn.Sequential(self.init_down_size, self.encoder, self.bottleneck,  self.cgp, self.unsq, self.decoder)

    def forward(self, x):
        for layer in self.net:
            print(x.shape)
            x = layer(x)
        return x


def test():


    '''
    encoder_output = torch.rand((8, 2, 60, 7, 7))

    bottle_neck = DilationBottleneck(in_channels=2, hidden_channels=2, out_channels=2)
    x = bottle_neck(encoder_output)
    print(x.shape, encoder_output.shape)
    '''

    '''
    channels = [[[744, 744], [744, 744]]]
    bottlneck_out = torch.rand((8, 744, 2,  7, 7))
    model = UpsampleLayer(channels, kernels_size=[[[3, 3, 3], [3, 3, 3]]],
                          stride=1, padding=1, upscale_kernel=(2, 2, 2), causel=False)
    x1 = model(bottlneck_out)

    x2 = model(x1)

    print(bottlneck_out.shape, x1.shape, x2.shape)
   
    '''

    relative_path = "../../content/Arc/"
    lay_fac = LayerFactory()
    lay_fac.read_from_file(relative_path + "model_3_reverse.csv")
    layers_dec = lay_fac.generate_layer_array()

    '''
    decoder = Decoder(layers)
    bottle_neck_output = torch.rand((8, 744, 1, 8, 8))

    x = decoder(bottle_neck_output)
    print(bottle_neck_output.shape, x.shape)
    print("num params ", get_n_params(decoder))
    '''

    lay_fac.read_from_file(relative_path + "model_3.csv")
    layers_enc = lay_fac.generate_layer_array()
    print(layers_enc)

    model = EncoderDecoder(layers_enc, layers_dec, 216, 216)

    x = torch.rand((8, 1, 60, 256, 256))

    x = model(x)

    print(x.shape)


if __name__ == "__main__":
    test()
