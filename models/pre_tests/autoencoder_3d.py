"""
Early implementation for auto encoder for 3d data.
This implementation is Deprecated.
Used models.finalImplementations.EncoderDecoder instead.
"""

import numpy as np
import torch
from torch import nn


class Cumulativ_global_pooling(nn.Module):
    """
    This class is used to apply a global pooling on the input tensor.
    """
    def __init__(self):
        super(Cumulativ_global_pooling, self).__init__()

    def forward(self, x):
        return torch.mean(x, dim=3)


class Chomp1d(nn.Module):
    """
    This class is used to remove the padding from the temporal convolution.
    """
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class Blocks3d(nn.Module):
    """
    This class is used to apply a 3d convolution on the input tensor.
    If two_plus_1 is True, then apply 2+1D convolution.
    Uses 3d transposed convolution for upsampling if upsampling is True.
    """
    def __init__(self, input_channels, output_channels, kernel_size=(3, 3, 3), special_stride=1, temp_stride=1,
                 dilation_size=1, two_plus_1=True, upsamplign=False):
        """

        :param input_channels: Number of input channels.
        :param output_channels: Number of output channels.
        :param kernel_size: Size of the kernel.
        :param special_stride: Stride of the convolution in the special dimension.
        :param temp_stride: Stride of the convolution in the temporal dimension.
        :param dilation_size: Dilation of the convolution.
        :param two_plus_1: If True, apply 2+1D convolution.
        :param upsamplign: If True, apply 3d transposed convolution.
        """
        super(Blocks3d, self).__init__()
        padding_to_chomp = (kernel_size[0] - 1) * dilation_size
        self.chomp = Chomp1d(padding_to_chomp)
        if two_plus_1:
            if upsamplign:
                self.conv = nn.ConvTranspose3d(input_channels, output_channels, kernel_size=(1, kernel_size[1], kernel_size[2]),
                                      stride=(1, 1, 1),
                                      padding=0, dilation=dilation_size)
            else:
                self.conv = nn.Conv3d(input_channels, output_channels, kernel_size=(1, kernel_size[1], kernel_size[2]),
                                      stride=(1, special_stride, special_stride),
                                      padding="same" if special_stride == 1 else (0, 1, 1), dilation=dilation_size)
            padding = (padding_to_chomp, 0, 0)

            if upsamplign:
                self.conv2 = nn.ConvTranspose3d(output_channels, output_channels, kernel_size=(kernel_size[0], 1, 1),
                                   stride=(1, 1, 1),
                                   padding=0, dilation=dilation_size)
            else:
                self.conv2 = nn.Conv3d(output_channels, output_channels, kernel_size=(kernel_size[0], 1, 1),
                                       stride=(temp_stride, 1, 1),
                                       padding=padding, dilation=dilation_size)
            if upsamplign:
                self.net = nn.Sequential(self.conv, self.conv2)
            else:
                self.net = nn.Sequential(self.conv, self.conv2, self.chomp)
        else:
            padding = (padding_to_chomp, 1, 1)

            self.conv = nn.Conv3d(input_channels, output_channels, kernel_size=kernel_size,
                                  stride=(temp_stride, special_stride, special_stride),
                                  padding=padding, dilation=dilation_size)

            self.net = nn.Sequential(self.conv, self.chomp)

    def forward(self, x):
        return self.net(x)


class Residual_Block3d(nn.Module):
    """
    This class is used to apply a residual block on the input tensor.
    uses Blocks3d and BatchNorm3d.
    uses leaky relu as activation function.
    """
    def __init__(self, input_channels, output_channels, kernel_size=(3, 3, 3), special_down_sample=1, temp_stride=1,
                 dilation_size=1, num_blocks=1, upsamplign=False):
        super(Residual_Block3d, self).__init__()

        self.conv_layers = nn.ModuleList()
        self.relu = nn.LeakyReLU()
        self.bachNorm = nn.BatchNorm3d(output_channels)
        self.upsampler = nn.Upsample(scale_factor=(temp_stride, special_down_sample, special_down_sample), mode='trilinear') if upsamplign else None

        self.special_down_sample = nn.AvgPool3d((1, 2, 2), (1, special_down_sample, special_down_sample)) \
            if upsamplign is False and special_down_sample > 1  else None
        for i in range(num_blocks):

            self.conv_layers.append(
                Blocks3d(output_channels if i != 0 else input_channels, output_channels, kernel_size,
                         1, 1, i+1, two_plus_1=True))
            self.conv_layers.append(self.relu)
        self.net = nn.Sequential(*self.conv_layers)

        self.down_sample = (nn.Conv3d(input_channels, output_channels, 1, stride=(1, 1, 1),
                                     padding=0) if input_channels != output_channels else None)

    def forward(self, x):
        if self.upsampler is not None:
            x = self.upsampler(x)
        elif self.special_down_sample is not None:
            x = self.special_down_sample(x)

        res = x if self.down_sample is None else self.down_sample(x)
        for layer in self.conv_layers:

            x = layer(x)
            x += res
            x = self.bachNorm(x)
            x = self.relu(x)
            res = x
        print(x.shape)
        return x


class Encoder(nn.Module):
    """
    Encoder class for the autoencoder model.
    Does not include bottleneck layer.

    """
    def __init__(self, input_channels,
                 layers=((16, 32, 2, 1, 1, 5), (32, 64, 2, 1, 5), (64, 64, 1, 1, 1, 5)), global_pooling=True):
        """
        :param input_channels: Number of input channels.
        :param output_channels: Number of output channels.
        :param layers: list of tuples, each tuple is a layer with the following format: (input_channels, output_channels,
        special_down_sample, temp_stride, dilation, num_3d_conv_blocks)
        """
        super(Encoder, self).__init__()

        if len(layers) <= 0:
            raise ValueError("layers must have at least one layer")

        # down sample special dim
        self.down_sample_special = nn.Conv3d(input_channels, layers[0][0], (1, 2, 2), padding=0, stride=(1, 2, 2))
        self.conv_layers = nn.ModuleList()
        for i, layer in enumerate(layers):
            self.conv_layers.append(
                Residual_Block3d(layer[0], layer[1], special_down_sample=layer[2], temp_stride=layer[3],
                                 dilation_size=layer[4], num_blocks=layer[5]))

        self.conv_net = nn.Sequential(self.down_sample_special, *self.conv_layers)

    def forward(self, x):
        x = self.conv_net(x)
        x = x[:, :, -1, :]  # take the last time step of the sequence

        return x


class Decoder(nn.Module):
    """
    Decoder class for the autoencoder model.
    """

    def __init__(self,
                 layers=((64, 64, 1, 1, 1, 5), (32, 64, 2, 1, 5), (16, 32, 2, 1, 1, 5))):
        """
        :param layers: list of tuples, each tuple is a layer with the following format: (input_channels, output_channels,
        special_down_sample, temp_stride, dilation, num_3d_conv_blocks)
        """
        super(Decoder, self).__init__()

        if len(layers) <= 0:
            raise ValueError("layers must have at least one layer")

        self.conv_layers = nn.ModuleList()
        for i, layer in enumerate(layers):
            self.conv_layers.append(
                Residual_Block3d(layer[0], layer[1], special_down_sample=layer[2], temp_stride=layer[3],
                                 dilation_size=layer[4], num_blocks=layer[5], upsamplign=True))

        self.conv_net = nn.Sequential(*self.conv_layers)

        self.final_conv = nn.Conv3d(layers[-1][1], 1, 1, stride=(1, 1, 1), padding=0)

    def forward(self, x):
        x = np.expand_dims(x, axis=2)
        x = torch.tensor(x, dtype=torch.float32)
        x = self.conv_net(x)
        x = self.final_conv(x)
        return x


class AutoEncoder(nn.Module):
    """
    AutoEncoder class for the autoencoder model.
    Does not use streambuffer
    No bottleneck layers
    """
    def __init__(self,
                 layers=((16, 32, 2, 1, 1, 5), (32, 64, 2, 1, 1, 5), (64, 64, 1, 1, 1, 5)), layers_decode=None):
        """
        uses revers layers in decoder if layers_decode is None
        :param layers: list of tuples, each tuple is a layer with the following format: (input_channels, output_channels,
        special_down_sample, temp_stride, dilation, num_3d_conv_blocks)
        :param layers_decode: list of tuples, each tuple is a layer with the following format: (input_channels, output_channels,
        special_down_sample, temp_stride, dilation, num_3d_conv_blocks)
        """
        super(AutoEncoder, self).__init__()
        self.encoder = Encoder(1, layers, global_pooling=False)
        reversed_layers = [(layer[1], layer[0], *layer[2:]) for layer in layers[::-1]] if layers_decode is None else layers_decode

        self.decoder = Decoder(layers=reversed_layers)

    def forward(self, x):
        x = self.encoder(x)
        print("intermidiat shape: ", x.shape)
        x = self.decoder(x)
        return x


def get_n_params(model):
    """
    Get the number of parameters in the model
    :param model: the model
    :return: the number of parameters
    """
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()

    size_all_mb = (param_size + buffer_size) / 1024 ** 2
    print('model size: {:.3f}MB'.format(size_all_mb))
    return param_size


def test():
    """
    test function of the blocks
    """
    layers = ((16, 32, 2, 1, 1, 5), (32, 64, 1, 1, 1, 5), (64, 64, 1, 1, 1, 5))

    model = Blocks3d(1, 1)
    x = torch.randn(64, 1, 4, 28, 28)
    y = model(x)
    print(x.shape)
    print(y.shape)

    print("------------------")
    model = Encoder(28, 1, 7*7*3, layers, global_pooling=False)
    x = torch.randn(64, 1, 4, 28, 28)
    y = model(x)
    print(x.shape)
    print(y.shape)

    print('number of params: {}'.format(get_n_params(model)))

    print("------------------")

    layers = ((64, 32, 1, 1, 1, 5), (32, 16, 1, 1, 1, 5), (16, 8, 1, 1, 1, 5))

    model = Decoder(1, (64, 7, 7), layers=layers)
    x = torch.randn(64, 49*64)
    y = model(x)
    print(x.shape)
    print(y.shape)

    print('number of params: {}'.format(get_n_params(model)))


def test_auto_encoder():
    """
    test function of the auto encoder
    """

    layers = ((16, 32, 2, 1, 1, 5), (32, 32, 2, 1, 1, 5), (32, 32, 2, 1, 1, 5), (32, 64, 1, 1, 1, 3))
    layers2 = ((64, 64, 2, 4, 1, 3), (64, 32, 2, 4, 1, 3),  (32, 16, 2, 4, 1, 3), (16, 8, 2, 2, 1, 3))

    model = AutoEncoder(layers=layers, layers_decode=layers2)
    x = torch.randn(1, 1, 120, 400, 600)
    y = model(x)
    print(x.shape)
    print(y.shape)

    print('number of params: {}'.format(get_n_params(model)))


if __name__ == '__main__':
    test_auto_encoder()
    test()
