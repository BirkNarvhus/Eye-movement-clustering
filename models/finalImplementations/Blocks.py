"""
This file contains the implementation of the blocks used in the final implementation of the model.
"""

from torch import nn
import torch


class Cumulativ_global_pooling(nn.Module):
    """
    This class is used to apply a global pooling on the input tensor.
    """
    def __init__(self):
        super(Cumulativ_global_pooling, self).__init__()

    def forward(self, x):
        return torch.mean(x, dim=2)


class Chomp1d(nn.Module):
    """
    This class is used to remove the padding from the temporal convolution.
    """
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class TempConvBlock(nn.Module):
    """
    This class is used to apply a temporal convolution on the input tensor.
    If casual convolution is used, then apply 2+1D convolution.
    """
    def __init__(self, input_channels, output_channels, kernel_size=(3, 3, 3), stride=1, padding=1, dilation_size=1, causel=True):
        """
        :param input_channels: Number of input channels.
        :param output_channels: Number of output channels.
        :param kernel_size: Size of the kernel.
        :param stride:  Stride of the convolution.
        :param padding: Padding of the convolution.
        :param dilation_size: Dilation of the convolution.
        :param causel: If True, apply 2+1D convolution with casual conv in temp dim.
        """
        super(TempConvBlock, self).__init__()

        zero_padding = (0 if causel else kernel_size[0] // 2, kernel_size[1] // 2, kernel_size[2] // 2)

        if causel:
            self.conv = nn.Conv3d(input_channels, output_channels, kernel_size=(1, kernel_size[1], kernel_size[2]),
                                  stride=stride, padding=zero_padding)

            padding_to_chomp = (kernel_size[0] - 1) * dilation_size

            if kernel_size[0] != 1:
                self.chomp = Chomp1d(padding_to_chomp)
                self.temp_Conv = nn.Conv3d(output_channels, output_channels, kernel_size=(kernel_size[0], 1, 1),
                                           stride=(stride, 1, 1),
                                           padding=(padding_to_chomp, 0, 0), dilation=dilation_size)
                self.net = nn.Sequential(self.conv, self.chomp, self.temp_Conv)
            else:
                self.net = nn.Sequential(self.conv)
        else:
            self.conv = nn.Conv3d(input_channels, output_channels, kernel_size=kernel_size, stride=stride,
                                  padding=zero_padding, dilation=dilation_size)
            self.net = nn.Sequential(self.conv)

    def forward(self, x):
        return self.net(x)


class Projection(nn.Module):
    """
    This class is used to apply a projection on the input tensor.
    changes num channels with 1x1x1 convolution.
    """
    def __init__(self, channels):
        super(Projection, self).__init__()
        self.projection = nn.Conv3d(channels[0], channels[1], kernel_size=(1, 1, 1), stride=1, padding=0)

    def forward(self, x):
        return self.projection(x)


class ResBlock(nn.Module):
    """
    This class is used to apply a residual block on the input tensor.
    uses TempConvBlock and Projection.
    used leakyRelu, BatchNorm3d and Dropout.
    """
    def __init__(self, channels=((64, 128), (128, 64)), kernels_size=((3, 3, 3), (3, 3, 3)), stride=1, padding=1,
                 dilation=1,  causel=True):
        """
        :param channels: List of channels for each layer.
        :param kernels_size: List of kernel sizes for each layer.
        :param stride: Stride of the convolution.
        :param padding: Padding of the convolution.
        :param dilation: Dilation of the convolution.
        :param causel: If True, apply 2+1D convolution with casual conv in temp dim.
        """
        super(ResBlock, self).__init__()

        modList = nn.ModuleList()
        self.relu = nn.LeakyReLU()

        for i in range(len(kernels_size)):
            if i != 0:
                if channels[i][0] != channels[i - 1][1]:
                    modList.append(Projection((channels[i - 1][1], channels[i][0])))

            modList.append(TempConvBlock(channels[i][0], channels[i][1], kernel_size=kernels_size[i], stride=stride,
                                         padding=padding, causel=causel, dilation_size=dilation))
            modList.append(self.relu)
            modList.append(nn.BatchNorm3d(channels[i][1]))
            modList.append(nn.Dropout(0.2))

        self.projection = nn.Conv3d(channels[0][0], channels[-1][1], kernel_size=(1, 1, 1), stride=stride,
                                    padding=0) if channels[0][0] != channels[-1][1] else None
        self.net = nn.Sequential(*modList)

    def forward(self, x):
        res = x
        x = self.net(x)
        if self.projection:
            res = self.projection(res)

        return self.relu(x + res)


class MultiResLayer(nn.Module):
    """
    This class is used to apply multiple residual blocks on the input tensor.
    uses ResBlock in sequential format
    """
    def __init__(self, channels, kernels_size=(((3, 3, 3), (3, 3, 3), (3, 3, 3)), ((3, 3, 3), (3, 3, 3), (3, 3, 3))),
                 stride=1, padding=1, dilation=1, causel=True):
        super(MultiResLayer, self).__init__()

        if len(channels) != len(kernels_size):
            raise ValueError("Channels and kernels_size must have the same length")

        self.res_blocks = nn.ModuleList()

        for i in range(len(channels)):
            if len(channels[i]) != len(kernels_size[i]):
                raise ValueError("Channels and kernels_size must have the same length")
            if i != 0:
                if channels[i - 1][-1][-1] != channels[i][0][0]:
                    self.res_blocks.append(Projection((channels[i - 1][-1][-1], channels[i][0][0])))
            self.res_blocks.append(ResBlock(channels[i], kernels_size[i], stride, padding, dilation, causel))

        self.net = nn.Sequential(*self.res_blocks)

    def forward(self, x):
        return self.net(x)


class DownsampleLayer(nn.Module):
    """
    This class is used to apply a downsample layer on the input tensor.
    uses AvgPool3d and MultiResLayer.
    This will be the full block in arc file
    """
    def __init__(self, channels, kernels_size=((3, 3, 3), (3, 3, 3), (3, 3, 3)), stride=1, padding=1):
        super(DownsampleLayer, self).__init__()
        self.downsample = nn.AvgPool3d((1, 2, 2), (1, 2, 2))

        self.resBlockLayers = MultiResLayer(channels, kernels_size, stride, padding)
        self.net = nn.Sequential(self.downsample, self.resBlockLayers)

    def forward(self, x):
        return self.net(x)


class TransposedConvUpsampling(nn.Module):
    """
    This class is used to apply transposed conv upsampling on the input tensor.
    """
    def __init__(self, in_channels, kernel_size=(2, 1, 1), stride=1):
        super(TransposedConvUpsampling, self).__init__()
        self.conv = nn.ConvTranspose3d(in_channels, in_channels, kernel_size=kernel_size, stride=stride)

    def forward(self, x):
        return self.conv(x)


class UpsampleLayer(nn.Module):
    """
    simular to DownsampleLayer but uses Upsample instead of AvgPool3d
    uses 3dUpsample with interpolation and MultiResLayer.
    """
    def __init__(self, channels, kernels_size=((3, 3, 3), (3, 3, 3), (3, 3, 3)), stride=1, padding=1,
                 upscale_kernel=(2, 2, 2), causel=False):
        super(UpsampleLayer, self).__init__()
        self.upsample = nn.Upsample(scale_factor=upscale_kernel)
        self.resBlockLayers = MultiResLayer(channels, kernels_size, stride, padding, causel=causel)

        self.net = nn.Sequential(self.upsample, self.resBlockLayers)

    def forward(self, x):
        return self.net(x)


def test():
    """
    test function of the blocks.
    """
    model = ResBlock()
    x = torch.randn(5, 64, 10, 32, 32)
    print(model(x).shape)

    print("==="*10)
    channels = (((64, 128), (128, 64), (64, 32)),
                ((32, 64), (64, 64), (64, 32)))
    kernels = (((3, 3, 3), (3, 3, 3), (3, 3, 3)),
               ((3, 3, 3), (3, 3, 3), (3, 3, 3)))
    model = UpsampleLayer(channels, kernels)
    x = torch.randn(5, 64, 10, 32, 32)
    print(model(x).shape)

    print("==="*10)
    model = DownsampleLayer(channels, kernels)
    x = torch.randn(5, 64, 10, 32, 32)
    print(model(x).shape)


if __name__ == '__main__':
    test()