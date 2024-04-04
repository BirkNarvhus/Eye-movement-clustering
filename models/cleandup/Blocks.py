from torch import nn
import torch


class Cumulativ_global_pooling(nn.Module):
    def __init__(self):
        super(Cumulativ_global_pooling, self).__init__()

    def forward(self, x):
        return torch.mean(x, dim=2)


class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class TempConvBlock(nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size=(3, 3, 3), stride=1, padding=1, dilation_size=1):
        super(TempConvBlock, self).__init__()

        zero_padding = (0, kernel_size[1] // 2, kernel_size[2] // 2)
        self.conv = nn.Conv3d(input_channels, output_channels, kernel_size=(1, kernel_size[1], kernel_size[2]),
                              stride=stride, padding=zero_padding)

        padding_to_chomp = (kernel_size[0] - 1) * dilation_size

        if kernel_size[0] != 1:
            self.chomp = Chomp1d(padding_to_chomp)
            self.temp_Conv = nn.Conv3d(output_channels, output_channels, kernel_size=(kernel_size[0], 1, 1),
                                       stride=(stride, 1, 1),
                                       padding=(padding_to_chomp, 0, 0), dilation=dilation_size)
            self.net = nn.Sequential(self.conv, self.temp_Conv, self.chomp)
        else:
            self.net = nn.Sequential(self.conv)

    def forward(self, x):
        return self.net(x)


class Projection(nn.Module):
    def __init__(self, channels):
        super(Projection, self).__init__()
        self.projection = nn.Conv3d(channels[0], channels[1], kernel_size=(1, 1, 1), stride=1, padding=0)

    def forward(self, x):
        return self.projection(x)


class ResBlock(nn.Module):
    def __init__(self, channels=((64, 128), (128, 64)), kernels_size=((3, 3, 3), (3, 3, 3)), stride=1, padding=1):
        super(ResBlock, self).__init__()

        modList = nn.ModuleList()
        self.relu = nn.LeakyReLU()

        for i in range(len(kernels_size)):
            if i != 0:
                if channels[i][0] != channels[i - 1][1]:
                    modList.append(Projection((channels[i - 1][1], channels[i][0])))

            modList.append(TempConvBlock(channels[i][0], channels[i][1], kernel_size=kernels_size[i], stride=stride,
                                         padding=padding))
            modList.append(self.relu)
            modList.append(nn.BatchNorm3d(channels[i][1]))

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
    def __init__(self, channels, kernels_size=(((3, 3, 3), (3, 3, 3), (3, 3, 3)), ((3, 3, 3), (3, 3, 3), (3, 3, 3))),
                 stride=1, padding=1):
        super(MultiResLayer, self).__init__()

        if len(channels) != len(kernels_size):
            raise ValueError("Channels and kernels_size must have the same length")

        self.res_blocks = nn.ModuleList()

        for i in range(len(channels)):
            if len(channels[i]) != len(kernels_size[i]):
                print(channels[i], kernels_size[i])
                raise ValueError("Channels and kernels_size must have the same length")
            if i != 0:
                if channels[i - 1][-1][-1] != channels[i][0][0]:
                    print(channels[i - 1][-1][-1], channels[i][0][0])
                    self.res_blocks.append(Projection((channels[i - 1][-1][-1], channels[i][0][0])))
            self.res_blocks.append(ResBlock(channels[i], kernels_size[i], stride, padding))

        self.net = nn.Sequential(*self.res_blocks)

    def forward(self, x):
        return self.net(x)


class DownsampleLayer(nn.Module):
    def __init__(self, channels, kernels_size=((3, 3, 3), (3, 3, 3), (3, 3, 3)), stride=1, padding=1):
        super(DownsampleLayer, self).__init__()
        self.downsample = nn.AvgPool3d((1, 2, 2), (1, 2, 2))

        self.resBlockLayers = MultiResLayer(channels, kernels_size, stride, padding)
        self.net = nn.Sequential(self.downsample, self.resBlockLayers)

    def forward(self, x):
        return self.net(x)


class UpsampleLayer(nn.Module):
    def __init__(self, channels, kernels_size=((3, 3, 3), (3, 3, 3), (3, 3, 3)), stride=1, padding=1):
        super(UpsampleLayer, self).__init__()
        self.upsample = nn.Upsample(scale_factor=(1, 2, 2), mode='trilinear')

        self.resBlockLayers = MultiResLayer(channels, kernels_size, stride, padding)

        self.net = nn.Sequential(self.upsample, self.resBlockLayers)

    def forward(self, x):
        return self.net(x)


def test():
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