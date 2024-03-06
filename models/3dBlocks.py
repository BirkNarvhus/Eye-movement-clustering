import torch
from torch import nn


class Cumulativ_global_pooling(nn.Module):
    def __init__(self):
        super(Cumulativ_global_pooling, self).__init__()

    def forward(self, x):
        return torch.mean(x, dim=3)


class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class Blocks3d(nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size=(3, 3, 3), special_stride=1, temp_stride=1, dilation_size=1):
        super(Blocks3d, self).__init__()
        self.conv = nn.Conv3d(input_channels, output_channels, kernel_size=(1, kernel_size[1], kernel_size[2]),
                              stride=(1, special_stride, special_stride), padding="same" if special_stride == 1 else (0, 1, 1), dilation=dilation_size)

        padding = (kernel_size[0] - 1, 0, 0) * dilation_size
        self.conv2 = nn.Conv3d(output_channels, output_channels, kernel_size=(kernel_size[0], 1, 1), stride=(temp_stride, 1, 1),
                               padding=padding, dilation=dilation_size)
        self.chomp = Chomp1d(padding[0])
        self.net = nn.Sequential(self.conv, self.conv2, self.chomp)

    def forward(self, x):
        x = self.conv(x)
        x = self.conv2(x)
        x = self.chomp(x)

        return x


class Residual_Block3d(nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size=(3, 3, 3), special_down_sample=1, temp_stride=1, dilation_size=1):
        super(Residual_Block3d, self).__init__()
        self.conv1 = Blocks3d(input_channels, output_channels, kernel_size, special_down_sample, temp_stride, dilation_size)
        self.conv2 = Blocks3d(output_channels, output_channels, kernel_size, 1, temp_stride, dilation_size)
        self.relu = nn.ReLU()
        self.net = nn.Sequential(self.conv1, self.relu, self.conv2, self.relu)

        self.down_sample = nn.Conv3d(input_channels, output_channels, 1, stride=(1, special_down_sample,
                                                                                 special_down_sample),
                                     padding=0) if input_channels != output_channels \
            else None

    def forward(self, x):
        res = x if self.down_sample is None else self.down_sample(x)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        return self.relu(x + res)

class Encoder(nn.Module):
    def __init__(self, input_channels, output_channels):
        super(Encoder, self).__init__()

        # down sample special dim
        self.down_sample_special = nn.Conv3d(input_channels, 16, (1, 2, 2), padding=0, stride=(1, 2, 2))

        self.conv1 = Residual_Block3d(16, 32, special_down_sample=2)
        self.conv2 = Residual_Block3d(32, 64, special_down_sample=1)
        self.conv3 = Residual_Block3d(64, 64, special_down_sample=1)
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(int(64 * (28/2/2)), output_channels*7*7)
        self.softmax = nn.Softmax(dim=1)

        self.global_pool = Cumulativ_global_pooling()
        self.relu = nn.ReLU()

        self.net = nn.Sequential(self.down_sample_special, self.conv1, self.conv2, self.conv3,
                                 self.global_pool, self.flatten, self.linear, self.softmax)

    def forward(self, x):
        x = self.down_sample_special(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.global_pool(x)
        x = x[:, :, -1, :]  # take the last time step of the sequence
        print(x.shape)
        x = self.flatten(x)
        x = self.linear(x)
        x = self.softmax(x)
        return x


def get_n_params(model):
    pp=0
    for p in list(model.parameters()):
        nn=1
        for s in list(p.size()):
            nn = nn*s
        pp += nn
    return pp


def test():
    model = Blocks3d(1, 1)
    x = torch.randn(64, 1, 4, 28, 28)
    y = model(x)
    print(x.shape)
    print(y.shape)

    print("------------------")
    model = Encoder(1, 1)
    x = torch.randn(64, 1, 4, 28, 28)
    y = model(x)
    print(x.shape)
    print(y.shape)

    print(get_n_params(model))


if __name__ == '__main__':
    test()