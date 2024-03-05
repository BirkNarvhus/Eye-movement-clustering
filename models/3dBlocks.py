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
    def __init__(self, input_channels, output_channels, kernel_size=(3, 3, 3), temp_stride=1, dilation_size=1):
        super(Blocks3d, self).__init__()
        self.conv = nn.Conv3d(input_channels, output_channels, kernel_size=(1, kernel_size[1], kernel_size[2]),
                              stride=1, padding=0)

        padding = (kernel_size[0] - 1, 0, 0) * dilation_size
        self.conv2 = nn.Conv3d(output_channels, output_channels, kernel_size=(kernel_size[0], 1, 1), stride=temp_stride,
                               padding=padding, dilation=dilation_size)
        self.chomp = Chomp1d(padding[0])
        self.net = nn.Sequential(self.conv, self.conv2, self.chomp)

    def forward(self, x):
        return self.net(x)


class Encoder(nn.Module):
    def __init__(self, input_channels, output_channels):
        super(Encoder, self).__init__()
        self.conv1 = Blocks3d(input_channels, 32)
        self.conv2 = Blocks3d(32, 64)
        self.conv3 = Blocks3d(64, 64)
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(64 * 22, output_channels*7*7)
        self.softmax = nn.Softmax(dim=1)

        self.global_pool = Cumulativ_global_pooling()
        self.relu = nn.ReLU()

        self.net = nn.Sequential(self.conv1, self.conv2, self.conv3,
                                 self.global_pool, self.flatten, self.linear, self.softmax)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.global_pool(x)
        x = x[:, :, -1, :]  # take the last time step of the sequence
        x = self.flatten(x)
        x = self.linear(x)
        x = self.softmax(x)
        return x



def test():
    model = Blocks3d(1, 1)
    x = torch.randn(64, 1, 4, 28, 28)
    y = model(x)
    print(x.shape)
    print(y.shape)


    model = Encoder(1, 1)
    x = torch.randn(64, 1, 4, 28, 28)
    y = model(x)
    print(x.shape)
    print(y.shape)




if __name__ == '__main__':
    test()