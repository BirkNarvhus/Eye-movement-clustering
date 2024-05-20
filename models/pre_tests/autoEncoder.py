"""
2d autoencoder model
Used in the first test iterations of the models.
"""

import torch
from torch import nn


class Encoder(nn.Module):
    """
    Encoder class for the autoencoder model.
    Simple network architecture with 3 convolutional layers.
    one linear layer to the output.
    """
    def __init__(self, input_channels, output_channels):
        """
        :param input_channels: Number of input channels.
        :param output_channels: Number of output channels.
        """
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, stride=1, padding=0)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=0)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0)
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(64*3*3, output_channels*7*7)
        self.sigmoid = nn.Sigmoid()

        self.maxPool = nn.MaxPool2d(2)
        self.relu = nn.ReLU()

        self.net = nn.Sequential(self.conv1, self.relu, self.maxPool, self.conv2, self.relu, self.maxPool, self.conv3,
                                 self.relu, self.flatten, self.linear, self.sigmoid)

    def forward(self, x):
        return self.net(x)


class Decoder(nn.Module):
    """
    Decoder class for the autoencoder model.
    Simple network architecture with transposed convolution
    and 2d convolution
    """
    def __init__(self, input_channels, output_channels):
        """
        :param input_channels: Number of input channels.
        :param output_channels: Number of output channels.
        """
        super(Decoder, self).__init__()
        self.input = input_channels
        self.conv1 = nn.ConvTranspose2d(input_channels, 32, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.conv2 = nn.ConvTranspose2d(32, 64, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.conv3 = nn.ConvTranspose2d(64, 64, kernel_size=3, stride=1)
        self.conv4 = nn.Conv2d(64, output_channels, kernel_size=3, stride=1)
        self.bachNorm1 = nn.BatchNorm2d(32)
        self.bachNorm2 = nn.BatchNorm2d(64)

        self.net = nn.Sequential(self.conv1, self.relu, self.bachNorm1, self.conv2, self.relu, self.bachNorm2,
                                 self.conv3, self.relu, self.conv4, self.sigmoid)

    def forward(self, x):
        x = x.view(x.shape[0], self.input, 7, 7)
        x = self.net(x)
        return x


class AutoEncoder(nn.Module):
    """
    AutoEncoder class for the autoencoder model.
    Combines the Encoder and Decoder classes.
    Uses a suboptimal way to turn off the decoder.
    This is done by to simplify loading the old models.
    """
    def __init__(self, input_channels, hidden_channels, output_channels, turn_off_decoder=False):
        """
        :param input_channels: Number of input channels.
        :param hidden_channels: Number of hidden channels.
        :param output_channels: Number of output channels.
        :param turn_off_decoder: If True, the decoder is turned off.
        """
        super(AutoEncoder, self).__init__()
        self.encoder = Encoder(input_channels, hidden_channels)
        self.decoder = Decoder(hidden_channels, output_channels)
        self.turn_off_decoder = turn_off_decoder

    def forward(self, x):
        x = self.encoder(x)
        if not self.turn_off_decoder:
            x = self.decoder(x)
        return x


def test():
    """
    Test the model
    """
    model = Encoder(1, 1)
    x = torch.randn(64, 1, 28, 28)
    y = model(x)
    print(x.shape)
    print(y.shape)

    model = Decoder(1, 1)
    z = model(y)
    print(z.shape)

    model = AutoEncoder(1, 32, 1)
    x = torch.randn(64, 1, 28, 28)
    y = model(x)
    print(x.shape)
    print(y.shape)


if __name__ == '__main__':
    test()