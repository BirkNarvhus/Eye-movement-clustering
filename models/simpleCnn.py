import torch
import torch.nn as nn


class cnnBlock(nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size, stride, padding):
        super(cnnBlock, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, input_channels*2, kernel_size, stride, padding)

        self.conv2 = nn.Conv2d(input_channels*2, output_channels, kernel_size, stride, padding)
        self.maxpool = nn.MaxPool2d(2)
        self.net = nn.Sequential(self.conv1, self.conv2, self.maxpool)

    def forward(self, x):
        x = self.net(x)
        return x


class SimpleCnn(nn.Module):
    def __init__(self, input_channels, num_classes):
        super(SimpleCnn, self).__init__()
        self.conv1 = cnnBlock(input_channels, input_channels*2, kernel_size=3, stride=1, padding=1)
        self.conv2 = cnnBlock(input_channels*2, input_channels*4, kernel_size=3, stride=1, padding=1)
        self.flatten = nn.Flatten()
        self.bcNorm = nn.BatchNorm2d(input_channels*4)
        self.fc1 = nn.Linear(input_channels*4*7*7, 128*2)
        self.fc2 = nn.Linear(128*2, num_classes)
        self.maxpool = nn.MaxPool2d(2)

        self.conv = nn.Sequential(self.conv1, self.conv2, self.bcNorm, self.flatten, self.fc1, nn.ReLU(),
                                  self.fc2)
        #self.lin = nn.Sequential(self.fc1, self.fc2, self.softmax)

    def forward(self, x):
        x = self.conv(x)

        #x = x.view(-1, 64*7*7)
        #x = self.lin(x)
        return x

    def weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)


def test():
    model = SimpleCnn(1, 10)
    x = torch.randn(64, 1, 28, 28)
    y = model(x)
    print(x.shape)
    print(y.shape)


if __name__ == '__main__':
    test()