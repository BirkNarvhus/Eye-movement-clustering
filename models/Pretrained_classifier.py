import torch
from torch import nn

from models.autoEncoder import Encoder


class finetuned_encoder_classifier(nn.Module):
    def __init__(self, encoder, encoder_out, classes):
        super(finetuned_encoder_classifier, self).__init__()
        self.encoder = encoder

        for p in self.encoder.parameters():
            p.requires_grad = False

        self.flatten = nn.Flatten()
        self.fc = nn.Linear(encoder_out*7*7, classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.encoder(x)
        x = self.flatten(x)
        x = self.fc(x)
        x = self.softmax(x)
        return x


def test():
    encoder = Encoder(1, 1)
    model = finetuned_encoder_classifier(encoder, 1*7*7, 10)
    x = torch.randn(64, 1, 28, 28)
    y = model(x)
    print(x.shape)
    print(y.shape)


if __name__ == '__main__':
    test()