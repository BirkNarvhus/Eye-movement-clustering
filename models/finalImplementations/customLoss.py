"""
This file contains the implementation of custom loss functions.
"""

import torch.nn


class DiceLoss(torch.nn.Module):
    """
    This class is used to calculate the Dice loss.
    """
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, x, y):
        return 1 - (2 * torch.sum(x * y) + 1) / (torch.sum(x) + torch.sum(y) + 1)


class BceDiceLoss(torch.nn.Module):
    """
    This class is used to calculate the BCE Dice loss.
    """
    def __init__(self):
        super(BceDiceLoss, self).__init__()
        self.bce = torch.nn.BCELoss()
        self.dice = DiceLoss()

    def forward(self, x, y):
        x, y = torch.nn.functional.sigmoid(x), torch.nn.functional.sigmoid(y)
        return self.bce(x, y) + self.dice(x, y)


class DiceCrossEntrepy(torch.nn.Module):
    """
    This class is used to calculate the Dice Cross Entropy loss.
    """
    def __init__(self):
        super(DiceCrossEntrepy, self).__init__()
        self.dice = DiceLoss()
        self.cross_entropy = torch.nn.CrossEntropyLoss()

    def forward(self, x, y):
        x, y = torch.nn.functional.sigmoid(x), torch.nn.functional.sigmoid(y)
        return self.dice(x, y) + self.cross_entropy(x, y)


class SmoothenGradiantWithHubertLoss(torch.nn.Module):
    """
    This class is used to calculate the Smoothen Gradiant with Hubert loss.
    """

    def __init__(self, loss, beta=0.1):
        super(SmoothenGradiantWithHubertLoss, self).__init__()
        self.loss = loss
        self.beta = beta

    def forward(self, x, y):
        loss = self.loss(x, y)
        x = x.view(-1)
        y = y.view(-1)
        diff = torch.abs(x - y)
        loss = torch.mean(loss + self.beta * diff)
        return loss