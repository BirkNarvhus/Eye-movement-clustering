import torch.nn


class DiceLoss(torch.nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, x, y):
        return 1 - (2 * torch.sum(x * y) + 1) / (torch.sum(x) + torch.sum(y) + 1)


class BceDiceLoss(torch.nn.Module):
    def __init__(self):
        super(BceDiceLoss, self).__init__()
        self.bce = torch.nn.BCELoss()
        self.dice = DiceLoss()

    def forward(self, x, y):
        x, y = torch.nn.functional.sigmoid(x), torch.nn.functional.sigmoid(y)
        return self.bce(x, y) + self.dice(x, y)
