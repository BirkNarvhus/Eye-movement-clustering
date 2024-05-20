from torch import nn


class PretrainedClassifier(nn.Module):
    def __init__(self, pre_traiend_model, pretrainedOut, num_classes):
        super(PretrainedClassifier, self).__init__()

        self.premodel = pre_traiend_model
        self.num_classes = num_classes

        for p in self.premodel.parameters():
            p.requires_grad = False

        self.last_layer = nn.Linear(pretrainedOut, num_classes)
        self.softmax = nn.Softmax(dim=1)
        #self.model = nn.Sequential(self.premodel, self.last_layer, self.softmax)

    def forward(self, x):
        x = self.premodel(x)
        x = self.last_layer(x)
        x = self.softmax(x)
        return x