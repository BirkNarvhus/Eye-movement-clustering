"""
Usage:
    python ./sim_clr_ds_finetuner.py <checkpoint_path>
Description:
    Adds a classifier to the clr model
    Trains the model on the mnist dataset
    Tests the model acc on the mnist dataset
"""


import os
import sys

import numpy as np
from torch import nn
import torch
from models.pre_tests.Pretrained_classifier import finetuned_encoder_classifier
from models.pre_tests.simpleCnn import SimpleCnn
from util.dataUtils.data import data_generator


device = "cpu"

tr_ep_loss = []
tr_ep_acc = []

val_ep_loss = []
val_ep_acc = []

min_val_loss = 100.0

EPOCHS = 10
num_cl = 10
batch_size = 128
root = '../data/mnist'


def main():
    """
    Function for testing and training the model

    """
    global min_val_loss
    train_loader, test_loader = data_generator(root, batch_size, clr=False)

    for epoch in range(20):

        print("=============== Epoch : %3d ===============" % (epoch + 1))

        loss_sublist = np.array([])
        acc_sublist = np.array([])

        # iter_num = 0
        dsmodel.train()

        dsoptimizer.zero_grad()

        for x, y in train_loader:
            x = x.to(device=device, dtype=torch.float)
            y = y.to(device=device)

            z = dsmodel(x)

            dsoptimizer.zero_grad()

            tr_loss = loss_fn(z, y)
            tr_loss.backward()

            preds = z.detach()

            dsoptimizer.step()

            loss_sublist = np.append(loss_sublist, tr_loss.cpu().data)
            acc_sublist = np.append(acc_sublist,
                                    np.array(np.argmax(preds, axis=1) == y.cpu().data.view(-1)).astype('int'), axis=0)

            # iter_num+=1

        print('Loss avg ', np.mean(loss_sublist))
        print('Acc avg ', np.mean(acc_sublist))

        dsmodel.eval()

        loss_sublist = np.array([])
        acc_sublist = np.array([])

        with torch.no_grad():
            for x, y in test_loader:
                x = x.to(device=device, dtype=torch.float)
                y = y.to(device=device)
                z = dsmodel(x)

                val_loss = loss_fn(z, y)

                preds = torch.exp(z.cpu().data) / torch.sum(torch.exp(z.cpu().data))

                loss_sublist = np.append(loss_sublist, val_loss.cpu().data)
                acc_sublist = np.append(acc_sublist,
                                        np.array(np.argmax(preds, axis=1) == y.cpu().data.view(-1)).astype('int'),
                                        axis=0)

        print('Val loss avg ', np.mean(loss_sublist))
        print('Val acc avg ', np.mean(acc_sublist))

        lr_scheduler.step()

    pytorch_total_params = sum(p.numel() for p in dsmodel.parameters())
    print('Number of params: {}'.format(pytorch_total_params))
    out = os.path.join('../content/saved_models/', 'simclr_classifer.pth')

    torch.save({'model_state_dict': dsmodel.state_dict(),
                'optimizer_state_dict': dsoptimizer.state_dict()}, out)


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python accuracy_tester.py <checkpoint_path>")
        sys.exit(1)

    modelname = sys.argv[1]
    if not os.path.exists(modelname):
        print("The file does not exist")
        sys.exit(1)

    dsmodel = SimpleCnn(1, 64)
    dsmodel.load_state_dict(torch.load(modelname)["model_state_dict"])
    dsmodel = finetuned_encoder_classifier(dsmodel, 64, 10)
    dsmodel.eval()

    dsoptimizer = torch.optim.SGD([params for params in dsmodel.parameters() if params.requires_grad], lr=0.01,
                                  momentum=0.9)

    lr_scheduler = torch.optim.lr_scheduler.StepLR(dsoptimizer, step_size=1, gamma=0.98, last_epoch=-1)

    loss_fn = nn.CrossEntropyLoss()
    main()
