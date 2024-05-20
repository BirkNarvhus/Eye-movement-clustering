import os
import time

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

modelname = "simclr_model_3.pth"

dsmodel = SimpleCnn(1, 64)
dsmodel.load_state_dict(torch.load("../content/saved_models/" + modelname)["model_state_dict"])
dsmodel = finetuned_encoder_classifier(dsmodel, 64, 10)
dsmodel.eval()

dsoptimizer = torch.optim.SGD([params for params in dsmodel.parameters() if params.requires_grad],lr = 0.01, momentum = 0.9)

lr_scheduler = torch.optim.lr_scheduler.StepLR(dsoptimizer, step_size=1, gamma=0.98, last_epoch=-1)

loss_fn = nn.CrossEntropyLoss()


def main():
    global min_val_loss
    train_loader, test_loader = data_generator(root, batch_size, clr=False)

    for epoch in range(20):

        stime = time.time()
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

            tr_loss = loss_fn(z, y)  # /accumulation_steps #y.to(dtype=torch.float), z)
            tr_loss.backward()

            preds = z.detach()

            # if (iter_num+1)%accumulation_steps==0:
            dsoptimizer.step()
            # dsoptimizer.zero_grad()

            loss_sublist = np.append(loss_sublist, tr_loss.cpu().data)
            acc_sublist = np.append(acc_sublist,
                                    np.array(np.argmax(preds, axis=1) == y.cpu().data.view(-1)).astype('int'), axis=0)

            # iter_num+=1

        print('ESTIMATING TRAINING METRICS.............')

        print('TRAINING BINARY CROSSENTROPY LOSS: ', np.mean(loss_sublist))
        print('TRAINING BINARY ACCURACY: ', np.mean(acc_sublist))
        # print('TRAINING AUC SCORE: ',roc_auc_score(gt,preds))

        tr_ep_loss.append(np.mean(loss_sublist))
        tr_ep_acc.append(np.mean(acc_sublist))

        # tr_ep_auc.append(roc_auc_score(gt, preds))

        print('ESTIMATING VALIDATION METRICS.............')

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

        print('VALIDATION BINARY CROSSENTROPY LOSS: ', np.mean(loss_sublist))
        print('VALIDATION BINARY ACCURACY: ', np.mean(acc_sublist))
        # print('VALIDATION AUC SCORE: ',roc_auc_score(gt, preds))

        val_ep_loss.append(np.mean(loss_sublist))
        val_ep_acc.append(np.mean(acc_sublist))

        # val_ep_auc.append(roc_auc_score(gt, preds))

        lr_scheduler.step()

        print("Time Taken : %.2f minutes" % ((time.time() - stime) / 60.0))

    pytorch_total_params = sum(p.numel() for p in dsmodel.parameters())
    print('Number of params: {}'.format(pytorch_total_params))
    out = os.path.join('../content/saved_models/', 'simclr_classifer.pth')

    torch.save({'model_state_dict': dsmodel.state_dict(),
                'optimizer_state_dict': dsoptimizer.state_dict()}, out)


if __name__ == '__main__':
    main()
