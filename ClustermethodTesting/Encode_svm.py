"""
Usage:
    python ./Encode_svm.py <checkpoint_path> <optional: model_type>
Description:
    Gets the encoder from the checkpoint and trains a svm model on the encoded data
    Uses targets from the data loader to train the svm model
    Model_type can be autoencoder or simclr

"""
import os
import sys

import numpy as np
import torch
from tqdm import tqdm

from models.pre_tests.autoEncoder import AutoEncoder
from models.pre_tests.simpleCnn import SimpleCnn
from util.dataUtils.data import data_generator
from sklearn.svm import SVC
import pandas as pd

root = '../data/mnist'

max_batches = 0
num_feats = 64
batch_size = 128

device = "cpu"
if torch.cuda.is_available():
    device = "cuda:0"


def train_svm(encoder, train_loader, test_loader, num_feats, batch_size):
    """
    Trains a svm model on the encoded data
    :param encoder: the encoder model
    :param train_loader: the train loader
    :param test_loader: the test loader
    :param num_feats: number of features
    :param batch_size: batch size
    """
    feats = np.array([]).reshape((0, num_feats))
    targets = np.array([]).reshape((0, 1))
    encoder.eval()
    with torch.no_grad():
        for idx, (x1, target) in tqdm(enumerate(train_loader), position=0, leave=True, desc="training model"):
            if 0 < max_batches < idx:
                break
            x1 = x1.to(device=device, dtype=torch.float)
            out = encoder(x1)
            out = out.cpu().data.numpy()
            if len(out) != batch_size:
                break
            out = out.reshape((batch_size, num_feats))
            feats = np.append(feats, out, axis=0)
            target = target.cpu().data.numpy().reshape((-1, 1))
            targets = np.append(targets, target, axis=0)

    clf = SVC()
    print("fitting svm model")
    clf.fit(feats, targets.squeeze())
    preds = np.array([]).reshape((0, 1))
    targets = np.array([]).reshape((0, 1))
    for idx, (x1, target) in tqdm(enumerate(test_loader), position=0, leave=True, desc="testing model"):
        if 0 < max_batches < idx:
            break

        out = encoder(x1)
        if len(out) != batch_size:
            break
        target = target.cpu().data.numpy().reshape((-1, 1))
        targets = np.append(targets, target, axis=0)
        x1 = x1.to(device=device, dtype=torch.float)

        preds = np.append(preds, clf.predict(out.cpu().data.numpy().reshape((batch_size, num_feats))))

    preds = pd.Series(preds, name="Label")

    print("Validation accuracy for svm model: " + str((preds == targets.squeeze()).sum() / len(targets.squeeze())))


def main():
    """
    Main function for parsing model file and mode
    """

    if len(sys.argv) < 2:
        print("Usage: python test_model.py <model_file> <optional: mode>")
        sys.exit(1)

    model_file = sys.argv[1]

    if not os.path.exists(model_file):
        print("The file does not exist")
        sys.exit(1)
    model_type = "simclr"
    if len(sys.argv) == 3:
        model_type = sys.argv[2]

    if model_type == "autoencoder":
        auto_encoder = AutoEncoder(1, 3, 1)
        auto_encoder.load_state_dict(torch.load("../content/saved_models/auto_encoder.pth"))
        encoder = auto_encoder.encoder
    else:
        encoder = SimpleCnn(1, 64)
        encoder.load_state_dict(torch.load("../content/saved_models/simclr_model_100.pth")['model_state_dict'])

    encoder.eval()
    train_loader, test_loader = data_generator(root, batch_size, clr=False, shuffle=True)

    train_svm(encoder, train_loader, test_loader, num_feats, batch_size)


if __name__ == '__main__':
    main()
