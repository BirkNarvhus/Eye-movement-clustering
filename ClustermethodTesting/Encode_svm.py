import numpy as np
import torch
from tqdm import tqdm

from models.autoEncoder import AutoEncoder
from util.data import data_generator
from matplotlib import pyplot as plt
from sklearn.svm import SVC
import pandas as pd

root = '../data/mnist'

max_batches = 0
num_feats = 3*7*7
batch_size = 128

device = "cpu"
if torch.cuda.is_available():
    device = "cuda:0"

def main():
    auto_encoder = AutoEncoder(1, 3, 1)
    auto_encoder.load_state_dict(torch.load("../content/saved_models/auto_encoder.pth"))
    encoder = auto_encoder.encoder
    encoder.eval()
    train_loader, test_loader = data_generator(root, batch_size, clr=False, shuffle=True)

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

    print("Validation accuracy for svm model: " + str((preds == targets.squeeze()).sum()/len(targets.squeeze())))


if __name__ == '__main__':
    main()
