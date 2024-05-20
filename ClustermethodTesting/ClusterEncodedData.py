import numpy as np
import scipy
import torch
from tqdm import tqdm

from models.pre_tests.autoEncoder import AutoEncoder
from util.dataUtils.data import data_generator

from util.plot_tsne import PlotUtil

root = '../data/mnist'

max_batches = 0

device = "cpu"
if torch.cuda.is_available():
    device = "cuda:0"


def plot_features(model, num_classes, num_feats, batch_size, test_loader):
    feats = np.array([]).reshape((0, num_feats))
    targets = np.array([]).reshape((0, 1))
    model.eval()
    with torch.no_grad():
        for idx, (x1, target) in tqdm(enumerate(test_loader), position=0, leave=True, desc="Eval model"):
            if 0 < max_batches < idx:
                break
            x1 = x1.to(device=device, dtype=torch.float)
            out = model(x1)
            if len(out) != batch_size:
                break
            out = out.cpu().data.numpy()
            out = out.reshape((batch_size, num_feats))
            target = target.cpu().data.numpy().reshape((-1, 1))
            targets = np.append(targets, target, axis=0)
            feats = np.append(feats, out, axis=0)
    print("Eval done starting t-SNE")

    plt_util = PlotUtil(feats, "test encoder", mode="pcak", show=True, dims=2, kmeans_after=True)
    plt_util.plot_tsne(targets=targets.squeeze())

    preds = plt_util.downstream(feats)

    print(scipy.stats.ks_2samp(preds, targets.squeeze(), alternative='two-sided', mode='auto'))


def main():
    model = AutoEncoder(1, 3, 1, turn_off_decoder=True).to(device=device)
    model.load_state_dict(torch.load("../content/saved_models/auto_encoder.pth"))
    #model = SimpleCnn(1, 64)
    #model.load_state_dict(torch.load("../content/saved_models/simclr_model_100.pth")['model_state_dict'])
    model = model.encoder
    model.eval()

    _, test_loader = data_generator(root, 128, clr=False, shuffle=True)

    plot_features(model, 10, 3*7*7, 128, test_loader)


if __name__ == '__main__':
    main()