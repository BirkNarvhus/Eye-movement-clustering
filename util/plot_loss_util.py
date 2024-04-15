import os

import matplotlib.pyplot as plt

import sys
sys.path.append('C:\\Users\\vizlab_stud\\Documents\\pythonProjects\\eye-movement-classification')

from util.checkpointsLogging import CheckpointUtil


def load_all_checkpoints(checkpoint_dir):
    checkpoint_util = CheckpointUtil(checkpoint_dir)
    test_loss = []
    train_loss = []
    for file in os.listdir(checkpoint_dir):
        if file.endswith(".pt.tar"):
            if file.startswith("model_best"):
                pass
            else:
                stats = checkpoint_util.load_checkpoint_stats(file)
                test_loss.append(stats['test_loss'])
                train_loss.append(stats['loss'].cpu().detach().numpy())
    return train_loss, test_loss


def plot_loss(checkpoint_dir):
    train_loss, test_loss = load_all_checkpoints(checkpoint_dir)
    plt.plot(train_loss, label="train loss")
    plt.plot(test_loss, label="test loss")
    plt.legend()
    plt.show()


def main():
    if len(sys.argv) < 2:
        print("Usage: python checkpointsLogging.py <checkpoint_dir>")
    checkpoint_dir = sys.argv[1]
    plot_loss(checkpoint_dir)


if __name__ == '__main__':
    main()