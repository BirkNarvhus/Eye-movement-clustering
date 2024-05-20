"""
Usage:
    python plot_loss_util.py <checkpoint_dir>
Description:
    plots the loss of the checkpoints in the checkpoint directory
    This script sorts the checkpoints by index and plots the loss of the checkpoints

"""
import os
import matplotlib.pyplot as plt

import sys
from pathlib import Path
path_root = Path(__file__).parents[2]
sys.path.append(str(path_root))

from util.testingUtils.checkpointsLogging import CheckpointUtil


def load_all_checkpoints(checkpoint_dir):
    """
    Load all the checkpoints in the directory
    :param checkpoint_dir:  the directory of the checkpoints
    :return:  the train and test loss of the checkpoints
    """
    checkpoint_util = CheckpointUtil(checkpoint_dir)
    test_loss = []
    train_loss = []
    file_list = os.listdir(checkpoint_dir)

    for file in file_list:
        if file.startswith("model_best"):
            pass
    for file in file_list:
        if file.endswith(".pt.tar"):
            if file.startswith("model_best"):
                pass
            else:
                index = file.find("_")
                if index == -1:
                    continue
                file_index = file[index + 1:file.find(".")]
                stats = checkpoint_util.load_checkpoint_stats(file)
                test_loss.append((file_index, stats['test_loss']))
                train_loss.append((file_index, stats['loss'].cpu().detach().numpy()))
    test_loss.sort(key=lambda x: x[0])
    train_loss.sort(key=lambda x: x[0])
    return train_loss, test_loss


def plot_loss(checkpoint_dir):
    """
    Plot the loss of the checkpoints
    :param checkpoint_dir: the directory of the checkpoints
    """
    train_loss, test_loss = load_all_checkpoints(checkpoint_dir)
    plt.plot([x[1] for x in train_loss], label="train loss")
    plt.plot([x[1] for x in test_loss], label="test loss")
    plt.legend()
    plt.show()


def main():
    """
    Runs load and plot methods
    """
    if len(sys.argv) < 2:
        print("Usage: python checkpointsLogging.py <checkpoint_dir>")
    checkpoint_dir = sys.argv[1]
    plot_loss(checkpoint_dir)


if __name__ == '__main__':
    main()