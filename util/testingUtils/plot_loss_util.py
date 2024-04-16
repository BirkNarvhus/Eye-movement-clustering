import os

import matplotlib.pyplot as plt

import sys
sys.path.append('/')

from util.testingUtils.checkpointsLogging import CheckpointUtil


def load_all_checkpoints(checkpoint_dir):
    checkpoint_util = CheckpointUtil(checkpoint_dir)
    test_loss = []
    train_loss = []
    file_list = os.listdir(checkpoint_dir)

    for file in file_list:
        if file.startswith("model_best"):
            pass
        test_loss.append(0)
        train_loss.append(0)
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
                test_loss[int(file_index) - 1] = stats['test_loss']
                train_loss[int(file_index) - 1] = stats['loss'].cpu().detach().numpy()
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