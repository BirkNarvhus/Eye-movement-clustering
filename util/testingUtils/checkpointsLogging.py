"""
Usage:
    python ./simClrTrainer.py <checkpoint_dir> <checkpoint_name>

Description:
    Returns the stats of the checkpoint

This file contains the CheckpointUtil class for saving and loading checkpoints.
"""

import os
import shutil
import sys

import torch


class CheckpointUtil:
    """
    Checkpoint utility class for saving and loading checkpoints.
    """
    def __init__(self, checkpoint_dir):
        """
        Constructor for the CheckpointUtil class.
        :param checkpoint_dir: the directory to save the checkpoints
        """
        self.checkpoint_dir = checkpoint_dir

    def save_checkpoint(self, model, optimizer, epoch, loss, test_loss, best_loss, is_best):
        """
        Save the checkpoint
        :param model:  the model to save
        :param optimizer:  the optimizer to save
        :param epoch:  the epoch of the checkpoint
        :param loss:  the loss of the checkpoint
        :param test_loss:  the test loss of the checkpoint
        :param best_loss:  the best loss of the checkpoint
        :param is_best:  if the checkpoint is the best
        :return:  the filename of the saved checkpoint
        """
        state = {
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'best_loss': best_loss,
            'loss': loss,
            'test_loss': test_loss,
            'optimizer': optimizer.state_dict(),
        }
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)

        print("Saving checkpoint - epoch: {} loss: {} test_loss: {}".format(epoch, loss, test_loss))

        filename = self.checkpoint_dir + '/checkpoint_{}.pt.tar'.format(epoch)
        torch.save(state, filename)
        if is_best:
            shutil.copyfile(filename, self.checkpoint_dir + '/model_best_{}.pt.tar'.format(epoch))

        return filename

    def load_checkpoint(self, model, optimizer, check_point_name, reset_optimizer=False, device='cpu'):
        """
        Load the checkpoint
        :param model: model to load to
        :param optimizer: optimizer to load to
        :param check_point_name: name of the checkpoint
        :param reset_optimizer: if True, reset the optimizer
        :param device: device to load the model to
        :return: model, optimizer, epoch, best_loss, loss
        """
        filename = self.checkpoint_dir + '/' + check_point_name
        if os.path.isfile(filename):
            print("=> loading checkpoint '{}'".format(filename))
            checkpoint = torch.load(filename, map_location=device)
            epoch = checkpoint['epoch']
            best_loss = checkpoint['best_loss']
            loss = checkpoint['loss']
            model.load_state_dict(checkpoint['state_dict'])
            if not reset_optimizer:
                optimizer.load_state_dict(checkpoint['optimizer'])

            print("=> loaded checkpoint '{}' (epoch {})".format(filename, checkpoint['epoch']))
            return model, optimizer, epoch, best_loss, loss
        else:
            print("=> no checkpoint found at '{}'".format(filename))
            return model, optimizer, 0, 100, 100

    def load_checkpoint_stats(self, checkpoint_name):
        """
        Load the checkpoint stats
        :param checkpoint_name:  the name of the checkpoint
        :return:  the stats of the checkpoint
        """
        filename = self.checkpoint_dir + "/" + checkpoint_name
        if os.path.isfile(filename):
            print("=> loading checkpoint '{}'".format(filename))
            checkpoint = torch.load(filename)
            epoch = checkpoint['epoch']
            best_loss = checkpoint['best_loss']
            loss = checkpoint['loss']
            test_loss = checkpoint['test_loss']
            return {'epoch': epoch, 'best_loss': best_loss, 'loss': loss, 'test_loss': test_loss}


def load_checkpoint_stats(checkpoint_dir, name):
    """
    Load the checkpoint stats
    Used for testing a checkpoint when running this script
    :param checkpoint_dir: the directory of the checkpoint
    :param name: the name of the checkpoint
    :return: the stats of the checkpoint
    """
    checkpoint_util = CheckpointUtil(checkpoint_dir)
    print(checkpoint_util.load_checkpoint_stats(name))


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python checkpointsLogging.py <checkpoint_dir> <checkpoint_name>")
        sys.exit(1)
    load_checkpoint_stats(sys.argv[1], sys.argv[2])
