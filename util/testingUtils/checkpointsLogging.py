import os
import shutil
import sys

import torch


class CheckpointUtil:
    def __init__(self, checkpoint_dir):
        self.checkpoint_dir = checkpoint_dir

    def save_checkpoint(self, model, optimizer, epoch, loss, test_loss, best_loss, is_best):
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
    checkpoint_util = CheckpointUtil(checkpoint_dir)
    print(checkpoint_util.load_checkpoint_stats(name))


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python checkpointsLogging.py <checkpoint_dir> <checkpoint_name>")
        sys.exit(1)
    load_checkpoint_stats(sys.argv[1], sys.argv[2])
