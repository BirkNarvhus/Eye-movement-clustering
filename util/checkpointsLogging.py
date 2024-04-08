import os
import shutil

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

    def load_checkpoint(self, model, optimizer):
        filename = self.checkpoint_dir + '/checkpoint.pt.tar'
        if os.path.isfile(filename):
            print("=> loading checkpoint '{}'".format(filename))
            checkpoint = torch.load(filename)
            epoch = checkpoint['epoch']
            best_loss = checkpoint['best_loss']
            loss = checkpoint['loss']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})".format(filename, checkpoint['epoch']))
            return model, optimizer, epoch, best_loss, loss
        else:
            print("=> no checkpoint found at '{}'".format(filename))
            return model, optimizer, 0, 100, 100