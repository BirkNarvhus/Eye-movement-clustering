import os

import numpy
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from matplotlib import pyplot as plt
from PIL import Image


###
# util class for loading the openEDS dataset
# the data is stored in npy files for each image
# all frames in a video are devided into video subfolders
# the data is stored in the following structure
# root
#   - video1
#       - frame1.npy
#       - frame2.npy
#       - ...
#   - video2
#       - frame1.npy
#       - frame2.npy
#       - ...
#   - ...
###


class Loader:
    def __init__(self, data, batch_size=32):
        self.data = data
        self.batch_size = batch_size
        self.currBatch = 0

    def get_batch(self):
        """
        Get a batch of data
        :param batch_size: the size of the batch
        :return: a batch of data
        """
        if self.currBatch + self.batch_size < len(self.data):
            self.currBatch += self.batch_size
            return self.data[self.currBatch - self.batch_size:self.currBatch]
        else:
            return self.data[self.currBatch:]

    def __iter__(self):
        return self

    def __next__(self):
        if self.currBatch < len(self.data):
            return torch.tensor(numpy.expand_dims(self.get_batch(), axis=1), dtype=torch.float32)
        else:
            raise StopIteration


class OpenEDSLoader:
    def __init__(self, root, batch_size=32, shuffle=True, max_videos=None, save_path=None, save_anyway=False):
        self.root = root
        self.videos = (os.listdir(root))[2:]
        self.max_videos = max_videos
        self.save_path = save_path
        self.save_anyway = save_anyway
        self.batch_size = batch_size
        self.currBatch = 0

        self.data = self.load_data() if root is not None else None
        if shuffle:
            self.shuffle()

        self.test = Loader((self.data[int(len(self.data) * 0.6):int(len(self.data) * 0.8)]), batch_size=self.batch_size)
        self.train = Loader(self.data[:int(len(self.data) * 0.6)], batch_size=self.batch_size)
        self.valid = Loader(self.data[int(len(self.data) * 0.8):], batch_size=self.batch_size)

        self.data = None

    def load_data(self):
        """
        Load the data from the root directory
        :return: a list of numpy arrays
        """
        if self.save_path is not None and os.path.exists(self.save_path) and not self.save_anyway:
            print("Loading openEDS dataset from ", self.save_path)
            with open(self.save_path, 'rb') as f:
                return np.load(f)

        print("Loading openEDS dataset from ", self.root)
        data = []
        for video in tqdm(self.videos):
            video_path = os.path.join(self.root, video)
            frames = os.listdir(video_path)
            frames.sort()
            video_data = []
            for frame in frames:
                if frame.endswith('.npy'):
                    continue
                frame_path = os.path.join(video_path, frame)
                frame_data = Image.open(frame_path)
                video_data.append(np.array(frame_data))

            if len(video_data) > 128:
                video_data = video_data[:128]
            else:
                while len(video_data) < 128:
                    video_data.append(np.zeros_like(video_data[0]))

            data.append(np.array(video_data))
            if self.max_videos is not None and len(data) >= self.max_videos:
                break

        data = np.array(data)
        if self.save_path is not None or self.save_anyway:
            print("Saving openEDS dataset to ", self.save_path)
            with open(self.save_path, 'wb') as f:
                np.save(f, data)
        return data

    def shuffle(self):
        """
        Shuffle the data
        """
        np.random.shuffle(self.data)

    def save(self, path):
        """
        Save the data to a file
        :param path: the path to save the data to
        """
        with open(path, 'wb') as f:
            np.save(f, self.data)

    def get_loaders(self):
        return self.train, self.valid, self.test


def test():
    root = '../data/openEDS/openEDS'
    save_path = '../data/openEDS/openEDS.npy'
    loader = OpenEDSLoader(root, batch_size=32, shuffle=True, max_videos=None, save_path=save_path, save_anyway=True)

    train, _, _ = loader.get_loaders()
    batch = next(iter(train))
    print(batch.shape)
    fig, ax = plt.subplots(1, 10)
    for i in range(10):
        ax[i].imshow(batch[0][10 + i].squeeze(), cmap='gray')
    plt.show()


if __name__ == '__main__':
    test()