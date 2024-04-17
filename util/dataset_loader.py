import os

import numpy
import numpy as np
import torch
from tqdm import tqdm
from matplotlib import pyplot as plt
from util.transformations import *


class Loader:
    def __init__(self, data, batch_size=32, transformations=None, sim_clr=False):
        self.data = data
        self.batch_size = batch_size
        self.currBatch = 0
        self.sim_clr = sim_clr

        self.transformations = transformations

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
        self.currBatch = 0
        return self

    def __next__(self):
        if self.currBatch + self.batch_size < len(self.data):
            bach = self.get_batch()
            return (torch.tensor(numpy.expand_dims(self.apply_transformation(bach), axis=1), dtype=torch.float32),
                    torch.tensor(numpy.expand_dims(self.apply_transformation(bach), axis=1),
                                 dtype=torch.float32)) if self.sim_clr else \
                (torch.tensor(numpy.expand_dims(self.apply_transformation(bach), axis=1), dtype=torch.float32))
        else:
            raise StopIteration

    def apply_transformation(self, data):
        """
        Apply the transformations to the data
        :param data:
        :return:
        """
        for transformation in self.transformations:
            data = list(map(transformation, data))

        data = np.array(data)

        return data

    def __len__(self):
        return len(self.data) // self.batch_size


class OpenEDSLoader:
    def __init__(self, root, batch_size=32, shuffle=True, max_videos=None, save_path=None, save_anyway=False,
                 transformations=None, sim_clr=False, split_frames=None):
        self.root = root
        self.videos = (os.listdir(root))[2:]
        self.max_videos = max_videos
        self.save_path = save_path
        self.save_anyway = save_anyway
        self.batch_size = batch_size
        self.currBatch = 0
        self.split_frames = split_frames
        self.data = self.load_data() if root is not None else None
        if shuffle:
            self.shuffle()

        self.test = Loader((self.data[int(len(self.data) * 0.8):]), batch_size=self.batch_size,
                           transformations=transformations, sim_clr=sim_clr)
        self.train = Loader(self.data[:int(len(self.data) * 0.8)], batch_size=self.batch_size,
                            transformations=transformations, sim_clr=sim_clr)
        self.valid = Loader(self.data[int(len(self.data) * 0.8):], batch_size=self.batch_size,
                            transformations=transformations, sim_clr=sim_clr)

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

            if self.split_frames is None:
                if len(video_data) > 120:
                    video_data = video_data[:120]
                else:
                    while len(video_data) < 120:
                        video_data.append(np.zeros_like(video_data[0]))

                data.append(np.array(video_data))
                if self.max_videos is not None and len(data) >= self.max_videos:
                    break
            else:
                for i in range(0, len(video_data), self.split_frames):
                    split_data = video_data[i:i + self.split_frames]
                    if len(split_data) < self.split_frames:
                        while len(split_data) < self.split_frames:
                            split_data.append(np.zeros_like(split_data[0]))
                    data.append(np.array(split_data))
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
    save_path = '../data/openEDS/openEDSSplit.npy'

    transformations = [
        Crop_top(20),  # centers the image better
        RandomCrop(20),
        Crop((256, 256)),
        Rotate(40),
        Noise(0.6),
    ]

    loader = OpenEDSLoader(root, batch_size=8, shuffle=True, max_videos=None, save_path=save_path, save_anyway=True,
                           transformations=transformations, sim_clr=True, split_frames=6)

    train, _, _ = loader.get_loaders()
    batch = next(iter(train))
    print(len(train))

    x_batch, y_batch = batch
    print(x_batch.shape, y_batch.shape)
    '''
    fig, ax = plt.subplots(2, 10)
    for x in range(10):
        ax[0, x].imshow(x_batch[0][0][10 + x].squeeze(), cmap='gray')
        ax[1, x].imshow(y_batch[0][0][10 + x].squeeze(), cmap='gray')

    plt.show()
    del x_batch, y_batch, batch, train, loader

    data = np.load(save_path)

    print("mean:" + str(np.mean(data)) + "   STD:" + str(np.std(data)))
    
    '''

if __name__ == '__main__':
    test()
