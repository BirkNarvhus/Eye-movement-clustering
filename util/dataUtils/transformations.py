"""
Custom transformations for the OpenEDS dataset class
"""

import math

import numpy.random

from PIL import Image


class Transformations:
    """
    Base class for transformations
    """
    def __init__(self):
        pass

    def __call__(self, data):
        return self.transform(data)

    def transform(self, data):
        return data


class Crop_top(Transformations):
    """
    Crop the top of the data
    """
    def __init__(self, size):
        """
        Constructor for the crop top transformation
        :param size: the size to crop
        """
        super().__init__()
        self.size = size

    def transform(self, data):
        """
        Crop the top of the data
        :param data: the data to crop
        :return: the cropped data
        """
        data_y = data.shape[2]
        if data_y < self.size:
            raise ValueError("Data is smaller than the crop size")
        data = data[:, :, :data_y-self.size]
        return data


class Crop(Transformations):
    """
    Crop the data
    """
    def __init__(self, size):
        """
        Constructor for the crop transformations
        :param size:  the size to crop
        """
        super().__init__()
        self.size = size

    def transform(self, data):
        """
        Crop the data
        :param data:  the data to crop
        :return:  the cropped data
        """
        data_x = data.shape[1]
        data_y = data.shape[2]
        if data_x < self.size[0] or data_y < self.size[1]:
            raise ValueError("Data is smaller than the crop size")
        start_x = math.floor((data_x - self.size[0]) / 2)
        start_y = math.floor((data_y - self.size[1]) / 2)
        data = data[:, start_x:start_x + self.size[0], start_y:start_y + self.size[1]]
        return data


class RandomCrop(Transformations):
    """
    Random crop the data
    """
    def __init__(self, crop_size):
        """
        Constructor for the random crop transformation
        :param crop_size:  the size to crop
        """
        super().__init__()
        self.crop_size = crop_size

    def transform(self, data):
        """
        Random crop the data
        :param data:  the data to crop
        :return:  the cropped data
        """
        crop_factor_x = numpy.random.rand()
        crop_factor_y = numpy.random.rand()
        data = map(lambda x: x[int(crop_factor_x * self.crop_size):,
                             int(crop_factor_y * self.crop_size):], data)
        return numpy.array(list(data))


class Noise(Transformations):
    """
    Add noise to the data
    """
    def __init__(self, noise_factor):
        """
        Constructor for the noise transformation
        :param noise_factor: the factor of the noise
        """
        super().__init__()
        self.noise_factor = noise_factor

    def transform(self, data):
        """
        Add noise to the data
        :param data:  the data to add noise
        :return:  the data with noise
        """
        return data + numpy.random.randn(*data.shape) * self.noise_factor


class Normalize(Transformations):
    """
    Normalize the data
    """
    def __init__(self, mean, std):
        """
        Constructor for the normalize transformation
        :param mean:  the mean to normalize
        :param std:  the standard deviation to normalize
        """
        super().__init__()
        self.mean = mean
        self.std = std

    def transform(self, data):
        """
        Normalize the data
        :param data:  the data to normalize
        :return:  the normalized data
        """
        return (data - self.mean) / self.std


class Rotate(Transformations):
    """
    Rotate the data randomly
    """
    def __init__(self, angle):
        """
        Constructor for the rotate transformation
        :param angle:  the angle to rotate
        """
        super().__init__()
        self.angle = angle

    def transform(self, data):
        """
        Rotate the data
        :param data:  the data to rotate
        :return:  the rotated data
        """
        rand_angel = (numpy.random.rand()-0.5) * self.angle
        return numpy.array(list(map(lambda x: numpy.array(Image.fromarray(x).rotate(rand_angel)), data)))


class TempStride(Transformations):
    """
    Temporal stride transformation
    """
    def __init__(self, stride):
        """
        Constructor for the temporal stride transformation
        :param stride:  the stride to apply
        """
        super().__init__()
        self.stride = stride

    def transform(self, data):
        """
        Apply the temporal stride transformation
        :param data:  the data to apply the transformation
        :return:  the transformed data
        """
        return data[::self.stride]


def test():
    """
    Test the transformations
    """
    x = numpy.random.randn(60, 120, 120)
    random_crop = RandomCrop(10)
    print(random_crop(x).shape)

    print(Crop((64, 64))(x).shape)

    print(Noise(0.1)(x).shape)

    print(Normalize(0, 1)(x).shape)

    print(Rotate(10)(x).shape)

    print(Crop_top(30)(x).shape)

    print(TempStride(2)(x).shape)

if __name__ == "__main__":
    test()
