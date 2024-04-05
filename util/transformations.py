import math

import numpy.random

from PIL import Image


class Transformations:
    def __init__(self):
        pass

    def __call__(self, data):
        return self.transform(data)

    def transform(self, data):
        return data


class Crop_top(Transformations):
    def __init__(self, size):
        super().__init__()
        self.size = size

    def transform(self, data):
        data_y = data.shape[2]
        if data_y < self.size:
            raise ValueError("Data is smaller than the crop size")
        data = data[:, :, :data_y-self.size]
        return data


class Crop(Transformations):
    def __init__(self, size):
        super().__init__()
        self.size = size

    def transform(self, data):
        data_x = data.shape[1]
        data_y = data.shape[2]
        if data_x < self.size[0] or data_y < self.size[1]:
            raise ValueError("Data is smaller than the crop size")
        start_x = math.floor((data_x - self.size[0]) / 2)
        start_y = math.floor((data_y - self.size[1]) / 2)
        data = data[:, start_x:start_x + self.size[0], start_y:start_y + self.size[1]]
        return data


class RandomCrop(Transformations):
    def __init__(self, crop_size):
        super().__init__()
        self.crop_size = crop_size

    def transform(self, data):
        crop_factor_x = numpy.random.rand()
        crop_factor_y = numpy.random.rand()
        data = map(lambda x: x[int(crop_factor_x * self.crop_size):,
                             int(crop_factor_y * self.crop_size):], data)
        return numpy.array(list(data))


class Noise(Transformations):
    def __init__(self, noise_factor):
        super().__init__()
        self.noise_factor = noise_factor

    def transform(self, data):
        return data + numpy.random.randn(*data.shape) * self.noise_factor


class Normalize(Transformations):
    def __init__(self, mean, std):
        super().__init__()
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std


class Rotate(Transformations):
    def __init__(self, angle):
        super().__init__()
        self.angle = angle

    def transform(self, data):
        rand_angel = (numpy.random.rand()-0.5) * self.angle
        return numpy.array(list(map(lambda x: numpy.array(Image.fromarray(x).rotate(rand_angel)), data)))


class TempStride(Transformations):
    def __init__(self, stride):
        super().__init__()
        self.stride = stride

    def transform(self, data):
        return data[::self.stride]

def test():
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
