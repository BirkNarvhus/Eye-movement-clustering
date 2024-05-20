"""
This file contains the LayerFactory class which is used to generate the model architecture from a file
or to write the model architecture to a file.

Uses the arc files
"""

class LayerFactory:
    """
    Class to generate the model architecture from a file
    """
    def __init__(self):
        """
        Constructor for the LayerFactory class
        """
        self.layers = {}

    def add_layer(self, name, pool_type="down"):
        """
        Add a layer to the model
        :param name:  name of the layer
        :param pool_type:  type of pooling
        :return:  name of the layer
        """
        new_index = len(self.layers.keys())
        self.layers[name + "_" + str(new_index)] = {}
        self.layers[name + "_" + str(new_index)]["pool_type"] = pool_type
        return name + "_" + str(new_index)

    def add_residual_block(self, name, channels, kernels_size):
        """
        Add a residual block to the layer
        :param name:  name of the layer
        :param channels:  channels of the block
        :param kernels_size:  kernel size of the block
        """
        new_index = len(self.layers.get(name, []))
        self.layers[name][new_index] = {}
        self.layers[name][new_index]["channels"] = channels
        self.layers[name][new_index]["kernels_size"] = kernels_size

    def generate_layer_array(self):
        """
        Converts the layers to an array (used for generating the model)
        :return: the layers as an array
        """
        layer_array = []
        for layer in self.layers:
            reslayer = []
            reslayer.append(self.layers[layer]["pool_type"])
            for key in self.layers[layer].keys():
                if key == "pool_type":
                    pass
                else:
                    reslayer.append(list(zip(self.layers[layer][key]["channels"], self.layers[layer][key]["kernels_size"])))
            layer_array.append(reslayer)
        return layer_array

    def generate_reverse_layer_array(self):
        """
        Generates the reverse layer array
        used for symetric models
        :return: the reverse layer array
        """
        layer_array = []
        for layer in self.layers:
            reslayer = []
            reslayer.append(self.layers[layer]["pool_type"])
            for key in self.layers[layer].keys():
                if key == "pool_type":
                    pass
                else:
                    reslayer.append(
                        list(zip([x[::-1] for x in self.layers[layer][key]["channels"]], self.layers[layer][key]["kernels_size"])))
            layer_array.insert(0, reslayer)
        return layer_array

    def get_last_size(self):
        """
        Get the last size of the model
        This is for calculating flattend output
        :return: upscale factor and feature size of the last layer
        """
        upscale_factor = 2 ** len([key for key in self.layers.keys() if self.layers[key]["pool_type"] == "down"])
        feature_size_last_layer = list((list(self.layers.values()))[-1].values())[-1]["channels"][-1][-1]
        return upscale_factor, feature_size_last_layer

    def read_from_file(self, filename, full_block_res=False, res_interval=3):
        """
        Read the model from a file
        :param filename: the filename
        :param full_block_res: if True, read the full block
        :param res_interval: the interval of the residual blocks
        """
        self.reset()
        print("Loading model from file: ", filename)
        kernels = []
        channels = []
        with open(filename, 'r') as f:
            layer_name = self.add_layer("down")
            for i, l in enumerate(f.readlines()):
                l.strip()
                if l == "\n":
                    continue
                l = l.replace("\n", "")
                if l == "down" or l == "none" or l == "up" or l == "temp_up":
                    if len(kernels) != 0:
                        self.add_residual_block(layer_name, channels, kernels)
                        kernels = []
                        channels = []
                    layer_name = self.add_layer("layer", pool_type=l)
                    pass
                else:
                    l = l.split(",")
                    if not full_block_res:
                        kernels = [tuple(map(int, l[:3]))]
                        channels = [tuple(map(int, l[3:]))]
                        self.add_residual_block(layer_name, channels, kernels)
                        kernels = []
                        channels = []
                    else:
                        kernels.append(tuple(map(int, l[:3])))
                        channels.append(tuple(map(int, l[3:])))
                        if res_interval != 0:
                            if i % res_interval == 0:

                                self.add_residual_block(layer_name, channels, kernels)
                                kernels = []
                                channels = []
            if len(kernels) != 0:
                self.add_residual_block(layer_name, channels, kernels)

            while True:
                if len(list(self.layers[list(self.layers.keys())[-1]].values())) == 0:
                    self.layers.pop(list(self.layers.keys())[-1])
                else:
                    break

    def write_to_file(self, filename):
        """
        Save the layers as arc file
        :param filename: the filename
        """
        with open(filename, 'w') as f:
            for i, layer in enumerate(self.layers):
                if i != 0:
                    f.write(self.layers[layer]["pool_type"] + "\n")
                del self.layers[layer]["pool_type"]
                for block in self.layers[layer]:
                    for channel, kernel in zip(self.layers[layer][block]["channels"], self.layers[layer][block]["kernels_size"]):
                        f.write(str(kernel[0]) + "," + str(kernel[1]) + "," + str(kernel[2]) + "," + str(channel[0]) + "," + str(channel[1]) + "\n")

    def revere_and_write_to_file(self, filename):
        """
        Reverse the layers and save it to file
        :param filename: the filename
        """
        layer_array = self.generate_reverse_layer_array()
        with open(filename, 'w') as f:
            for layer in layer_array:
                if layer[0] == "down":
                    layer[0] = "up"
                if layer[0] == "up" or layer[0] == "none":
                    f.write(layer[0] + "\n")
                for block in layer[1:]:
                    for channel, kernel in block:
                        f.write(str(kernel[0]) + "," + str(kernel[1]) + "," + str(kernel[2]) + "," + str(channel[0]) + "," + str(channel[1]) + "\n")
    def reset(self):
        self.layers = {}


def test():
    """
    Test the LayerFactory class.
    May be used to generate new arcs, if you don't want to write it manually
    This is hardcoded params
    """
    factory = LayerFactory()

    name = factory.add_layer("down")

    factory.add_residual_block(name, ((64, 128), (128, 64), (64, 32)), ((3, 3, 3), (3, 3, 3), (3, 3, 3)))
    factory.add_residual_block(name, ((32, 64), (64, 64), (64, 32)), ((3, 3, 3), (3, 3, 3), (3, 3, 3)))
    factory.add_residual_block(name, ((64, 128), (128, 64), (64, 32)), ((3, 3, 3), (3, 3, 3), (3, 3, 3)))
    factory.add_residual_block(name, ((64, 128), (128, 64), (64, 32)), ((3, 3, 3), (3, 3, 3), (3, 3, 3)))

    name = factory.add_layer("down")

    factory.add_residual_block(name, ((64, 128), (128, 64), (64, 32)), ((3, 3, 3), (3, 3, 3), (3, 3, 3)))
    factory.add_residual_block(name, ((32, 64), (64, 64), (64, 32)), ((3, 3, 3), (3, 3, 3), (3, 3, 3)))
    factory.add_residual_block(name, ((64, 128), (128, 64), (64, 32)), ((3, 3, 3), (3, 3, 3), (3, 3, 3)))

    factory.reset()

    factory.read_from_file("../models/cleandup/Arc/model_1.csv", full_block_res=True, res_interval=0)


    print(factory.generate_layer_array())
    #print([len(x) for x in factory.generate_layer_array()])
    factory.write_to_file("../models/cleandup/Arc/model_2.csv")


if __name__ == '__main__':
    test()