

class LayerFactory:
    def __init__(self):
        self.layers = {}

    def add_layer(self, name, pool_type="down"):
        new_index = len(self.layers.keys())
        self.layers[name + "_" + str(new_index)] = {}
        self.layers[name + "_" + str(new_index)]["pool_type"] = pool_type
        return name + "_" + str(new_index)

    def add_residual_block(self, name, channels, kernels_size):
        new_index = len(self.layers.get(name, []))
        self.layers[name][new_index] = {}
        self.layers[name][new_index]["channels"] = channels
        self.layers[name][new_index]["kernels_size"] = kernels_size

    def generate_layer_array(self):
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
        layer_array = []
        for layer in self.layers:
            reslayer = []
            for key in self.layers[layer].keys():
                reslayer.append(list(zip(self.layers[layer][key]["channels"], self.layers[layer][key]["kernels_size"])))
            layer_array.insert(0, reslayer)
        return layer_array

    def get_last_size(self):
        upscale_factor = 2 ** len([key for key in self.layers.keys() if self.layers[key]["pool_type"] == "down"])
        feature_size_last_layer = list((list(self.layers.values()))[-1].values())[-1]["channels"][-1][-1]
        return upscale_factor, feature_size_last_layer

    def read_from_file(self, filename, full_block_res=False, res_interval=3):
        kernels = []
        channels = []
        with open(filename, 'r') as f:
            layer_name = self.add_layer("down")
            for i, l in enumerate(f.readlines()):
                l.strip()
                if l == "\n":
                    continue
                l = l.replace("\n", "")
                if l == "down" or l == "none":
                    if len(kernels) != 0:
                        self.add_residual_block(layer_name, channels, kernels)
                        kernels = []
                        channels = []
                    layer_name = self.add_layer("down", pool_type=l)
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
        with open(filename, 'w') as f:
            for i, layer in enumerate(self.layers):
                if i != 0:
                    f.write(self.layers[layer]["pool_type"] + "\n")
                del self.layers[layer]["pool_type"]
                for block in self.layers[layer]:
                    for channel, kernel in zip(self.layers[layer][block]["channels"], self.layers[layer][block]["kernels_size"]):
                        f.write(str(kernel[0]) + "," + str(kernel[1]) + "," + str(kernel[2]) + "," + str(channel[0]) + "," + str(channel[1]) + "\n")
    def reset(self):
        self.layers = {}


def test():
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