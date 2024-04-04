

class LayerFactory:
    def __init__(self):
        self.layers = {}

    def add_layer(self, name):
        new_index = len(self.layers.keys())
        self.layers[name + "_" + str(new_index)] = {}
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
            for key in self.layers[layer].keys():
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
        upscale_factor = 2 ** len(self.layers.keys())
        feature_size_last_layer = list((list(self.layers.values()))[-1].values())[-1]["channels"][-1][-1]
        return upscale_factor, feature_size_last_layer

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

    print(factory.generate_layer_array()[0])
    print(factory.generate_layer_array()[1])
    print(factory.generate_reverse_layer_array()[0])
    print(factory.generate_reverse_layer_array()[1])


    print(factory.get_upscale_factor())

if __name__ == '__main__':
    test()