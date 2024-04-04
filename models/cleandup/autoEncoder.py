import torch
from torch import nn
from Blocks import UpsampleLayer, DownsampleLayer, Cumulativ_global_pooling
from util.layerFactory import LayerFactory


class Encoder(nn.Module):
    def __init__(self, layers, global_pooling=False):
        super(Encoder, self).__init__()

        self.convLayers = nn.ModuleList()

        for layer in layers:
            self.convLayers.append(DownsampleLayer([[y[0] for y in x] for x in layer],
                                                   kernels_size=[[y[1] for y in x] for x in layer]))

        self.global_pooling = Cumulativ_global_pooling() if global_pooling else None


    def forward(self, x):
        buffer = x.split(12, 2)
        stream_buffer = [[] for x in range(len(self.convLayers))]
        output_buffer = []
        for i in range(len(buffer)):
            x = buffer[i]
            print("Running on slice: ", i + 1)
            for buffer_index, layer in enumerate(self.convLayers):
                if len(stream_buffer[buffer_index]) > 0:
                    x = torch.cat([stream_buffer[buffer_index], x], 2)
                stream_buffer[buffer_index] = x[:, :, -3:, :, :]
                print("input size", x.shape)
                x = layer(x)[:, :, -12:, :, :]  # dont care about the first 3 because they are already in the buffer
            print("==="*10)
            output_buffer.append(x)
        if self.global_pooling is not None:
            # todo: implement avrage pooling
            print(torch.cat(output_buffer, dim=2).shape)
            x = self.global_pooling(torch.cat(output_buffer, dim=2))
        else:
            x = x[:, :, -1:, :, :]  # take the last time slize witch hopefully contains all the information
        return x


def test():
    layerfac = LayerFactory()

    layername = layerfac.add_layer("down")

    layerfac.add_residual_block(layername, ((1, 16), (16, 64), (64, 128)), ((3, 3, 3), (3, 3, 3), (3, 3, 3)))

    layername = layerfac.add_layer("down")
    layerfac.add_residual_block(layername, ((128, 128), (128, 64), (64, 64)), ((3, 3, 3), (3, 3, 3), (3, 3, 3)))

    model = Encoder(layerfac.generate_layer_array(), global_pooling=True)

    x = torch.randn(32, 1, 120, 32, 32)
    out = model(x)
    print(out.shape)

if __name__ == '__main__':
    test()