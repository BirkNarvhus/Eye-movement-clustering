import torch
import sys
sys.path.append('C:\\Users\\vizlab_stud\\Documents\\pythonProjects\\eye-movement-classification')

from models.cleandup.EncoderDecoder import EncoderDecoder
from util.layerFactory import LayerFactory


def get_n_params(model):
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()

    size_all_mb = (param_size + buffer_size) / 1024 ** 2
    print('model size: {:.3f}MB'.format(size_all_mb))
    return param_size


def load_auto_encoder(enc_layer_file, dec_layer_file, data_size, bottleneck_input_channels=600,
                      bottleneck_output_channels=1000, lr=0.0001, optimizer_class=torch.optim.Adam):
    layerfac = LayerFactory()

    layerfac.read_from_file(enc_layer_file, full_block_res=True, res_interval=2)
    layers_enc = layerfac.generate_layer_array()

    layerfac.read_from_file(dec_layer_file, full_block_res=True, res_interval=2)
    layers_dec = layerfac.generate_layer_array()

    model = EncoderDecoder(layers_enc, layers_dec, data_size, bottleneck_input_channels, bottleneck_output_channels)

    optimizer = optimizer_class(
        [params for params in model.parameters() if params.requires_grad],
        lr=lr,
        weight_decay=1e-6,
    )

    return model, optimizer

