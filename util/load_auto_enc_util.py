"""
This file contains a function to load an autoencoder from arc files
"""

import torch
import sys
sys.path.append('C:\\Users\\vizlab_stud\\Documents\\pythonProjects\\eye-movement-classification')

from models.finalImplementations.EncoderDecoder import EncoderDecoder
from util.layerFactory import LayerFactory


def load_auto_encoder(enc_layer_file, dec_layer_file, bottleneck_input_channels=600,
                      bottleneck_output_channels=1000, lr=0.0001, optimizer_class=torch.optim.Adam,
                      remove_decoder=False, res_interval=2, weight_decay=1e-6):
    """
    Load an autoencoder from two files containing the layer information of the encoder and decoder
    :param enc_layer_file: the file containing the layer information of the encoder
    :param dec_layer_file: the file containing the layer information of the decoder
    :param bottleneck_input_channels: the number of input channels of the bottleneck
    :param bottleneck_output_channels: the number of output channels of the bottleneck
    :param lr: the learning rate
    :param optimizer_class: the optimizer class
    :param remove_decoder: if True, remove the decoder
    :param res_interval: the resolution interval
    :param weight_decay: the weight decay
    :return: the model and the optimizer
    """
    layerfac = LayerFactory()

    layerfac.read_from_file(enc_layer_file, full_block_res=True, res_interval=res_interval)
    layers_enc = layerfac.generate_layer_array()

    layerfac.read_from_file(dec_layer_file, full_block_res=True, res_interval=res_interval)
    layers_dec = layerfac.generate_layer_array()

    model = EncoderDecoder(layers_enc, layers_dec, bottleneck_input_channels, bottleneck_output_channels,
                           remove_decoder=remove_decoder)

    optimizer = optimizer_class(
        [params for params in model.parameters() if params.requires_grad],
        lr=lr,
        weight_decay=weight_decay,
    )

    return model, optimizer

