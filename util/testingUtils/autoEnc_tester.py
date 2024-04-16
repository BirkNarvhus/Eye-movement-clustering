import os
import sys

sys.path.append('C:\\Users\\vizlab_stud\\Documents\\pythonProjects\\eye-movement-classification')
from util.load_auto_enc_util import load_auto_encoder
from util.testingUtils.checkpointsLogging import CheckpointUtil
relative_path = ""


def load_auto_enc_no_decoder(checkpoint_file=None):
    if checkpoint_file is None or os.path.isfile(checkpoint_file) is False:
        raise NotImplementedError("Could not find checkpoint file. Please provide a valid checkpoint file.")
    checkpoint_dir_path = os.path.dirname(checkpoint_file)
    checkpoint_file_name = os.path.basename(checkpoint_file)

    enc_layer_file = relative_path + "content/Arc/model_3.csv"
    dec_layer_file = relative_path + "content/Arc/model_3_reverse.csv"

    bottleneck_input_channels = 216
    bottleneck_output_channels = 216
    lr = 0.0001

    model, optimizer = load_auto_encoder(enc_layer_file, dec_layer_file, bottleneck_input_channels, bottleneck_output_channels, lr)

    check_util = CheckpointUtil(checkpoint_dir_path)
    return check_util.load_checkpoint(model=model, optimizer=optimizer, check_point_name=checkpoint_file_name)


def main():
    if len(sys.argv) > 2:
        print("Usage: python autoEnc_tester.py <checkpoint_file>")
        sys.exit(1)

    model, optimizer, _, _, _  = load_auto_enc_no_decoder(checkpoint_file=sys.argv[1])
    print(model)
    print(optimizer)


if __name__ == '__main__':
    main()