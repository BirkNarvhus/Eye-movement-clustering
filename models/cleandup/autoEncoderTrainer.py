from torch import nn
import torch
from datetime import datetime
import warnings


import sys


sys.path.append('C:\\Users\\vizlab_stud\\Documents\\pythonProjects\\eye-movement-classification')

from util.testingUtils.checkpointsLogging import CheckpointUtil
from util.dataset_loader import OpenEDSLoader
from util.load_auto_enc_util import load_auto_encoder
from util.transformations import *

warnings.simplefilter("ignore", UserWarning)

device = "cpu"
if torch.cuda.is_available():
    device = "cuda:0"

relative_path = ""


root = relative_path + 'data/openEDS/openEDS'
save_path = relative_path + 'data/openEDS/openEDS.npy'


batch_size = 8
log_interval = 1
lr = 0.0001
n_epochs = 20
steps = 0
max_batches = 0  # all if 0
lossfunction = nn.MSELoss()

arc_filename_enc = relative_path + "content/Arc/model_3.csv"
arc_filename_dec = relative_path + "content/Arc/model_3_reverse.csv"

model_name = arc_filename_enc.split('/')[2].split('.')[0] + "auto_encoder_mse"

checkpoint_dir = relative_path + 'content/saved_models/autoEncMSEloss/' + model_name
output_dir = relative_path + 'content/saved_outputs/autoEnc/'


transformations = [
    TempStride(2),
    Crop_top(20),  # centers the image better
    Crop((256, 256)),
    Normalize(76.3, 41.7)
]


loader = OpenEDSLoader(root, batch_size=batch_size, shuffle=True, max_videos=None, save_path=save_path,
                       save_anyway=False,
                       transformations=transformations, sim_clr=False)

train_loader, test_loader, _ = loader.get_loaders()


model, optimizer = load_auto_encoder(arc_filename_enc, arc_filename_dec, 216, 216, lr, torch.optim.Adam)

'''
optimizer = LARS(
        [params for params in model.parameters() if params.requires_grad],
        lr=0.2,
        weight_decay=1e-6,
        exclude_from_weight_decay=["batch_normalization", "bias"],
    )

'''

print("running on ", device)


def test(test_loader, model):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        idx = 0
        for x in test_loader:
            idx += 1
            x = x.to(device=device, dtype=torch.float)

            z = model(x)
            if len(z) != batch_size:
                break
            loss = lossfunction(x, z)
            test_loss += loss.item()
        if idx == 0:
            return test_loss
        test_loss /= idx
        print('\nTest set: Average loss: {:.4f}'.format(
            test_loss))
        return test_loss


if __name__ == '__main__':
    model.to(device)

    loss_fn = lossfunction

    start_time = datetime.now().replace(microsecond=0)

    checkpoint_util = CheckpointUtil(checkpoint_dir)

    print("Starting training at {}...".format(start_time))
    best_loss_test = 1000000
    best_loss = 1000000
    for epoch in range(n_epochs):
        time0 = datetime.now()
        print("")
        train_loss = 0
        model.train()
        for idx, x_i in enumerate(train_loader):
            x_i = x_i.to(device, dtype=torch.float)

            z_i = model(x_i)
            loss = loss_fn(x_i, z_i)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # Validation
            train_loss = loss
            steps += batch_size
            if idx > 0:

                current_time = datetime.now().replace(microsecond=0) - start_time
                delta_time = datetime.now() - time0
                predicted_finish = delta_time * (len(train_loader)) * (n_epochs - epoch - 1) + current_time
                time_left = predicted_finish - current_time

                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tSteps: {}\t{}s/it\tRunning Time: {} - {}\tTime left: {}'.format(
                    epoch, idx * batch_size, len(train_loader) * batch_size,
                           100. * idx / len(train_loader), train_loss.item(), batch_size,
                           str(delta_time), str(current_time), str(predicted_finish), str(time_left)))

                time0 = datetime.now()

        loss = test(test_loader, model)
        if loss < best_loss_test:
            best_loss_test = loss

        if train_loss < best_loss:
            best_loss = train_loss

        if (loss == best_loss or train_loss == best_loss) and epoch > 0:
            checkpoint_util.save_checkpoint(model, optimizer, epoch, train_loss, loss, best_loss, True)
        elif epoch % log_interval == 0 and epoch > 0 or epoch == n_epochs - 1:
            checkpoint_util.save_checkpoint(model, optimizer, epoch, train_loss, loss, best_loss, False)
    pytorch_total_params = sum(p.numel() for p in model.parameters())
    print('Number of params: {}'.format(pytorch_total_params))
    print("Training finished. at {}".format(datetime.now().replace(microsecond=0)))
