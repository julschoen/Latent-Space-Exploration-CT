import os
import matplotlib.pyplot as plt
import numpy as np
import torch.optim as optim

import torchvision
from torch.utils.data import DataLoader
import torchvision.utils as vutils
import pytorch_fid_wrapper as FID
import sys

sys.path.append('../')
sys.path.append('../Models')

from Dataset import LIDC
from Models.DCGAN import *

gpu = torch.cuda.is_available()
device = torch.device("cuda" if gpu else "cpu")
print("device:", device)

dataset_train = LIDC(augment=True)
dataset_test = LIDC(train=False)
batch_size = 128
generator_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=2)
batch_size = dataset_test.__len__()
generator_test = DataLoader(dataset_test, batch_size=batch_size, shuffle=True, num_workers=2)

netG = Generator().to(device)
netD = Discriminator().to(device)

# Apply the weights_init function to randomly initialize all weights
#  to mean=0, stdev=0.2.
netG.apply(weights_init)
netD.apply(weights_init)

criterion = nn.BCEWithLogitsLoss()

# Create batch of latent vectors to visualize progress of generator
fixed_noise = torch.randn(128, nz, 1, 1, device=device).float()


# Establish convention for real and fake labels during training
def make_labels(size):
    labels = (torch.randint(900, 1000, size, device=device) / 1000).float()
    return labels


real_label = 1.
fake_label = 0.

# Setup Adam optimizers for both G and D
lr = 0.0002
beta1 = 0.5
optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))

### Training
# Loggers
img_list = []
G_losses = []
D_losses = []
FID.set_config(device=device)
fid = []
fid_epoch = []
iters = 0
epochs = 30
sig = nn.Sigmoid()
path = 'gan_log'
if not os.path.exists(path):
    os.makedirs(path)

print("Starting Training Loop...")
for epoch in range(epochs):
    if epoch < 20 and epoch % 5 == 0 and (not epoch == 0):
        netD.std_reduce = netD.std_reduce + 10
    elif epoch == 20:
        netD.std_reduce = 100
    elif 20 < epoch < 26:
        netD.std_reduce = netD.std_reduce*4
    else:
        netD.noise = False
    for i, data in enumerate(generator_train, 0):
        netD.zero_grad()
        real = data.to(device)

        b_size = real.size(0)
        label = make_labels((b_size,))


        output = netD(real).view(-1)
        errD_real = criterion(output, label)
        errD_real.backward()
        D_x = sig(output).mean()

        noise = torch.randn(b_size, nz, 1, 1, dtype=torch.float, device=device)
        fake = netG(noise)
        label = label.fill_(fake_label)


        output = netD(fake.detach()).view(-1)
        errD_fake = criterion(output, label)
        errD_fake.backward()
        D_G_z1 = sig(output).mean()
        errD = errD_real + errD_fake

        optimizerD.step()

        netG.zero_grad()
        label = label.fill_(real_label)  # fake labels are real for generator cost


        output = netD(fake).view(-1)
        errG = criterion(output, label)
        errG.backward()
        D_G_z2 = sig(output).mean()

        optimizerG.step()
        if i % 100 == 0:
            with torch.no_grad():
                fid.append(
                    FID.fid(
                        fake.expand(-1, 3, -1, -1),
                        real_images=real.expand(-1, 3, -1, -1)
                    )
                )
            eG = errG.item()
            eD = errD.item()
            G_losses.append(eG)
            D_losses.append(eD)
            print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f\tFID %.4f'
                  % (epoch + 1, epochs, i, len(generator_train),
                     eD, eG, D_x.item(), D_G_z1.item(), D_G_z2.item(), fid[-1]))

        if (iters % 500 == 0) or ((epoch == epochs) and (i == len(generator_train) - 1)):
            with torch.no_grad():
                fake = netG(fixed_noise).detach().cpu()
            torchvision.utils.save_image(
                vutils.make_grid(fake, padding=2, normalize=True)
                , os.path.join(path, f'{iters}.png'))

        iters += 1

    fid_epoch.append(np.array(fid).mean())
    fid = []
    if epoch + 1 % 5 == 0 and (not epoch == 0):
        netD.std_reduce = netD.std_reduce + 10
    torch.save({
        'epoch': epoch,
        'modelG_state_dict': netG.state_dict(),
        'modelD_state_dict': netD.state_dict(),
        'optimizerG_state_dict': optimizerG.state_dict(),
        'optimizerD_state_dict': optimizerD.state_dict(),
        'lossG': G_losses,
        'lossD': D_losses,
        'fid': fid_epoch,
    }, os.path.join(path, 'checkpoint.pt'))

print('...Done')
