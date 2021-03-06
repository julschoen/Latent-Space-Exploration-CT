import os
import numpy as np
import torch
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader
import torchvision.utils as vutils
import pytorch_fid_wrapper as FID
import sys

sys.path.append('../')
sys.path.append('../Models')

from Dataset import LIDC
from Models.VAE import VAE

gpu = torch.cuda.is_available()
device = torch.device("cuda" if gpu else "cpu")

dataset_train = LIDC(augment=True)
dataset_test = LIDC(train=False)

batch_size = 128
generator_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=2)
generator_test = DataLoader(dataset_test, batch_size=dataset_test.__len__(), shuffle=True, num_workers=2)


class LOGMSELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(self, input, target):
        return torch.log(self.mse(input, target))


vae = VAE(device=device)
vae.to(device)
criterion = LOGMSELoss()
optimizer = torch.optim.Adam(vae.parameters(), lr=1e-4)

beta = 0.01

# Stats
losses = []
FID.set_config(device=device)
fid = []
fid_epoch = []

iters = 0
epochs = 50
path = 'vae_log'
if not os.path.exists(path):
    os.makedirs(path)

print("Starting Training Loop...")
for epoch in range(epochs):
    for i, data in enumerate(generator_train):
        data = data.to(device)

        kl, pred = vae(data)

        rec_loss = criterion(target=data,input=pred)
        loss = beta*kl + rec_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (iters % 500 == 0) or ((epoch == epochs) and (i == len(generator_train)-1)):
            with torch.no_grad():
                fid.append(
                    FID.fid(
                        pred.expand(-1,3,-1,-1), 
                        real_images=data.expand(-1,3,-1,-1)
                        )
                    )
            
            l = loss.item()
            losses.append(l)
            print('[%d/%d][%d/%d]\tRec Loss: %.4f\tKL Divergence: %.4f\tKL %.4f\tFID: %.4f'
                    % (epoch+1, epochs, i, len(generator_train),rec_loss,beta*kl, kl,fid[-1]))
            _, z = vae.encoder(data)
            torchvision.utils.save_image(
                vutils.make_grid(pred, padding=2, normalize=True)
                , os.path.join(path, f'{iters}.png'))
        elif i % 100 == 0:
            l = loss.item()
            losses.append(l)
            print('[%d/%d][%d/%d]\tRec Loss: %.4f\tKL Divergence: %.4f\tKL %.4f'
                    % (epoch+1, epochs, i, len(generator_train),rec_loss,beta*kl, kl))

        iters += 1
    
    fid_epoch.append(np.array(fid).mean())
    fid = []    
    torch.save({
        'epoch': epoch,
        'state_dict': vae.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': losses,
        'fid': fid_epoch,
        }, os.path.join(path, 'checkpoint.pt'))


torch.save({
    'state_dict': vae.decoder.state_dict(),
}, os.path.join(path, 'decoder.pt'))
torch.save({
    'state_dict': vae.encoder.state_dict(),
}, os.path.join(path, 'encoder.pt'))

print('...Done')
