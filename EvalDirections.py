import torch
from torchvision.utils import make_grid
import torchvision.utils as vutils
from matplotlib import pyplot as plt
import json
import os

from Models.utils import to_image, interpolate, DEFORMATOR_TYPE_DICT, load_from_dir
from Models.Deformator import LatentDeformator
from Models.DCGAN import make_gan
from Models.VAE import make_vae
from Models.ShiftPredictor import LeNetShiftPredictor, ResNetShiftPredictor

gpu = torch.cuda.is_available()
device = torch.device("cuda" if gpu else "cpu")
gen = 'vae'


# Path to generator checkpoint
gen_path = 'Training/vae_log/decoder.pt'
# Path to latent direction log
latent_log = 'Training/LeNetProj/'

if gen == 'gan':
    G = make_gan(gen_path)
else:
    G = make_vae(gen_path)

deformator, G, shift_predictor = load_from_dir(
    latent_log,
    G=G,
    device=device
)

rows = 8
plt.figure(figsize=(20, rows), dpi=250)

inspection_dim = 14
zs = torch.randn([rows, G.dim_z, 1, 1], device=device)

for z, i in zip(zs, range(rows)):
    interpolation_deformed = interpolate(
        G, z.unsqueeze(0),
        shifts_r=6,
        shifts_count=3,
        dim=inspection_dim,
        deformator=deformator,
        with_central_border=True,
        device=device
    )

    plt.subplot(rows, 1, i + 1)
    plt.axis('off')
    grid = make_grid(interpolation_deformed, nrow=11, padding=1, pad_value=0.0)
    grid = torch.clamp(grid, -1, 1)
    plt.imshow(to_image(grid))

# # Generate Images
noise = torch.randn(128, G.dim_z, 1, 1, dtype=torch.float, device=device)
fake = G(noise)
g = vutils.make_grid(fake, padding=2, normalize=True, nrow=8).cpu()
plt.figure(figsize=(12, 20))
plt.imshow(g.permute(1, 2, 0).detach().numpy())
plt.tick_params(
    axis='both',
    which='both',
    bottom=False,
    top=False,
    labelbottom=False)
plt.tick_params(
    axis='y',
    which='both',
    left=False,
    right=False,
    labelleft=False)

plt.show()
