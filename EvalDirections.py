import torch
from torchvision.utils import make_grid
import torchvision.utils as vutils
from matplotlib import pyplot as plt
import json
import os
import sys

sys.path.append('./Models')

from Models.utils import to_image, interpolate, DEFORMATOR_TYPE_DICT
from Models.Deformator import LatentDeformator
from Models.DCGAN import make_gan
from Models.VAE import make_vae
from Models.ShiftPredictor import LeNetShiftPredictor, ResNetShiftPredictor

def load_from_dir(root_dir, G, model_index=None, shift_in_w=True, device='cpu'):
    args = json.load(open(os.path.join(root_dir, 'args.json')))
    args['w_shift'] = shift_in_w

    models_dir = os.path.join(root_dir, 'models')
    if model_index is None:
        models = os.listdir(models_dir)
        model_index = max(
            [int(name.split('.')[0].split('_')[-1]) for name in models
             if name.startswith('deformator')])

    if 'resolution' not in args.keys():
        args['resolution'] = 128

    deformator = LatentDeformator(
        shift_dim=G.dim_shift,
        input_dim=args['directions_count'] if 'directions_count' in args.keys() else None,
        out_dim=args['max_latent_dim'] if 'max_latent_dim' in args.keys() else None,
        type=DEFORMATOR_TYPE_DICT[args['deformator']])

    if 'shift_predictor' not in args.keys() or args['shift_predictor'] == 'ResNet':
        shift_predictor = ResNetShiftPredictor(G.dim_shift)
    elif args['shift_predictor'] == 'LeNet':
        shift_predictor = LeNetShiftPredictor(
            deformator.input_dim, 1)

    deformator_model_path = os.path.join(models_dir, 'deformator_{}.pt'.format(model_index))
    shift_model_path = os.path.join(models_dir, 'shift_predictor_{}.pt'.format(model_index))
    if os.path.isfile(deformator_model_path):
        deformator.load_state_dict(
            torch.load(deformator_model_path, map_location=torch.device('cpu')))
    if os.path.isfile(shift_model_path):
        shift_predictor.load_state_dict(
            torch.load(shift_model_path, map_location=torch.device('cpu')))

    return deformator.eval().to(device), G.eval().to(device), shift_predictor.eval().to(device)
        

gpu = torch.cuda.is_available()
device = torch.device("cuda" if gpu else "cpu")
gen = 'vae'


# Path to generator checkpoint
gen_path = 'Training/vae_log/decoder.pt'
# Path to latent direction log
latent_log = 'Training/save_location/'

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
