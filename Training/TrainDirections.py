import torch
import os
import matplotlib
from Models.DCGAN import make_gan
from Models.VAE import make_vae
from Models.Deformator import LatentDeformator
from LatentTrainer import Params, Trainer
from Models.ShiftPredictor import ResNetShiftPredictor, LeNetShiftPredictor
from Models.utils import save_run_params, DEFORMATOR_TYPE_DICT, make_noise, inspect_all_directions

matplotlib.use("Agg")


def save_results_charts(G, deformator, params, out_dir, device):
    deformator.eval()
    G.eval()
    z = make_noise(3, G.dim_z, params.truncation).cuda()
    inspect_all_directions(
        G, deformator, os.path.join(out_dir, 'charts_s{}'.format(int(params.shift_scale))),
        zs=z, shifts_r=params.shift_scale, device=device)
    inspect_all_directions(
        G, deformator, os.path.join(out_dir, 'charts_s{}'.format(int(3 * params.shift_scale))),
        zs=z, shifts_r=3 * params.shift_scale, device=device)


gpu = torch.cuda.is_available()
device = torch.device("cuda" if gpu else "cpu")
print("device:", device)

torch.set_default_dtype(torch.float32)

#If Generator VAE pass decoder checkpoint
args = {
    'deformator': 'proj',
    'directions_count': 100,
    'shift_predictor': 'LeNet',
    'out': 'save_location',
    'gen_path': 'gan_log/checkpoint.pt',
    'def_random_init': True,
    'gen': 'gan'
}

save_run_params(args)

if args['gen'] == 'gan':
    G = make_gan(args['gen_path'])
else:
    G = make_vae(args['gen_path'])

deformator = LatentDeformator(shift_dim=100,
                              input_dim=args['directions_count'],
                              type=DEFORMATOR_TYPE_DICT[args['deformator']],
                              random_init=args['def_random_init']).to(device)

if args['shift_predictor'] == 'ResNet':
    shift_predictor = ResNetShiftPredictor(deformator.input_dim, 1).to(device)
else:
    shift_predictor = LeNetShiftPredictor(deformator.input_dim, 1).to(device)

params = Params(**args)
trainer = Trainer(params, out_dir=args['out'], device=device)
trainer.train(G, deformator, shift_predictor)

save_results_charts(G, deformator, params, trainer.log_dir, device=device)
