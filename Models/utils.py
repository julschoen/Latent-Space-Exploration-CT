from enum import Enum
import numpy as np
import os
import json
import types
from functools import wraps
import io
from PIL import Image
import matplotlib.pyplot as plt

import torch
from torchvision.transforms import ToPILImage
from torchvision.utils import make_grid
from torchvision.transforms import Resize


class DeformatorType(Enum):
    FC = 1
    LINEAR = 2
    ID = 3
    ORTHO = 4
    PROJECTIVE = 5
    RANDOM = 6

DEFORMATOR_TYPE_DICT = {
    'fc': DeformatorType.FC,
    'linear': DeformatorType.LINEAR,
    'id': DeformatorType.ID,
    'ortho': DeformatorType.ORTHO,
    'proj': DeformatorType.PROJECTIVE,
    'random': DeformatorType.RANDOM,
}

class ShiftDistribution(Enum):
    NORMAL = 0,
    UNIFORM = 1,

SHIFT_DISTRIDUTION_DICT = {
    'normal': ShiftDistribution.NORMAL,
    'uniform': ShiftDistribution.UNIFORM,
    None: None
}


def torch_expm(A):
    n_A = A.shape[0]
    A_fro = torch.sqrt(A.abs().pow(2).sum(dim=(1, 2), keepdim=True))

    # Scaling step
    maxnorm = torch.tensor([5.371920351148152], dtype=A.dtype, device=A.device)
    zero = torch.tensor([0.0], dtype=A.dtype, device=A.device)
    n_squarings = torch.max(zero, torch.ceil(torch_log2(A_fro / maxnorm)))
    A_scaled = A / 2.0 ** n_squarings
    n_squarings = n_squarings.flatten().type(torch.int64)

    # Pade 13 approximation
    U, V = torch_pade13(A_scaled)
    P = U + V
    Q = -U + V
    R, _ = torch.solve(P, Q)

    # Unsquaring step
    res = [R]
    for i in range(int(n_squarings.max())):
        res.append(res[-1].matmul(res[-1]))
    R = torch.stack(res)
    expmA = R[n_squarings, torch.arange(n_A)]
    return expmA[0]

def make_noise(batch, dim, truncation=None):
    if isinstance(dim, int):
        dim = [dim]
    if truncation is None or truncation == 1.0:
        return torch.randn([batch] + dim+[1,1])
    else:
        return torch.from_numpy(truncated_noise([batch] + dim, truncation)).to(torch.float)


class MeanTracker(object):
    def __init__(self, name):
        self.values = []
        self.name = name

    def add(self, val):
        self.values.append(float(val))

    def mean(self):
        return np.mean(self.values)

    def flush(self):
        mean = self.mean()
        self.values = []
        return self.name, mean

def one_hot(dims, value, indx):
    vec = torch.zeros(dims)
    vec[indx] = value
    return vec

def save_run_params(args):
    os.makedirs(args['out'], exist_ok=True)
    with open(os.path.join(args['out'], 'args.json'), 'w') as args_file:
        json.dump(args, args_file)

def to_image(tensor, adaptive=False):
    if len(tensor.shape) == 4:
        tensor = tensor[0]
    if adaptive:
        tensor = (tensor - tensor.min()) / (tensor.max() - tensor.min())
        return ToPILImage()((255 * tensor.cpu().detach()).to(torch.uint8))
    else:
        tensor = (tensor + 1) / 2
        tensor.clamp(0, 1)
        return ToPILImage()((255 * tensor.cpu().detach()).to(torch.uint8))


def torch_log2(x):
    return torch.log(x) / np.log(2.0)


def torch_pade13(A):
    b = torch.tensor([64764752532480000., 32382376266240000., 7771770303897600.,
                      1187353796428800., 129060195264000., 10559470521600.,
                      670442572800., 33522128640., 1323241920., 40840800.,
                      960960., 16380., 182., 1.], dtype=A.dtype, device=A.device)

    ident = torch.eye(A.shape[1], dtype=A.dtype).to(A.device)
    A2 = torch.matmul(A, A)
    A4 = torch.matmul(A2, A2)
    A6 = torch.matmul(A4, A2)
    U = torch.matmul(A,
                     torch.matmul(A6, b[13] * A6 + b[11] * A4 + b[9] * A2) + b[7] * A6 + b[5] * A4 +
                     b[3] * A2 + b[1] * ident)
    V = torch.matmul(A6, b[12] * A6 + b[10] * A4 + b[8] * A2) + b[6] * A6 + b[4] * A4 + b[2] * A2 +\
        b[0] * ident
    return U, V

def add_forward_with_shift(generator):
    def gen_shifted(self, z, shift, *args, **kwargs):
        return self.forward(z + shift, *args, **kwargs)

    generator.gen_shifted = types.MethodType(gen_shifted, generator)
    generator.dim_shift = generator.dim_z


def gan_with_shift(gan_factory):
    @wraps(gan_factory)
    def wrapper(*args, **kwargs):
        gan = gan_factory(*args, **kwargs)
        add_forward_with_shift(gan)
        return gan

    return wrapper


def fig_to_image(fig):
    buf = io.BytesIO()
    fig.savefig(buf)
    buf.seek(0)
    return Image.open(buf)


@torch.no_grad()
def interpolate(G, z, shifts_r, shifts_count, dim, deformator=None, with_central_border=False, device='cpu'):
    shifted_images = []
    for shift in np.arange(-shifts_r, shifts_r + 1e-9, shifts_r / shifts_count):
        if deformator is not None:
            latent_shift = deformator(one_hot(deformator.input_dim, shift, dim).to(device))
        else:
            latent_shift = one_hot(G.dim_shift, shift, dim).to(device)
        shifted_image = G.gen_shifted(z, latent_shift).cpu()[0]
        if shift == 0.0 and with_central_border:
            shifted_image = add_border(shifted_image)

        shifted_images.append(shifted_image)
    return shifted_images


def add_border(tensor):
    border = 3
    for ch in range(tensor.shape[0]):
        color = 1.0 if ch == 0 else -1
        tensor[ch, :border, :] = color
        tensor[ch, -border:,] = color
        tensor[ch, :, :border] = color
        tensor[ch, :, -border:] = color
    return tensor


@torch.no_grad()
def make_interpolation_chart(G, deformator=None, z=None,
                             shifts_r=10.0, shifts_count=5,
                             dims=None, dims_count=10, texts=None, device='cpu', **kwargs):
    with_deformation = deformator is not None
    if with_deformation:
        deformator_is_training = deformator.training
        deformator.eval()
    z = z if z is not None else make_noise(1, G.dim_z).to(device)

    if with_deformation:
        original_img = G(z).cpu()
    else:
        original_img = G(z).cpu()
    imgs = []
    if dims is None:
        dims = range(dims_count)
    for i in dims:
        imgs.append(interpolate(G, z, shifts_r, shifts_count, i, deformator, device=device))

    rows_count = len(imgs) + 1
    fig, axs = plt.subplots(rows_count, **kwargs)

    axs[0].axis('off')
    axs[0].imshow(to_image(original_img, True))

    if texts is None:
        texts = dims
    for ax, shifts_imgs, text in zip(axs[1:], imgs, texts):
        ax.axis('off')
        plt.subplots_adjust(left=0.5)
        ax.imshow(to_image(make_grid(shifts_imgs, nrow=(2 * shifts_count + 1), padding=1), True))
        ax.text(-20, 21, str(text), fontsize=10)

    if deformator is not None and deformator_is_training:
        deformator.train()

    return fig


@torch.no_grad()
def inspect_all_directions(G, deformator, out_dir, zs=None, num_z=3, shifts_r=8.0, device='cpu'):
    os.makedirs(out_dir, exist_ok=True)

    step = 20
    max_dim = G.dim_shift
    zs = zs if zs is not None else make_noise(num_z, G.dim_z).to(device)
    shifts_count = zs.shape[0]

    for start in range(0, max_dim - 1, step):
        imgs = []
        dims = range(start, min(start + step, max_dim))
        for z in zs:
            z = z.unsqueeze(0)
            fig = make_interpolation_chart(
                G, deformator=deformator, z=z,
                shifts_count=shifts_count, dims=dims, shifts_r=shifts_r,
                dpi=250, figsize=(int(shifts_count * 4.0), int(0.5 * step) + 2), device=device)
            fig.canvas.draw()
            plt.close(fig)
            img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))

            # crop borders
            nonzero_columns = np.count_nonzero(img != 255, axis=0)[:, 0] > 0
            img = img.transpose(1, 0, 2)[nonzero_columns].transpose(1, 0, 2)
            imgs.append(img)

        out_file = os.path.join(out_dir, '{}_{}.jpg'.format(dims[0], dims[-1]))
        print('saving chart to {}'.format(out_file))
        Image.fromarray(np.hstack(imgs)).save(out_file)
        