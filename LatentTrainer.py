import os
import json
from tensorboardX import SummaryWriter
import torch
import torch.nn as nn
from Models.utils import ShiftDistribution, make_noise, make_interpolation_chart, fig_to_image
from Models.utils import MeanTracker, DeformatorType


class Params(object):
    def __init__(self, **kwargs):
        self.shift_scale = 6.0
        self.min_shift = 0.5
        self.shift_distribution = ShiftDistribution.UNIFORM

        self.deformator_lr = 0.0001
        self.shift_predictor_lr = 0.0001
        self.n_steps = int(1e+5)
        self.batch_size = 256

        self.directions_count = None
        self.max_latent_dim = None

        self.label_weight = 1.0
        self.shift_weight = 0.1

        self.steps_per_log = 10
        self.steps_per_save = 10000
        self.steps_per_img_log = 1000
        self.steps_per_backup = 1000

        self.truncation = None

        for key, val in kwargs.items():
            if val is not None:
                self.__dict__[key] = val


class Trainer(object):
    def __init__(self, params=Params(), out_dir='', verbose=False, device='cpu'):
        if verbose:
            print('Trainer inited with:\n{}'.format(str(params.__dict__)))
        self.p = params
        self.log_dir = os.path.join(out_dir, 'logs')
        os.makedirs(self.log_dir, exist_ok=True)
        self.cross_entropy = nn.CrossEntropyLoss()

        self.device = device
        tb_dir = os.path.join(out_dir, 'tensorboard')
        self.models_dir = os.path.join(out_dir, 'models')
        self.images_dir = os.path.join(self.log_dir, 'images')
        os.makedirs(tb_dir, exist_ok=True)
        os.makedirs(self.models_dir, exist_ok=True)
        os.makedirs(self.images_dir, exist_ok=True)

        self.checkpoint = os.path.join(out_dir, 'checkpoint.pt')
        self.writer = SummaryWriter(tb_dir)
        self.out_json = os.path.join(self.log_dir, 'stat.json')
        self.fixed_test_noise = None

    def make_shifts(self, latent_dim):
        target_indices = torch.randint(
            0, self.p.directions_count, [self.p.batch_size], device=self.device)
        if self.p.shift_distribution == ShiftDistribution.NORMAL:
            shifts = torch.randn(target_indices.shape, device=self.device)
        elif self.p.shift_distribution == ShiftDistribution.UNIFORM:
            shifts = 2.0 * torch.rand(target_indices.shape, device=self.device) - 1.0

        shifts = self.p.shift_scale * shifts
        shifts[(shifts < self.p.min_shift) & (shifts > 0)] = self.p.min_shift
        shifts[(shifts > -self.p.min_shift) & (shifts < 0)] = -self.p.min_shift

        try:
            latent_dim[0]
            latent_dim = list(latent_dim)
        except Exception:
            latent_dim = [latent_dim]

        z_shift = torch.zeros([self.p.batch_size] + latent_dim, device=self.device)
        for i, (index, val) in enumerate(zip(target_indices, shifts)):
            z_shift[i][index] += val

        return target_indices, shifts, z_shift

    def log_train(self, step, should_print=True, stats=()):
        if should_print:
            out_text = '{}% [step {}]'.format(int(100 * step / self.p.n_steps), step)
            for named_value in stats:
                out_text += (' | {}: {:.2f}'.format(*named_value))
            print(out_text)
        for named_value in stats:
            self.writer.add_scalar(named_value[0], named_value[1], step)

        with open(self.out_json, 'w') as out:
            stat_dict = {named_value[0]: named_value[1] for named_value in stats}
            json.dump(stat_dict, out)

    def log_interpolation(self, G, deformator, step):
        noise = make_noise(1, G.dim_z, self.p.truncation).to(self.device)
        if self.fixed_test_noise is None:
            self.fixed_test_noise = noise.clone()
        for z, prefix in zip([noise, self.fixed_test_noise], ['rand', 'fixed']):
            fig = make_interpolation_chart(
                G, deformator, z=z, shifts_r=3 * self.p.shift_scale, shifts_count=3, dims_count=15,
                device=self.device, dpi=500)

            self.writer.add_figure('{}_deformed_interpolation'.format(prefix), fig, step)
            fig_to_image(fig).convert("RGB").save(
                os.path.join(self.images_dir, '{}_{}.jpg'.format(prefix, step)))

    def start_from_checkpoint(self, deformator, shift_predictor):
        step = 0
        if os.path.isfile(self.checkpoint):
            state_dict = torch.load(self.checkpoint)
            step = state_dict['step']
            deformator.load_state_dict(state_dict['deformator'])
            shift_predictor.load_state_dict(state_dict['shift_predictor'])
            print('starting from step {}'.format(step))
        return step

    def save_checkpoint(self, deformator, shift_predictor, step):
        state_dict = {
            'step': step,
            'deformator': deformator.state_dict(),
            'shift_predictor': shift_predictor.state_dict(),
        }
        torch.save(state_dict, self.checkpoint)

    def save_models(self, deformator, shift_predictor, step):
        torch.save(deformator.state_dict(),
                   os.path.join(self.models_dir, 'deformator_{}.pt'.format(step)))
        torch.save(shift_predictor.state_dict(),
                   os.path.join(self.models_dir, 'shift_predictor_{}.pt'.format(step)))

    def log_accuracy(self, G, deformator, shift_predictor, step):
        deformator.eval()
        shift_predictor.eval()

        accuracy = validate_classifier(G, deformator, shift_predictor, trainer=self, device=self.device)
        self.writer.add_scalar('accuracy', accuracy.item(), step)

        deformator.train()
        shift_predictor.train()
        return accuracy

    def log(self, G, deformator, shift_predictor, step, avgs):
        if step % self.p.steps_per_log == 0:
            self.log_train(step, True, [avg.flush() for avg in avgs])

        if step % self.p.steps_per_img_log == 0:
            self.log_interpolation(G, deformator, step)

        if step % self.p.steps_per_backup == 0 and step > 0:
            self.save_checkpoint(deformator, shift_predictor, step)
            accuracy = self.log_accuracy(G, deformator, shift_predictor, step)
            print('Step {} accuracy: {:.3}'.format(step, accuracy.item()))

        if step % self.p.steps_per_save == 0 and step > 0:
            self.save_models(deformator, shift_predictor, step)

    def train(self, G, deformator, shift_predictor):
        G.to(self.device).eval()
        deformator.to(self.device).train()
        shift_predictor.to(self.device).train()

        deformator_opt = torch.optim.Adam(deformator.parameters(), lr=self.p.deformator_lr) if deformator.type not in [
            DeformatorType.ID, DeformatorType.RANDOM] else None
        shift_predictor_opt = torch.optim.Adam(
            shift_predictor.parameters(), lr=self.p.shift_predictor_lr)

        avgs = MeanTracker('percent'), MeanTracker('loss'), MeanTracker('direction_loss'), MeanTracker('shift_loss')
        avg_correct_percent, avg_loss, avg_label_loss, avg_shift_loss = avgs

        recovered_step = self.start_from_checkpoint(deformator, shift_predictor)

        for step in range(recovered_step, self.p.n_steps, 1):
            G.zero_grad()
            deformator.zero_grad()
            shift_predictor.zero_grad()

            z = make_noise(self.p.batch_size, G.dim_z, self.p.truncation).to(self.device)
            target_indices, shifts, basis_shift = self.make_shifts(deformator.input_dim)

            # Deformation
            shift = deformator(basis_shift)

            imgs = G(z)
            imgs_shifted = G.gen_shifted(z, shift)

            logits, shift_prediction = shift_predictor(imgs, imgs_shifted)
            logit_loss = self.p.label_weight * self.cross_entropy(logits, target_indices)
            shift_loss = self.p.shift_weight * torch.mean(torch.abs(shift_prediction - shifts))

            # total loss
            loss = logit_loss + shift_loss
            loss.backward()

            if deformator_opt is not None:
                deformator_opt.step()
            shift_predictor_opt.step()

            # update statistics trackers
            avg_correct_percent.add(torch.mean(
                (torch.argmax(logits, dim=1) == target_indices).to(torch.float32)).detach())
            avg_loss.add(loss.item())
            avg_label_loss.add(logit_loss.item())
            avg_shift_loss.add(shift_loss)

            self.log(G, deformator, shift_predictor, step, avgs)


@torch.no_grad()
def validate_classifier(G, deformator, shift_predictor, params_dict=None, trainer=None, device='cpu'):
    n_steps = 100
    if trainer is None:
        trainer = Trainer(params=Params(**params_dict), verbose=False)

    percents = torch.empty([n_steps])
    for step in range(n_steps):
        z = make_noise(trainer.p.batch_size, G.dim_z, trainer.p.truncation).to(device)
        target_indices, shifts, basis_shift = trainer.make_shifts(deformator.input_dim)

        imgs = G(z)
        imgs_shifted = G.gen_shifted(z, deformator(basis_shift))

        logits, _ = shift_predictor(imgs, imgs_shifted)
        percents[step] = (torch.argmax(logits, dim=1) == target_indices).to(torch.float32).mean()

    return percents.mean()