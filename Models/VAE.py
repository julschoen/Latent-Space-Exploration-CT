import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, Independent
from torch.distributions.kl import kl_divergence as KLD
from Models.utils import gan_with_shift


# Residual down sampling block for the encoder
# Average pooling is used to perform the downsampling
class Res_down(nn.Module):
    def __init__(self, channel_in, channel_out, scale=2):
        super(Res_down, self).__init__()

        self.conv1 = nn.Conv2d(channel_in, channel_out // 2, 3, 1, 1)
        self.BN1 = nn.BatchNorm2d(channel_out // 2)
        self.conv2 = nn.Conv2d(channel_out // 2, channel_out, 3, 1, 1)
        self.BN2 = nn.BatchNorm2d(channel_out)

        self.conv3 = nn.Conv2d(channel_in, channel_out, 3, 1, 1)

        self.AvePool = nn.AvgPool2d(scale, scale)

    def forward(self, x):
        skip = self.conv3(self.AvePool(x))

        x = F.rrelu(self.BN1(self.conv1(x)))
        x = self.AvePool(x)
        x = self.BN2(self.conv2(x))

        x = F.rrelu(x + skip)
        return x


# Residual up sampling block for the decoder
# Nearest neighbour is used to perform the upsampling
class Res_up(nn.Module):
    def __init__(self, channel_in, channel_out, scale=2):
        super(Res_up, self).__init__()

        self.conv1 = nn.Conv2d(channel_in, channel_out // 2, 3, 1, 1)
        self.BN1 = nn.BatchNorm2d(channel_out // 2)
        self.conv2 = nn.Conv2d(channel_out // 2, channel_out, 3, 1, 1)
        self.BN2 = nn.BatchNorm2d(channel_out)

        self.conv3 = nn.Conv2d(channel_in, channel_out, 3, 1, 1)

        self.UpNN = nn.Upsample(scale_factor=scale, mode="nearest")

    def forward(self, x):
        skip = self.conv3(self.UpNN(x))

        x = F.rrelu(self.BN1(self.conv1(x)))
        x = self.UpNN(x)
        x = self.BN2(self.conv2(x))

        x = F.rrelu(x + skip)
        return x


class Encoder(nn.Module):
    def __init__(self, channels, ch=64, z=100, device='cpu'):
        super(Encoder, self).__init__()
        self.conv1 = Res_down(channels, ch)  # 64
        self.conv2 = Res_down(ch, 2 * ch)  # 32
        self.conv3 = Res_down(2 * ch, 4 * ch)  # 16
        self.conv4 = Res_down(4 * ch, 8 * ch)  # 8
        self.conv5 = Res_down(8 * ch, 8 * ch)  # 4
        self.conv_mu = nn.Conv2d(8 * ch, z, 4, 4)  # 2
        self.conv_logvar = nn.Conv2d(8 * ch, z, 4, 4)  # 2
        self.mu_0 = torch.zeros((1, z)).to(device)
        self.sigma_0 = torch.ones((1, z)).to(device)

    def sample(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x, Train=True):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)

        if Train:
            mu = self.conv_mu(x)
            log_var = self.conv_logvar(x)

            # log_var = softplus(log_var)
            sigma = torch.exp(log_var / 2)

            # Instantiate a diagonal Gaussian with mean=mu, std=sigma
            # This is the approximate latent distribution q(z|x)
            posterior = Independent(Normal(loc=mu, scale=sigma), 1)
            z = posterior.rsample()

            # Instantiate a standard Gaussian with mean=mu_0, std=sigma_0
            # This is the prior distribution p(z)
            prior = Independent(Normal(loc=self.mu_0, scale=self.sigma_0), 1)

            # Estimate the KLD between q(z|x)|| p(z)
            kl = KLD(posterior, prior).mean()

        else:
            mu = self.conv_mu(x)
            log_var = self.conv_logvar(x)
            sigma = torch.exp(log_var / 2)

            posterior = Independent(Normal(loc=mu, scale=sigma), 1)
            z = posterior.rsample()
            kl = None
            
        return kl, z


# Decoder block
# Built to be a mirror of the encoder block
class Decoder(nn.Module):
    def __init__(self, channels, ch=64, z=100):
        super(Decoder, self).__init__()
        self.conv1 = Res_up(z, ch * 16)
        self.conv12 = Res_up(ch * 16, ch * 8)
        self.conv2 = Res_up(ch * 8, ch * 8)
        self.conv3 = Res_up(ch * 8, ch * 4)
        self.conv4 = Res_up(ch * 4, ch * 2)
        self.conv5 = Res_up(ch * 2, ch)
        self.conv6 = Res_up(ch, ch // 2)
        self.conv7 = nn.Conv2d(ch // 2, channels, 3, 1, 1)
        self.act = nn.Tanh()
        self.dim_z = z

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv12(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)

        return self.act(x)

    # VAE network, uses the above encoder and decoder blocks


class VAE(nn.Module):
    def __init__(self, channel_in=1, z=32, device='cpu'):
        super(VAE, self).__init__()
        """Res VAE Network
        channel_in  = number of channels of the image 
        z = the number of channels of the latent representation (for a 64x64 image this is the size of the latent vector)"""

        self.encoder = Encoder(channel_in, z=z, device=device)
        self.decoder = Decoder(channel_in, z=z)

    def forward(self, x, Train=True):
        kl, z = self.encoder(x, Train)
        recon = self.decoder(z)
        return kl, recon


@gan_with_shift
def make_vae(gan_dir):
    dec = Decoder(channels=1)
    checkpoint = torch.load(gan_dir)
    dec.load_state_dict(checkpoint['state_dict'])
    dec = dec.eval()
    return dec
