import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, transposed=False):
        super().__init__()

        if transposed:
            self.conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        else:
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.norm = nn.GroupNorm(32, out_channels)
        self.act = nn.SiLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)
        return x


class Encoder(nn.Module):
    def __init__(self, latent_dim=200, channels=[64, 128, 256]):
        super().__init__()

        self.latent_dim = latent_dim

        self.conv1 = ConvBlock(2, channels[0], 4, 2, 1) # H -> H/2, 14
        self.conv2 = ConvBlock(channels[0], channels[1], 4, 2, 1) # H/2 -> H/4, 7
        self.conv3 = ConvBlock(channels[1], channels[2], 3, 2, 1) # H/4 -> (H+4)/8, 4
        self.conv4 = nn.Conv2d(channels[2], latent_dim, 4, 1, 0) # (H+4)/8 -> (H-20)/8, 1
        self.output_layer_mean = nn.Linear(latent_dim, latent_dim)
        self.output_layer_var = nn.Linear(latent_dim, latent_dim)


    def forward(self, x, y):
        x = torch.cat([x, y], dim=1)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = x.view(-1, self.latent_dim)
        mean = self.output_layer_mean(x).float()
        log_var = self.output_layer_var(x).float()
        return mean, log_var

class Decoder(nn.Module):
    def __init__(self, latent_dim=200, channels=[64, 128, 256]):
        super().__init__()

        self.latent_dim = latent_dim

        self.conv1 = ConvBlock(latent_dim * 2, channels[2], 4, 1, 0, transposed=True) # h -> h+3, 4
        self.conv2 = ConvBlock(channels[2] * 2, channels[1], 3, 2, 1, transposed=True) # h+3 -> 2h+5, 7
        self.conv3 = ConvBlock(channels[1] * 2, channels[0], 4, 2, 1, transposed=True) # 2h+5 -> 4h+10, 14
        self.conv4 = nn.ConvTranspose2d(channels[0] * 2, 1, 4, 2, 1) # 4h+10 -> 8h+20, 28

        self.y_conv1 = ConvBlock(1, channels[0], 4, 2, 1) # H -> H/2, 14
        self.y_conv2 = ConvBlock(channels[0], channels[1], 4, 2, 1) # H/2 -> H/4, 7
        self.y_conv3 = ConvBlock(channels[1], channels[2], 3, 2, 1) # H/4 -> (H+4)/8, 4
        self.y_conv4 = nn.Conv2d(channels[2], latent_dim, 4, 1, 0) # (H+4)/8 -> (H-20)/8, 1


    def forward(self, z, y):
        y1 = self.y_conv1(y)
        y2 = self.y_conv2(y1)
        y3 = self.y_conv3(y2)
        y4 = self.y_conv4(y3)

        x = self.conv1(torch.cat([z[..., None, None], y4], dim=1))
        x = self.conv2(torch.cat([x, y3], dim=1))
        x = self.conv3(torch.cat([x, y2], dim=1))
        x = self.conv4(torch.cat([x, y1], dim=1))
        x = torch.tanh(x)
        return x


class VAE(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.encoder = Encoder(latent_dim)
        self.decoder = Decoder(latent_dim)

    @staticmethod
    def reparameterization(mean, var):
        epsilon = torch.randn_like(var, device=var.device)
        z = mean + var * epsilon  # reparameterization trick
        return z

    def forward(self, x, y):
        mean, log_var = self.encoder(x, y)
        z = self.reparameterization(mean, torch.exp(0.5 * log_var))
        x_hat = self.decoder(z, y)
        return x_hat, mean, log_var

    def loss_fn(self, x, y):
        x_hat, mean, log_var = self.forward(x, y)
        reconstruction_loss = torch.sum((x_hat - x) ** 2, dim=(1, 2, 3))
        kld = - 0.5 * torch.sum(1 + log_var - mean ** 2 - torch.exp(log_var), dim=1)  # KL divergence
        loss = reconstruction_loss + kld
        return loss
