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


class Critic(nn.Module):
    def __init__(self, num_classes=10, channels=[64, 128, 256]):
        super().__init__()

        self.num_classes = num_classes

        self.conv1 = ConvBlock(1+num_classes, channels[0], 4, 2, 1) # H -> H/2, 14
        self.conv2 = ConvBlock(channels[0], channels[1], 4, 2, 1) # H/2 -> H/4, 7
        self.conv3 = ConvBlock(channels[1], channels[2], 3, 2, 1) # H/4 -> (H+4)/8, 4
        self.conv4 = nn.Conv2d(channels[2], 1, 4, 1, 0) # (H+4)/8 -> (H-20)/8, 1

    def forward(self, x, y):
        _, _, h, w = x.shape
        x = torch.cat([x, y[..., None, None].repeat(1, 1, h, w)], dim=1)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = x.view(-1, 1).squeeze(1)
        return x


class Generator(nn.Module):
    def __init__(self, num_classes=10, latent_dim=100, channels=[64, 128, 256]):
        super().__init__()

        self.num_classes = num_classes
        self.latent_dim = latent_dim

        self.conv1 = ConvBlock(latent_dim+num_classes, channels[2], 4, 1, 0, transposed=True) # h -> h+3, 4
        self.conv2 = ConvBlock(channels[2], channels[1], 3, 2, 1, transposed=True) # h+3 -> 2h+5, 7
        self.conv3 = ConvBlock(channels[1], channels[0], 4, 2, 1, transposed=True) # 2h+5 -> 4h+10, 14
        self.conv4 = nn.ConvTranspose2d(channels[0], 1, 4, 2, 1) # 4h+10 -> 8h+20, 28

    def forward(self, z, y):
        x = torch.cat([z, y], dim=1)[..., None, None]
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = torch.tanh(x)
        return x


class GAN(nn.Module):
    def __init__(self, num_classes=10, latent_dim=200):
        super().__init__()
        self.latent_dim = latent_dim
        self.generator = Generator(num_classes, latent_dim)
        self.critic = Critic(num_classes)

    def forward(self, z, y):
        return self.generator(z, y)

    def get_gradient_penalty(self, x, z, y):
        eps = torch.rand((x.shape[0], 1, 1, 1), device=x.device)
        interp = torch.autograd.Variable(x * eps + (1 - eps) * z, requires_grad=True)
        pred = self.critic(interp, y)
        grads = torch.autograd.grad(pred, interp, grad_outputs=(torch.ones_like(pred, device=pred.device)),
                                    create_graph=True, retain_graph=True)[0]
        return (grads.norm(2, dim=(1, 2, 3), keepdim=True) - 1) ** 2

    def critic_loss_fn(self, x, y):
        x, y = x.float(), y.float()
        target_pred = self.critic(x, y)
        z = torch.randn((x.shape[0], self.latent_dim), device=x.device)
        fake = self.generator(z, y)
        fake_pred = self.critic(fake.detach(), y)
        adv_loss = fake_pred - target_pred
        gp = self.get_gradient_penalty(x, fake.detach(), y)
        return adv_loss, gp

    def generator_loss_fn(self, x, y):
        x, y = x.float(), y.float()
        z = torch.randn((x.shape[0], self.latent_dim), device=x.device)
        fake = self.generator(z, y)
        fake_pred = self.critic(fake, y)
        return -fake_pred

