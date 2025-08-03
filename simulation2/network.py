import math

import torch
import torch.nn as nn


def init_weights(m):
    if isinstance(m, nn.Linear):
        if hasattr(m, 'weight') and m.weight is not None:
            nn.init.xavier_uniform_(m.weight)
        if hasattr(m, 'bias') and m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.LayerNorm):
        nn.init.ones_(m.weight)
        nn.init.zeros_(m.bias)


class TimeEmbedding(nn.Module):
    def __init__(self, embed_dim: int):
        super().__init__()
        self.embed_dim = embed_dim

    def forward(self, t):
        half_dim = self.embed_dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device) * -emb)
        emb = t[:, None] * emb[None, :]
        return torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)


class ResidualBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.block = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.SiLU(),
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
        )
        self.activation = nn.SiLU()

    def forward(self, x):
        return self.activation(x + self.block(x))


# Diffusion

class MLP(nn.Module):
    def __init__(self,
                 input_dim,
                 cond_dim,
                 time_embed_dim=128,
                 hidden_dim=16,
                 n_blocks=1
                 ):
        super().__init__()
        self.input_dim = input_dim
        self.cond_dim = cond_dim
        self.time_embed_dim = time_embed_dim
        self.hidden_dim = hidden_dim
        self.n_blocks = n_blocks
        self.time_embedding = TimeEmbedding(time_embed_dim)
        total_input_dim = input_dim + cond_dim + time_embed_dim

        self.input_layer = nn.Sequential(
            nn.Linear(total_input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU()
        )
        self.res_blocks = nn.Sequential(*[ResidualBlock(hidden_dim) for _ in range(n_blocks)])
        self.output_layer = nn.Linear(hidden_dim, input_dim)
        self.apply(init_weights)

    def forward(self, x, t, y):
        y, x = y.float(), x.float()
        t = t.float().flatten()
        t_emb = self.time_embedding(t)
        h = torch.cat([x, y, t_emb], dim=-1)
        h = self.input_layer(h)
        h = self.res_blocks(h)
        return self.output_layer(h).float()


# VAE Encoder & Decoder

class EncoderMLP(nn.Module):
    def __init__(self,
                 data_tim,
                 cond_dim,
                 latent_dim,
                 hidden_dim=16,
                 n_blocks=1
                 ):
        super().__init__()
        self.input_layer = nn.Sequential(
            nn.Linear(data_tim + cond_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU()
        )
        self.res_blocks = nn.Sequential(*[ResidualBlock(hidden_dim) for _ in range(n_blocks)])
        self.output_layer_mean = nn.Linear(hidden_dim, latent_dim)
        self.output_layer_var = nn.Linear(hidden_dim, latent_dim)
        self.apply(init_weights)

    def forward(self, x, y):
        x, y = x.float(), y.float()
        h = torch.cat([x, y], dim=-1)
        h = self.input_layer(h)
        h = self.res_blocks(h)
        mean = self.output_layer_mean(h).float()
        log_var = self.output_layer_var(h).float()
        return mean, log_var


class DecoderMLP(nn.Module):
    def __init__(self,
                 data_dim,
                 cond_dim,
                 latent_dim,
                 hidden_dim=16,
                 n_blocks=1
                 ):
        super().__init__()
        self.input_layer = nn.Sequential(
            nn.Linear(latent_dim + cond_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU()
        )
        self.res_blocks = nn.Sequential(*[ResidualBlock(hidden_dim) for _ in range(n_blocks)])
        self.output_layer = nn.Linear(hidden_dim, data_dim)
        self.apply(init_weights)

    def forward(self, x, y):
        x, y = x.float(), y.float()
        h = torch.cat([x, y], dim=-1)
        h = self.input_layer(h)
        h = self.res_blocks(h)
        h = self.output_layer(h)
        return h


class VAE(nn.Module):
    def __init__(self, data_dim, cond_dim, latent_dim):
        super().__init__()
        self.encoder = EncoderMLP(data_dim, cond_dim, latent_dim)
        self.decoder = DecoderMLP(data_dim, cond_dim, latent_dim)

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
        reconstruction_loss = torch.sum((x_hat - x) ** 2, dim=1)
        kld = - 0.5 * torch.sum(1 + log_var - mean ** 2 - torch.exp(log_var), dim=1)  # KL divergence
        loss = reconstruction_loss + kld
        return loss


# GAN Generator & Critic

class GeneratorMLP(nn.Module):
    def __init__(self,
                 data_tim,
                 cond_dim,
                 latent_dim,
                 hidden_dim=16,
                 n_blocks=1
                 ):
        super().__init__()
        self.input_layer = nn.Sequential(
            nn.Linear(latent_dim + cond_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU()
        )
        self.res_blocks = nn.Sequential(*[ResidualBlock(hidden_dim) for _ in range(n_blocks)])
        self.output_layer = nn.Linear(hidden_dim, data_tim)
        self.apply(init_weights)

    def forward(self, x, y):
        x, y = x.float(), y.float()
        h = torch.cat([x, y], dim=-1)
        h = self.input_layer(h)
        h = self.res_blocks(h)
        h = self.output_layer(h)
        return h


class CriticMLP(nn.Module):
    def __init__(self,
                 data_tim,
                 cond_dim,
                 hidden_dim=16,
                 n_blocks=1
                 ):
        super().__init__()
        self.input_layer = nn.Sequential(
            nn.Linear(data_tim + cond_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU()
        )
        self.res_blocks = nn.Sequential(*[ResidualBlock(hidden_dim) for _ in range(n_blocks)])
        self.output_layer = nn.Linear(hidden_dim, 1)
        self.apply(init_weights)

    def forward(self, x, y):
        x, y = x.float(), y.float()
        h = torch.cat([x, y], dim=-1)
        h = self.input_layer(h)
        h = self.res_blocks(h)
        return self.output_layer(h).float()


class GAN(nn.Module):
    def __init__(self, data_dim, cond_dim, latent_dim):
        super().__init__()
        self.latent_dim = latent_dim
        self.generator = GeneratorMLP(data_dim, cond_dim, latent_dim)
        self.critic = CriticMLP(data_dim, cond_dim)

    def forward(self, z, y):
        return self.generator(z, y)

    def get_gradient_penalty(self, x, z, y):
        eps = torch.rand((x.shape[0], 1), device=x.device)
        interp = torch.autograd.Variable(x * eps + (1 - eps) * z, requires_grad=True)
        pred = self.critic(interp, y)
        grads = torch.autograd.grad(pred, interp, grad_outputs=(torch.ones_like(pred, device=pred.device)),
                                    create_graph=True, retain_graph=True)[0]
        return (grads.norm(2, dim=1, keepdim=True) - 1) ** 2

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


if __name__ == '__main__':
    net = MLP(data_dim=1, cond_dim=2)
    print(net)
