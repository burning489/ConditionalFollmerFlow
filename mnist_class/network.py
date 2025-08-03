import torch
import torch.nn as nn
import math

def kaiming_init(m):
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
        nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)

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


class UNet(nn.Module):
  def __init__(self, num_classes=10, channels=[32, 64, 128, 256], embed_dim=256):
    super().__init__()
    self.time_embed = nn.Sequential(
        TimeEmbedding(embed_dim=embed_dim),
        nn.Linear(embed_dim, embed_dim),
        nn.SiLU(),
        nn.Linear(embed_dim, embed_dim)
        )
    self.cond_embed = nn.Sequential(
        nn.Linear(num_classes, embed_dim),
        nn.SiLU(),
        nn.Linear(embed_dim, embed_dim)
        )
    self.conv0 = nn.Conv2d(1, channels[0], 3, 1, 1, bias=False)
    self.conv1 = nn.Conv2d(channels[0], channels[0], 4, 2, 1, bias=False)
    self.dense1 = nn.Linear(embed_dim * 2, channels[0])
    self.gnorm1 = nn.GroupNorm(4, num_channels=channels[0])
    self.conv2 = nn.Conv2d(channels[0], channels[1], 4, 2, 1, bias=False)
    self.dense2 = nn.Linear(embed_dim * 2, channels[1])
    self.gnorm2 = nn.GroupNorm(8, num_channels=channels[1])
    self.conv3 = nn.Conv2d(channels[1], channels[2], 3, 2, 1, bias=False)
    self.dense3 = nn.Linear(embed_dim * 2, channels[2])
    self.gnorm3 = nn.GroupNorm(16, num_channels=channels[2])
    self.conv4 = nn.Conv2d(channels[2], channels[3], 4, 2, 1, bias=False)
    self.dense4 = nn.Linear(embed_dim * 2, channels[3])
    self.gnorm4 = nn.GroupNorm(32, num_channels=channels[3])    

    self.tconv4 = nn.ConvTranspose2d(channels[3], channels[2], 4, 2, 1, bias=False)
    self.dense5 = nn.Linear(embed_dim * 2, channels[2])
    self.tgnorm4 = nn.GroupNorm(16, num_channels=channels[2])
    self.tconv3 = nn.ConvTranspose2d(channels[2] + channels[2], channels[1], 3, 2, 1, bias=False)
    self.dense6 = nn.Linear(embed_dim * 2, channels[1])
    self.tgnorm3 = nn.GroupNorm(8, num_channels=channels[1])
    self.tconv2 = nn.ConvTranspose2d(channels[1] + channels[1], channels[0], 4, 2, 1, bias=False)
    self.dense7 = nn.Linear(embed_dim * 2, channels[0])
    self.tgnorm2 = nn.GroupNorm(4, num_channels=channels[0])
    self.tconv1 = nn.ConvTranspose2d(channels[0] + channels[0], 1, 4, 2, 1)
    self.act = nn.SiLU()
    self.apply(kaiming_init)
  
  def forward(self, x, t, y): 
    x = self.conv0(x)
    t_embed = self.time_embed(t.squeeze())
    y_embed = self.cond_embed(y)
    embed = torch.cat([t_embed, y_embed], 1)

    h1 = self.conv1(x)    
    h1 += self.dense1(embed)[..., None, None]
    h1 = self.gnorm1(h1)
    h1 = self.act(h1)
    h2 = self.conv2(h1)
    h2 += self.dense2(embed)[..., None, None]
    h2 = self.gnorm2(h2)
    h2 = self.act(h2)
    h3 = self.conv3(h2)
    h3 += self.dense3(embed)[..., None, None]
    h3 = self.gnorm3(h3)
    h3 = self.act(h3)
    h4 = self.conv4(h3)
    h4 += self.dense4(embed)[..., None, None]
    h4 = self.gnorm4(h4)
    h4 = self.act(h4)

    h = self.tconv4(h4)
    h += self.dense5(embed)[..., None, None]
    h = self.tgnorm4(h)
    h = self.act(h)
    h = self.tconv3(torch.cat([h, h3], dim=1))
    h += self.dense6(embed)[..., None, None]
    h = self.tgnorm3(h)
    h = self.act(h)
    h = self.tconv2(torch.cat([h, h2], dim=1))
    h += self.dense7(embed)[..., None, None]
    h = self.tgnorm2(h)
    h = self.act(h)
    h = self.tconv1(torch.cat([h, h1], dim=1))
    return h


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
    def __init__(self, num_classes, latent_dim):
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


class Encoder(nn.Module):
    def __init__(self, num_classes=10, latent_dim=100, channels=[64, 128, 256]):
        super().__init__()

        self.num_classes = num_classes
        self.latent_dim = latent_dim

        self.conv1 = ConvBlock(1+num_classes, channels[0], 4, 2, 1) # H -> H/2, 14
        self.conv2 = ConvBlock(channels[0], channels[1], 4, 2, 1) # H/2 -> H/4, 7
        self.conv3 = ConvBlock(channels[1], channels[2], 3, 2, 1) # H/4 -> (H+4)/8, 4
        self.conv4 = nn.Conv2d(channels[2], latent_dim, 4, 1, 0) # (H+4)/8 -> (H-20)/8, 1
        self.output_layer_mean = nn.Linear(latent_dim, latent_dim)
        self.output_layer_var = nn.Linear(latent_dim, latent_dim)


    def forward(self, x, y):
        _, _, h, w = x.shape
        x = torch.cat([x, y[..., None, None].repeat(1, 1, h, w)], dim=1)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = x.view(-1, self.latent_dim)
        mean = self.output_layer_mean(x).float()
        log_var = self.output_layer_var(x).float()
        return mean, log_var

class Decoder(nn.Module):
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


class VAE(nn.Module):
    def __init__(self, num_classes, latent_dim):
        super().__init__()
        self.encoder = Encoder(num_classes, latent_dim)
        self.decoder = Decoder(num_classes, latent_dim)

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


if __name__ == "__main__":
    x = torch.randn(16, 1, 28, 28)
    y = torch.randn(16, 10)
    t = torch.randint(0, 1000, (16,))
    model = UNet()
    y = model(x, t, y)
    print(y.shape)