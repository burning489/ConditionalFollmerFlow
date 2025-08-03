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
  def __init__(self, channels=[32, 64, 128, 256], embed_dim=256):
    super().__init__()
    self.time_embed = nn.Sequential(
        TimeEmbedding(embed_dim=embed_dim),
        nn.Linear(embed_dim, embed_dim),
        nn.SiLU(),
        nn.Linear(embed_dim, embed_dim)
        )
    self.conv0 = nn.Conv2d(2, channels[0], 3, 1, 1, bias=False)
    self.conv1 = nn.Conv2d(channels[0], channels[0], 4, 2, 1, bias=False)
    self.dense1 = nn.Linear(embed_dim, channels[0])
    self.gnorm1 = nn.GroupNorm(4, num_channels=channels[0])
    self.conv2 = nn.Conv2d(channels[0], channels[1], 4, 2, 1, bias=False)
    self.dense2 = nn.Linear(embed_dim, channels[1])
    self.gnorm2 = nn.GroupNorm(8, num_channels=channels[1])
    self.conv3 = nn.Conv2d(channels[1], channels[2], 3, 2, 1, bias=False)
    self.dense3 = nn.Linear(embed_dim, channels[2])
    self.gnorm3 = nn.GroupNorm(16, num_channels=channels[2])
    self.conv4 = nn.Conv2d(channels[2], channels[3], 4, 2, 1, bias=False)
    self.dense4 = nn.Linear(embed_dim, channels[3])
    self.gnorm4 = nn.GroupNorm(32, num_channels=channels[3])    

    self.tconv4 = nn.ConvTranspose2d(channels[3], channels[2], 4, 2, 1, bias=False)
    self.dense5 = nn.Linear(embed_dim, channels[2])
    self.tgnorm4 = nn.GroupNorm(16, num_channels=channels[2])
    self.tconv3 = nn.ConvTranspose2d(channels[2] + channels[2], channels[1], 3, 2, 1, bias=False)
    self.dense6 = nn.Linear(embed_dim, channels[1])
    self.tgnorm3 = nn.GroupNorm(8, num_channels=channels[1])
    self.tconv2 = nn.ConvTranspose2d(channels[1] + channels[1], channels[0], 4, 2, 1, bias=False)
    self.dense7 = nn.Linear(embed_dim, channels[0])
    self.tgnorm2 = nn.GroupNorm(4, num_channels=channels[0])
    self.tconv1 = nn.ConvTranspose2d(channels[0] + channels[0], 1, 4, 2, 1)
    self.act = nn.SiLU()
    self.apply(kaiming_init)
  
  def forward(self, x, t, y): 
    x = self.conv0(torch.cat([x, y], dim=1))
    embed = self.time_embed(t.squeeze())

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


if __name__ == "__main__":
    x = torch.randn(16, 1, 28, 28)
    y = torch.randn(16, 10)
    t = torch.randint(0, 1000, (16,))
    model = UNet()
    y = model(x, t, y)
    print(y.shape)