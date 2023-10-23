import torch.nn as nn
from config import *
import torch

latent_size = 64
n_channel = 3
n_g_feature = 64
n_d_feature = 64
class Discriminator(nn.Module):
    def __init__(self, channels=DEFAULTGANCHANNELS, batchsize=DEFAULTGANBATCHSIZE, imagesize=DEFAULTGANIMAGESIZE, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.channels = channels
        self.batchsize = batchsize
        self.imagesize = imagesize
        self.conv = self._make_conv_layer()
        self.linear = nn.Sequential(nn.Flatten(),
                                    nn.Linear(
                                        self.channels[0]*self.imagesize[0]*self.imagesize[1], 1),
                                    nn.Sigmoid())

    def _make_conv_layer(self):
        layer = []
        for i in range(len(self.channels)-1):
            in_channel = self.channels[i]
            out_channel = self.channels[i+1]
            layer.append(nn.Conv2d(in_channel, out_channel, 3, 1, 1))
            layer.append(nn.PReLU())
        in_channel = self.channels[-1]
        out_channel = self.channels[0]
        layer.append(nn.Conv2d(in_channel, out_channel, 3, 1, 1))
        layer.append(nn.PReLU())
        return nn.Sequential(*layer)

    def forward(self, x):
        x = self.conv(x)
        x = self.linear(x)
        return x


if __name__ == "__main__":
    discrimintor = Discriminator()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # device='cpu'
    discrimintor.to(device)
    print(device)
    # print(discrimintor)
    from torchsummary import summary
    x = torch.rand((2, 3, 32, 32))
    x = x.to(device)
    y = discrimintor(x)
    print(y)
    summary(discrimintor, (3, 32, 32), device=device)
