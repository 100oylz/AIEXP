import torch.nn as nn
import torch
from torch.distributions import Normal, kl_divergence

from config import DEFAULTGANBATCHSIZE, DEFAULTGANCHANNELS, DEFAULTGANIMAGESIZE

latent_size = 64
n_channel = 3
n_g_feature = 64
class Generator(nn.Module):
    def __init__(self, channels=DEFAULTGANCHANNELS, batchsize=DEFAULTGANBATCHSIZE, imagesize=DEFAULTGANIMAGESIZE, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.channels = channels
        self.batchsize = batchsize
        self.imagesize = imagesize
        self.unflatten = nn.Unflatten(
            -1, (self.channels[0], self.imagesize[0], self.imagesize[1]))
        self.conv = self._make_convTranspose_layer()

    def _make_convTranspose_layer(self):
        layer = []
        for i in range(len(self.channels)-1):
            in_channel, out_channel = self.channels[i], self.channels[i+1]
            layer.append(nn.ConvTranspose2d(in_channels=in_channel, out_channels=out_channel,
                         kernel_size=3, stride=1, padding=1, output_padding=0))
            layer.append(nn.PReLU())
            layer.append(nn.BatchNorm2d(out_channel))
        in_channel = self.channels[-1]
        out_channel = self.channels[0]
        layer.append(nn.ConvTranspose2d(in_channels=in_channel, out_channels=out_channel,
                     kernel_size=3, stride=1, padding=1, output_padding=0))
        layer.append(nn.Sigmoid())
        return nn.Sequential(*layer)

    def forward(self,x):
        x = self.unflatten(x)
        x = self.conv(x)
        return x


if __name__ == '__main__':
    generator = Generator()
    x = generator()
    print(x.shape)
    print(generator)
