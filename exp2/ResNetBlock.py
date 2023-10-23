import torch
import torch.nn as nn


class ResNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, activeFunc=nn.ReLU()):
        super(ResNetBlock, self).__init__()
        self.activeFunc = activeFunc
        self.main_branch = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=stride,
                      padding=1),
            nn.BatchNorm2d(out_channels),
            self.activeFunc(),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1,
                      padding=1),
            nn.BatchNorm2d(out_channels)
        )

        self.short_cut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.short_cut = nn.Sequential(
                nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=stride, padding=0),
                nn.BatchNorm2d(out_channels)
            )

        self.FinalRelu = nn.Sequential(
            self.activeFunc()
        )

    def forward(self, x):
        output = self.main_branch(x)

        output1 = self.short_cut(x)

        output += output1

        return self.FinalRelu(output)
