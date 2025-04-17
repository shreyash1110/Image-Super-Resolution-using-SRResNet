import torch
import torch.nn as nn

# Squeeze-and-Excitation Block
class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super(SEBlock, self).__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // reduction, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        scale = self.se(x)
        return x * scale

# Residual Block with SE Attention
class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.PReLU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        )
        self.attention = SEBlock(channels)

    def forward(self, x):
        out = self.block(x)
        out = self.attention(out)
        return x + out

# Upsample Block using PixelShuffle
class UpsampleBlock(nn.Module):
    def __init__(self, in_channels, scale_factor=2):
        super(UpsampleBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, in_channels * (scale_factor ** 2), kernel_size=3, padding=1),
            nn.PixelShuffle(scale_factor),
            nn.PReLU()
        )

    def forward(self, x):
        return self.block(x)

# SRResNet with SE Attention
class SRResNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, base_channels=64, num_res_blocks=16):
        super(SRResNet, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, kernel_size=9, padding=4),
            nn.PReLU()
        )

        self.res_blocks = nn.Sequential(*[
            ResidualBlock(base_channels) for _ in range(num_res_blocks)
        ])

        self.conv2 = nn.Conv2d(base_channels, base_channels, kernel_size=3, padding=1)

        self.upsample = nn.Sequential(
            UpsampleBlock(base_channels, scale_factor=2),
            UpsampleBlock(base_channels, scale_factor=2)
        )

        self.conv3 = nn.Conv2d(base_channels, out_channels, kernel_size=9, padding=4)

    def forward(self, x):
        out1 = self.conv1(x)
        out = self.res_blocks(out1)
        out = self.conv2(out)
        out = out + out1
        out = self.upsample(out)
        out = self.conv3(out)
        return out
