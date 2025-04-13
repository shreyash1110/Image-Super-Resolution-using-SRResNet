import torch
import torch.nn as nn

# Residual Block without BatchNorm
class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.PReLU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        )

    def forward(self, x):
        return x + self.block(x)

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

# SRResNet Model
class SRResNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, base_channels=64, num_res_blocks=16):
        super(SRResNet, self).__init__()

        # Initial conv
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, kernel_size=9, padding=4),
            nn.PReLU()
        )

        # Residual blocks
        res_blocks = [ResidualBlock(base_channels) for _ in range(num_res_blocks)]
        self.res_blocks = nn.Sequential(*res_blocks)

        # Second conv after residuals (no norm now)
        self.conv2 = nn.Conv2d(base_channels, base_channels, kernel_size=3, padding=1)

        # Upsampling (4x = 2x + 2x)
        self.upsample = nn.Sequential(
            UpsampleBlock(base_channels, scale_factor=2),
            UpsampleBlock(base_channels, scale_factor=2)
        )

        # Final output conv
        self.conv3 = nn.Conv2d(base_channels, out_channels, kernel_size=9, padding=4)

    def forward(self, x):
        out1 = self.conv1(x)
        out = self.res_blocks(out1)
        out = self.conv2(out)
        out = out + out1  # Residual skip connection
        out = self.upsample(out)
        out = self.conv3(out)
        return out
