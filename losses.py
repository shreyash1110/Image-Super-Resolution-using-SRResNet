import torch
import torch.nn.functional as F
import torchvision.models as models


def ssim_loss(y_true, y_pred, window_size=11, size_average=True, val_range=1.0):
    C1 = (0.01 * val_range) ** 2
    C2 = (0.03 * val_range) ** 2

    def gaussian(window_size, sigma):
        x = torch.arange(window_size).float()
        gauss = torch.exp(-(x - window_size // 2) ** 2 / (2 * sigma ** 2))
        return gauss / gauss.sum()

    def create_window(window_size, sigma):
        _1D = gaussian(window_size, sigma).unsqueeze(1)
        _2D = _1D.mm(_1D.t()).unsqueeze(0).unsqueeze(0)
        return _2D.expand(y_true.size(1), 1, window_size, window_size).contiguous()

    window = create_window(window_size, 1.5).to(y_true.device)

    pad = window_size // 2
    y_true = F.pad(y_true, (pad, pad, pad, pad), mode='reflect')
    y_pred = F.pad(y_pred, (pad, pad, pad, pad), mode='reflect')

    mu1 = F.conv2d(y_true, window, padding=0, groups=y_true.size(1))
    mu2 = F.conv2d(y_pred, window, padding=0, groups=y_pred.size(1))

    mu1_sq, mu2_sq = mu1.pow(2), mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(y_true * y_true, window, padding=0, groups=y_true.size(1)) - mu1_sq
    sigma2_sq = F.conv2d(y_pred * y_pred, window, padding=0, groups=y_pred.size(1)) - mu2_sq
    sigma12 = F.conv2d(y_true * y_pred, window, padding=0, groups=y_true.size(1)) - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return 1 - ssim_map.mean()
    else:
        return 1 - ssim_map.mean([1, 2, 3])


class VGGPerceptualLoss(torch.nn.Module):
    def __init__(self, layers=(3, 8, 15), device='cpu'):
        super(VGGPerceptualLoss, self).__init__()
        vgg = models.vgg16(pretrained=True).features[:max(layers)+1].to(device).eval()
        for param in vgg.parameters():
            param.requires_grad = False
        self.vgg = vgg
        self.layers = layers

    def forward(self, y_true, y_pred):
        loss = 0
        for i, layer in enumerate(self.vgg):
            y_true = layer(y_true)
            y_pred = layer(y_pred)
            if i in self.layers:
                loss += F.mse_loss(y_pred, y_true)
        return loss


def total_loss(y_true, y_pred, perceptual_model, lambda_mse=1.0, lambda_perceptual=0.1, lambda_ssim=0.1):
    mse = F.mse_loss(y_pred, y_true)
    perceptual = perceptual_model(y_true, y_pred)
    ssim = ssim_loss(y_true, y_pred)
    return lambda_mse * mse + lambda_perceptual * perceptual + lambda_ssim * ssim


## Example usgae

from losses import total_loss, VGGPerceptualLoss

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
perceptual_model = VGGPerceptualLoss(device=device)

loss = total_loss(hr_images, sr_images, perceptual_model)

