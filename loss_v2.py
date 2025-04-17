import torch
import torch.nn as nn
import torch.nn.functional as F
import lpips

# Charbonnier Loss
class CharbonnierLoss(nn.Module):
    def __init__(self, eps=1e-6):
        super(CharbonnierLoss, self).__init__()
        self.eps = eps

    def forward(self, y_pred, y_true):
        return torch.mean(torch.sqrt((y_pred - y_true) ** 2 + self.eps))


# Static Hybrid Loss for Short Training
class HybridLoss(nn.Module):
    def __init__(self, device='cuda'):
        super(HybridLoss, self).__init__()
        self.lpips_loss = lpips.LPIPS(net='vgg').to(device)
        self.charbonnier = CharbonnierLoss()

    def ssim_loss(self, y_true, y_pred, val_range=1.0, window_size=11):
        C1 = (0.01 * val_range) ** 2
        C2 = (0.03 * val_range) ** 2

        mu1 = F.avg_pool2d(y_true, window_size, stride=1, padding=window_size // 2)
        mu2 = F.avg_pool2d(y_pred, window_size, stride=1, padding=window_size // 2)

        sigma1_sq = F.avg_pool2d(y_true ** 2, window_size, stride=1, padding=window_size // 2) - mu1 ** 2
        sigma2_sq = F.avg_pool2d(y_pred ** 2, window_size, stride=1, padding=window_size // 2) - mu2 ** 2
        sigma12 = F.avg_pool2d(y_true * y_pred, window_size, stride=1, padding=window_size // 2) - mu1 * mu2

        ssim_map = ((2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)) / \
                   ((mu1 ** 2 + mu2 ** 2 + C1) * (sigma1_sq + sigma2_sq + C2))
        return 1 - ssim_map.mean()

    def forward(self, y_pred, y_true):
        # Ensure LPIPS inputs are 3-channel in [-1, 1]
        if y_pred.size(1) == 1:
            y_pred_lpips = y_pred.repeat(1, 3, 1, 1)
            y_true_lpips = y_true.repeat(1, 3, 1, 1)
        else:
            y_pred_lpips = y_pred
            y_true_lpips = y_true

        loss_lpips = self.lpips_loss(y_pred_lpips, y_true_lpips).mean()
        loss_ssim = self.ssim_loss(y_true, y_pred)
        loss_charb = self.charbonnier(y_pred, y_true)

        # Fixed weights for short training
        total = 0.4 * loss_lpips + 0.3 * loss_ssim + 0.3 * loss_charb
        return total
