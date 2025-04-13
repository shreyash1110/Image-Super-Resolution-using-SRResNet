def psnr_batch(y_true, y_pred, max_val=1.0):
    """
    Compute average PSNR over a batch of images.

    Args:
        y_true (Tensor): Ground truth images, shape [B, C, H, W]
        y_pred (Tensor): Predicted images, shape [B, C, H, W]
        max_val (float): Maximum pixel value (default: 1.0 for normalized images)

    Returns:
        Tensor: Average PSNR over the batch
    """
    mse = F.mse_loss(y_pred, y_true, reduction='none')
    mse_per_image = mse.view(mse.size(0), -1).mean(dim=1)
    psnr_per_image = 10 * torch.log10((max_val ** 2) / mse_per_image.clamp(min=1e-10))  # Avoid log(0)
    return psnr_per_image.mean()
