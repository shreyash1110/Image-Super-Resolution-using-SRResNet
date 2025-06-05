import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms as T

def visualize_benchmark_results(benchmark_lr, benchmark_hr, model, state_dict_path, device='cuda' if torch.cuda.is_available() else 'cpu'):
    """
    benchmark_lr: np.ndarray of shape (5, 32, 32, 3)
    benchmark_hr: np.ndarray of shape (5, 128, 128, 3)
    model: a PyTorch model instance (not loaded yet)
    state_dict_path: path to the saved model weights (.pt or .pth)
    """

    model.load_state_dict(torch.load(state_dict_path, map_location=device))
    model.eval()
    model.to(device)

    # Normalize and transpose to match model input: (N, C, H, W)
    transform = T.ToTensor()
    lr_tensor = torch.stack([transform(img.astype(np.uint8)) for img in benchmark_lr]).to(device)
    hr_tensor = torch.stack([transform(img.astype(np.uint8)) for img in benchmark_hr]).to(device)

    # Bicubic Upsample to 128x128
    upsampled_lr = F.interpolate(lr_tensor, size=(128, 128), mode='bicubic', align_corners=False)

    # Model Prediction
    with torch.no_grad():
        sr_pred = model(lr_tensor)

    # Convert to numpy for plotting
    def to_numpy(t):
        t = torch.clamp(t, 0, 1)
        return t.permute(0, 2, 3, 1).cpu().numpy()

    lr_imgs = to_numpy(lr_tensor)
    up_imgs = to_numpy(upsampled_lr)
    pred_imgs = to_numpy(sr_pred)
    hr_imgs = to_numpy(hr_tensor)

    # Plotting
    fig, axes = plt.subplots(nrows=5, ncols=4, figsize=(16, 10))
    titles = ['LR (32x32)', 'Bicubic (128x128)', 'Model Output', 'Ground Truth HR']

    for row in range(5):
        images = [lr_imgs[row], up_imgs[row], pred_imgs[row], hr_imgs[row]]
        for col in range(4):
            axes[row, col].imshow(images[col])
            axes[row, col].axis('off')
            if row == 0:
                axes[row, col].set_title(titles[col], fontsize=12)

    plt.tight_layout()
    plt.show()
## Example usage

# benchmark_lr = np.load("benchmark_lr.npy")
# benchmark_hr = np.load("benchmark_hr.npy")

# # Initialize your model architecture (without weights)
# from your_model_file import YourSRModel
# model = YourSRModel()

# # Call the function
# visualize_benchmark_results(
#     benchmark_lr,
#     benchmark_hr,
#     model=model,
#     state_dict_path="checkpoints/model_best.pth"
# )
