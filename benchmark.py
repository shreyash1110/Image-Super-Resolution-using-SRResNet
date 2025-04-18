import os
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
from model import SRResNet
from loss import psnr_batch, ssim_loss
from tqdm import tqdm

# --------- Config ---------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
checkpoint_path = 'srresnet_epoch20.pth'  # Update with your checkpoint
test_folder = './test_images/'  # Folder of input test images
results_folder = './benchmark_results/'  # Output folder
os.makedirs(results_folder, exist_ok=True)

img_size = 128
lr_size = 32

# --------- Load Model ---------
model = SRResNet().to(device)
model.load_state_dict(torch.load(checkpoint_path, map_location=device))
model.eval()

# --------- Transforms ---------
to_tensor = transforms.ToTensor()
resize_hr = transforms.Resize((img_size, img_size), interpolation=Image.BICUBIC)
resize_lr = transforms.Resize((lr_size, lr_size), interpolation=Image.BICUBIC)

# --------- Benchmark Loop ---------
image_files = [f for f in os.listdir(test_folder) if f.lower().endswith(('png', 'jpg', 'jpeg'))]

for file in tqdm(image_files, desc="Benchmarking"):
    img_path = os.path.join(test_folder, file)
    image = Image.open(img_path).convert('RGB')

    # HR and LR images
    hr_image = resize_hr(image)
    lr_image = resize_lr(hr_image)

    hr_tensor = to_tensor(hr_image).unsqueeze(0).to(device)
    lr_tensor = to_tensor(lr_image).unsqueeze(0).to(device)

    # Bicubic Upsampling
    bicubic_tensor = F.interpolate(lr_tensor, size=(img_size, img_size), mode='bicubic', align_corners=True)

    # SRResNet Output
    with torch.no_grad():
        sr_tensor = model(lr_tensor)

    # PSNR + SSIM
    psnr_bicubic = psnr_batch(hr_tensor, bicubic_tensor).item()
    psnr_sr = psnr_batch(hr_tensor, sr_tensor).item()
    ssim_bicubic = 1 - ssim_loss(hr_tensor, bicubic_tensor).item()
    ssim_sr = 1 - ssim_loss(hr_tensor, sr_tensor).item()

    # Convert tensors to images
    def tensor_to_pil(t):
        return transforms.ToPILImage()(t.squeeze(0).cpu().clamp(0, 1))

    hr_img = tensor_to_pil(hr_tensor)
    lr_img = tensor_to_pil(lr_tensor)
    bicubic_img = tensor_to_pil(bicubic_tensor)
    sr_img = tensor_to_pil(sr_tensor)

    # Save visual comparison
    fig, axs = plt.subplots(1, 4, figsize=(18, 5))
    axs[0].imshow(hr_img); axs[0].set_title('Original HR')
    axs[1].imshow(lr_img); axs[1].set_title('Low-Res Input')
    axs[2].imshow(bicubic_img); axs[2].set_title(f'Bicubic\nPSNR: {psnr_bicubic:.2f}, SSIM: {ssim_bicubic:.3f}')
    axs[3].imshow(sr_img); axs[3].set_title(f'Super-Res\nPSNR: {psnr_sr:.2f}, SSIM: {ssim_sr:.3f}')

    for ax in axs: ax.axis('off')
    plt.tight_layout()
    save_path = os.path.join(results_folder, f'benchmark_{os.path.splitext(file)[0]}.png')
    plt.savefig(save_path)
    plt.close()
