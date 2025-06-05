# Image-Super-Resolution-using-SRResNet

Kaggle Notebook (Implementation) : https://www.kaggle.com/code/shreyash1110/srresnet-demo

This project implements **SRResNet** for single image super-resolution, aiming to reconstruct high-resolution (HR) images from their low-resolution (LR) counterparts using deep convolutional neural networks.

---
## Project Tree:
tree -L 2 > structure.md
---

## 🔍 Key Features:

- SRResNet architecture using **PyTorch**
- Trained on **facial image datasets** with LR size: **32×32**, HR size: **128×128**
- **Random cropping** and **bicubic downsampling** for LR-HR pair generation
- Supports **PSNR/SSIM evaluation** and **visual comparison**
- Modular training pipeline with easily customizable loss functions, datasets, and network layers

---

## 🧪 Goals & Experiments:

- Fine-tune **perceptual loss** for better texture restoration
- Explore improved **content loss** using intermediate VGG layers
- Investigate impact of different **upsampling techniques** (e.g., sub-pixel convolution vs transposed convolution)

## Model graph
![srresnet](https://github.com/user-attachments/assets/1a46d42e-5ee8-4c30-8779-8435980e4cf6)

## Residual Block
![image](https://github.com/user-attachments/assets/2c388469-bea4-42f5-9112-10664006cb3b)

## Upsampling block
![image](https://github.com/user-attachments/assets/0b1529c6-6c99-4709-a9ae-d05bb7189d95)

---

