# Image-Super-Resolution-using-SRResNet

This project implements **SRResNet** for single image super-resolution, aiming to reconstruct high-resolution (HR) images from their low-resolution (LR) counterparts using deep convolutional neural networks.

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

---

## 📁 Project Structure:

```bash
├── models/             # SRResNet model definitions  
├── data/               # Dataset loading and preprocessing  
├── utils/              # Helper functions (metrics, visualization, etc.)  
├── train.py            # Training script  
├── evaluate.py         # Evaluation script  
└── README.md           # Project documentation
