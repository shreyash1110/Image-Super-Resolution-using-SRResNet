# Image-Super-Resolution-using-SRResNet

Kaggle Notebook (Implementation) : https://www.kaggle.com/code/shreyash1110/srresnet-demo

This project implements **SRResNet** for single image super-resolution, aiming to reconstruct high-resolution (HR) images from their low-resolution (LR) counterparts using deep convolutional neural networks.

---
## Project Tree:
```
.
|-- README.md
|-- dataloader
|   |-- Training_and_Validation_data_creation.py
|   `-- dataloading.py
|-- dataset
|   |-- benchmark_hr.npy
|   |-- benchmark_lr.npy
|   |-- hr_train_1.npy
|   |-- hr_train_2.npy
|   |-- hr_valid.npy
|   |-- lr_train.npy
|   |-- lr_valid.npy
|   |-- valid_hr.npy
|   `-- valid_lr.npy
|-- initialisation.py
|-- loss
|   |-- loss_v2.py
|   |-- loss_v3.py
|   |-- psnr.py
|   `-- vanilla_loss.py
|-- models
|   |-- _init_.py
|   |-- model_v2.py
|   `-- vanilla_model.py
|-- structure.md
|-- training_loop.py
`-- visualise
    |-- benchmark.py
    `-- visualise_after_traininig.py

5 directories, 24 files
```
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

