import os
import torch

# Set CUDA memory allocation behavior
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize model
model = SRResNet().to(device)

# Optimizer and Scheduler
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

# Training hyperparameters
num_epochs = 20

# Initialize metric tracking lists
psnr_epochwise = [0.0 for _ in range(num_epochs)]
valid_psnr_epochwise = [0.0 for _ in range(num_epochs)]
loss_epochwise = [0.0 for _ in range(num_epochs)]
valid_loss_epochwise = [0.0 for _ in range(num_epochs)]
