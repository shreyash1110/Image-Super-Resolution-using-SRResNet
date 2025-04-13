import matplotlib.pyplot as plt

def plot_training_curves(
    train_losses, val_losses, train_psnrs, val_psnrs, save_path=None
):
    epochs = list(range(1, len(train_losses) + 1))

    plt.figure(figsize=(12, 5))

    # Plot Losses
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, label='Train Loss', marker='o')
    plt.plot(epochs, val_losses, label='Val Loss', marker='o')
    plt.title('Loss vs Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    # Plot PSNR
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_psnrs, label='Train PSNR', marker='s')
    plt.plot(epochs, val_psnrs, label='Val PSNR', marker='s')
    plt.title('PSNR vs Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('PSNR (dB)')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        print(f"📊 Plot saved to {save_path}")
    else:
        plt.show()

### After training, simply call
plot_training_curves(
    loss_epochwise,
    valid_loss_epochwise,
    psnr_epochwise,
    valid_psnr_epochwise,
    save_path="plots/training_curves.png"
)

