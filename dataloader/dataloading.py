import torch
from torch.utils.data import Dataset, DataLoader


class ImageDataset(Dataset):
    def __init__(self, downsampled_images, highres_images):
        self.lr_images = torch.tensor(downsampled_images, dtype=torch.float32) / 255.0
        self.hr_images = torch.tensor(highres_images, dtype=torch.float32) / 255.0

    def __len__(self):
        return len(self.lr_images)

    def __getitem__(self, idx):
        lr = self.lr_images[idx].permute(2, 0, 1)  # [H, W, C] → [C, H, W]
        hr = self.hr_images[idx].permute(2, 0, 1)
        return lr, hr


def get_dataloader(downsampled_images, highres_images, batch_size=16, shuffle=True):
    dataset = ImageDataset(downsampled_images, highres_images)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


# Example usage
if __name__ == "__main__":
    # Assume downsampled_images and images are numpy arrays [N, H, W, C]
    from your_loader_script import downsampled_images, images  # Replace with actual loading

    dataloader = get_dataloader(downsampled_images, images)

    for lr_batch, hr_batch in dataloader:
        print("LR batch shape:", lr_batch.shape)   # [B, C, H, W]
        print("HR batch shape:", hr_batch.shape)
        break
