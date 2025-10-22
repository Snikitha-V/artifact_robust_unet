import torch
from torch.utils.data import DataLoader
from dataset.acdc_dataset import ACDCDataset
from models.attention_unet import AttUNet
import torchio as tio
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ----------------------
    # Data Augmentation
    # ----------------------
    train_transform = tio.Compose([
        tio.RandomFlip(axes=(0,1)),
        tio.RandomAffine(scales=(0.9,1.1), degrees=10),
        tio.RescaleIntensity((0,1))
    ])

    # Load dataset
    train_ds = ACDCDataset("data/training", transform=train_transform)
    train_loader = DataLoader(train_ds, batch_size=4, shuffle=True, num_workers=0)  # num_workers=0 for macOS

    # ----------------------
    # Model, Loss, Optimizer
    # ----------------------
    model = AttUNet(in_ch=1, out_ch=4).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    # ----------------------
    # Training loop
    # ----------------------
    for epoch in range(5):  # start small
        model.train()
        total_loss = 0
        for imgs, masks in tqdm(train_loader):
            imgs, masks = imgs.to(device), masks.to(device)

            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1}: Loss = {total_loss / len(train_loader):.4f}")

    # ----------------------
    # Save model
    # ----------------------
    torch.save(model.state_dict(), "attention_unet_acdc.pth")
    print("✅ Attention U-Net saved as attention_unet_acdc.pth")

if __name__ == "__main__":
    import torch.multiprocessing
    torch.multiprocessing.set_start_method('spawn', force=True)
    main()
