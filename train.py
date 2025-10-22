import argparse
import random
import torch
from torch.utils.data import DataLoader
from dataset.acdc_dataset import ACDCDataset
from models.unet import UNet
import torchio as tio
import torch.nn as nn
import torch.optim as optim
from torch.amp import autocast, GradScaler
from monai.losses import DiceCELoss
from tqdm import tqdm
import numpy as np
import os


def parse_args():
    parser = argparse.ArgumentParser(description="Train UNet on ACDC with improved loss")
    parser.add_argument("--epochs", type=int, default=5, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--num-workers", type=int, default=4, help="DataLoader workers")
    parser.add_argument("--amp", action="store_true", help="Enable mixed precision (AMP)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--deterministic", action="store_true", help="Deterministic CuDNN (slower)")
    return parser.parse_args()


def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Leave CuDNN settings to main() based on flag


def main():
    args = parse_args()
    seed_everything(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    # cuDNN settings: prefer speed unless deterministic requested
    if args.deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
    # Matmul precision (PyTorch 2.x)
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass

    train_transform = tio.Compose([
        tio.RandomFlip(axes=(0, 1)),
        tio.RandomAffine(scales=(0.9, 1.1), degrees=10),
        tio.RescaleIntensity((0, 1)),
        # Ensure all samples have identical spatial dims (X, Y, Z)
        tio.CropOrPad((256, 256, 1)),
    ])

    train_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), 'data', 'training'))
    train_ds = ACDCDataset(train_dir, transform=train_transform)
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True if device == 'cuda' else False,
        persistent_workers=True if args.num_workers > 0 else False,
    )

    model = UNet(in_ch=1, out_ch=4).to(device)
    # channels_last can improve tensor core usage on Ampere GPUs
    if device == 'cuda':
        model = model.to(memory_format=torch.channels_last)
    # Improved loss: Dice + CrossEntropy combined
    criterion = DiceCELoss(
        softmax=True,
        to_onehot_y=True,
        include_background=True,
        lambda_dice=1.0,
        lambda_ce=1.0,
        squared_pred=True,
    )

    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=args.lr,
        steps_per_epoch=len(train_loader),
        epochs=args.epochs,
        pct_start=0.1,
        div_factor=25.0,
        final_div_factor=1e4,
        anneal_strategy='cos',
    )

    scaler = GradScaler(enabled=(args.amp and device == 'cuda'))

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        pbar = tqdm(train_loader)
        for imgs, masks in pbar:
            imgs, masks = imgs.to(device, non_blocking=True), masks.to(device, dtype=torch.long, non_blocking=True)
            if device == 'cuda':
                imgs = imgs.to(memory_format=torch.channels_last)
            optimizer.zero_grad(set_to_none=True)

            with autocast(device_type='cuda', dtype=torch.float16, enabled=args.amp and device == 'cuda'):
                outputs = model(imgs)
                # DiceCELoss expects targets of shape (N, 1, H, W)
                loss = criterion(outputs, masks.unsqueeze(1))

            if scaler.is_enabled():
                scaler.scale(loss).backward()
                # gradient clipping
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            scheduler.step()
            total_loss += loss.item()
            pbar.set_description(f"Epoch {epoch+1}/{args.epochs} LR {scheduler.get_last_lr()[0]:.2e} Loss {loss.item():.4f}")

        print(f"Epoch {epoch + 1}: Avg Loss = {total_loss / len(train_loader):.4f}")

    torch.save(model.state_dict(), "unet_acdc.pth")
    print("✅ Model saved as unet_acdc.pth")


if __name__ == "__main__":
    main()
