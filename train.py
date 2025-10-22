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
from tqdm import tqdm
import numpy as np
import os


def parse_args():
    parser = argparse.ArgumentParser(description="Train UNet on ACDC with class weighting and early stopping")
    parser.add_argument("--epochs", type=int, default=100, help="Max epochs (early stopping may stop earlier)")
    parser.add_argument("--batch-size", type=int, default=12, help="Batch size (larger for 12GB VRAM)")
    parser.add_argument("--lr", type=float, default=5e-4, help="Learning rate")
    parser.add_argument("--num-workers", type=int, default=8, help="DataLoader workers (CPU cores feed GPU pipeline)")
    parser.add_argument("--amp", action="store_true", default=True, help="Enable mixed precision (AMP)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--deterministic", action="store_true", help="Deterministic CuDNN (slower)")
    parser.add_argument("--early-stopping-patience", type=int, default=10, help="Early stopping patience (epochs)")
    parser.add_argument("--val-freq", type=int, default=5, help="Validation frequency (epochs)")
    return parser.parse_args()


def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Leave CuDNN settings to main() based on flag


def compute_class_weights(dataset, num_classes=4):
    """Compute class weights for imbalanced dataset (inverse frequency)."""
    class_counts = np.zeros(num_classes)
    for i in range(len(dataset)):
        _, mask = dataset[i]
        for c in range(num_classes):
            class_counts[c] += (mask == c).sum()
    
    # Inverse frequency weighting
    total = class_counts.sum()
    weights = total / (num_classes * class_counts + 1e-8)
    weights = weights / weights.sum()  # Normalize to mean=1
    return torch.from_numpy(weights).float()


def validate(model, val_loader, criterion, device):
    """Run validation loop and return average loss."""
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for imgs, masks in val_loader:
            imgs, masks = imgs.to(device, non_blocking=True), masks.to(device, dtype=torch.long, non_blocking=True)
            if device == 'cuda':
                imgs = imgs.to(memory_format=torch.channels_last)
            outputs = model(imgs)
            loss = criterion(outputs, masks)
            total_loss += loss.item()
    model.train()
    return total_loss / len(val_loader)


def main():
    args = parse_args()
    seed_everything(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n🔥 GPU Accelerated Training")
    print(f"   Device: {device}")
    if device == 'cuda':
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
        print(f"   CUDA Capability: {torch.cuda.get_device_capability(0)}")
        print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        torch.cuda.reset_peak_memory_stats()
    print()
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
        persistent_workers=True if args.num_workers > 0 and device == 'cuda' else False,
        prefetch_factor=2 if args.num_workers > 0 else 0,
    )

    model = UNet(in_ch=1, out_ch=4).to(device)
    # channels_last can improve tensor core usage on Ampere GPUs
    if device == 'cuda':
        model = model.to(memory_format=torch.channels_last)
    
    # Compute class weights for imbalanced dataset
    class_weights = compute_class_weights(train_ds, num_classes=4)
    print(f"📊 Class weights (inverse frequency): {class_weights.numpy()}")
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))

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
    print(f"⚡ AMP (Mixed Precision): {args.amp and device == 'cuda'}")
    print(f"📦 Batch: {args.batch_size} | Workers: {args.num_workers} | LR: {args.lr:.0e}\n")

    # Create validation split (10% of training data)
    val_size = max(1, len(train_ds) // 10)
    train_size = len(train_ds) - val_size
    train_ds_split, val_ds_split = torch.utils.data.random_split(train_ds, [train_size, val_size])
    val_loader = DataLoader(
        val_ds_split,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True if device == 'cuda' else False,
        persistent_workers=True if args.num_workers > 0 and device == 'cuda' else False,
    )

    # Early stopping
    best_val_loss = float('inf')
    patience_counter = 0

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
                # CrossEntropyLoss expects (N, C, H, W) logits and (N, H, W) class indices
                loss = criterion(outputs, masks)

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

        train_avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch + 1}: Train Loss = {train_avg_loss:.4f}", end="")

        # Validation every N epochs
        if (epoch + 1) % args.val_freq == 0 or epoch == 0:
            val_loss = validate(model, val_loader, criterion, device)
            print(f" | Val Loss = {val_loss:.4f}", end="")
            
            # Early stopping logic
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                torch.save(model.state_dict(), "unet_acdc_best.pth")
                print(" ✓ (best)")
            else:
                patience_counter += 1
                print(f" (patience {patience_counter}/{args.early_stopping_patience})")
                if patience_counter >= args.early_stopping_patience:
                    print(f"\n⏸️ Early stopping at epoch {epoch + 1} (val loss not improving)")
                    model.load_state_dict(torch.load("unet_acdc_best.pth"))
                    break
        else:
            print()

    torch.save(model.state_dict(), "unet_acdc.pth")
    print("✅ Model saved as unet_acdc.pth")


if __name__ == "__main__":
    main()
