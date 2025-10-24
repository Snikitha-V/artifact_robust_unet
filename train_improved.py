import argparse
import random
import torch
from torch.utils.data import DataLoader, ConcatDataset
from dataset.acdc_dataset import ACDCDataset
from models.unet import UNet
from models.attention_unet import AttUNet
import torchio as tio
import torch.nn as nn
import torch.optim as optim
from torch.amp import autocast, GradScaler
from tqdm import tqdm
import numpy as np
import os
import json
from datetime import datetime


def parse_args():
    parser = argparse.ArgumentParser(description="Improved UNet training with artifact augmentation")
    parser.add_argument("--model", choices=["unet", "attention"], default="unet", help="Model architecture")
    parser.add_argument("--epochs", type=int, default=150, help="Max epochs")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Initial learning rate")
    parser.add_argument("--num-workers", type=int, default=8, help="DataLoader workers")
    parser.add_argument("--amp", action="store_true", default=True, help="Enable mixed precision")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--early-stopping-patience", type=int, default=15, help="Early stopping patience")
    parser.add_argument("--val-freq", type=int, default=3, help="Validation frequency")
    parser.add_argument("--accumulation-steps", type=int, default=2, help="Gradient accumulation steps")
    parser.add_argument("--artifact-augment", action="store_true", default=True, help="Add artifact augmentation")
    parser.add_argument("--artifact-prob", type=float, default=0.3, help="Probability of artifact augmentation")
    parser.add_argument("--checkpoint", type=str, default=None, help="Resume from checkpoint")
    return parser.parse_args()


def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def compute_class_weights(dataset, num_classes=4):
    """Compute class weights for imbalanced dataset."""
    class_counts = np.zeros(num_classes)
    for i in range(min(len(dataset), 500)):  # Sample first 500 for efficiency
        _, mask = dataset[i]
        for c in range(num_classes):
            class_counts[c] += (mask == c).sum()
    
    total = class_counts.sum()
    weights = total / (num_classes * class_counts + 1e-8)
    weights = weights / weights.sum()
    return torch.from_numpy(weights).float()


def get_augmentation_transforms(artifact_prob=0.3):
    """Create transforms with spatial and intensity augmentation."""
    artifact_transforms = [
        tio.RandomNoise(mean=0, std=(0.01, 0.03), p=artifact_prob),
        tio.RandomMotion(degrees=(5, 15), translation=(2, 8), p=artifact_prob),
        tio.RandomBiasField(coefficients=(0.1, 0.5), p=artifact_prob),
        tio.RandomSpike(intensity=(0.5, 1.5), num_spikes=(2, 8), p=artifact_prob),
        tio.RandomGamma(log_gamma=(-0.3, 0.3), p=artifact_prob),
        tio.RandomBlur(std=(0.5, 1.5), p=artifact_prob),
    ]
    
    return tio.Compose([
        tio.RandomFlip(axes=(0, 1), p=0.5),
        tio.RandomAffine(scales=(0.8, 1.2), degrees=15, p=0.5),
        tio.OneOf(artifact_transforms, p=0.6) if artifact_transforms else None,
        tio.RescaleIntensity((0, 1)),
        tio.CropOrPad((256, 256, 1)),
    ])


def validate(model, val_loader, criterion, device):
    """Validation loop."""
    model.eval()
    total_loss = 0
    num_batches = 0
    
    with torch.no_grad():
        for imgs, masks in val_loader:
            imgs = imgs.to(device, non_blocking=True)
            masks = masks.to(device, dtype=torch.long, non_blocking=True)
            
            if device == 'cuda':
                imgs = imgs.to(memory_format=torch.channels_last)
            
            outputs = model(imgs)
            loss = criterion(outputs, masks)
            total_loss += loss.item()
            num_batches += 1
    
    model.train()
    return total_loss / max(num_batches, 1)


def load_model(model_type, device):
    """Load model architecture."""
    if model_type.lower() == "unet":
        return UNet(in_ch=1, out_ch=4).to(device)
    elif model_type.lower() == "attention":
        return AttUNet(in_ch=1, out_ch=4).to(device)
    else:
        raise ValueError(f"Unknown model: {model_type}")


def main():
    args = parse_args()
    
    # Setup
    seed_everything(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print("\n" + "="*60)
    print("🔥 IMPROVED UNet Training with Full Optimization")
    print("="*60)
    print(f"Device: {device}")
    if device == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # CuDNN optimization
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
    try:
        torch.set_float32_matmul_precision("high")
    except:
        pass
    
    # Data loading
    print("\n📊 Loading data...")
    train_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), 'data', 'training'))
    
    train_transform = get_augmentation_transforms(args.artifact_prob)
    train_ds = ACDCDataset(train_dir, transform=train_transform, slice_policy='random')
    
    # Create validation split
    val_size = max(1, len(train_ds) // 8)  # 12.5% validation
    train_size = len(train_ds) - val_size
    train_ds_split, val_ds_split = torch.utils.data.random_split(
        train_ds, [train_size, val_size], 
        generator=torch.Generator().manual_seed(args.seed)
    )
    
    train_loader = DataLoader(
        train_ds_split,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True if device == 'cuda' else False,
        persistent_workers=True if args.num_workers > 0 else False,
        prefetch_factor=3 if args.num_workers > 0 else 0,
        drop_last=True,
    )
    
    val_loader = DataLoader(
        val_ds_split,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True if device == 'cuda' else False,
        persistent_workers=True if args.num_workers > 0 else False,
    )
    
    print(f"   Train: {len(train_ds_split)} | Val: {len(val_ds_split)}")
    print(f"   Batch size: {args.batch_size} | Workers: {args.num_workers}")
    
    # Model
    print("\n🧠 Loading model...")
    model = load_model(args.model, device)
    if device == 'cuda':
        model = model.to(memory_format=torch.channels_last)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   Total params: {total_params:,} | Trainable: {trainable_params:,}")
    
    # Loss & Optimizer
    print("\n⚙️ Setup training...")
    class_weights = compute_class_weights(train_ds, num_classes=4)
    print(f"   Class weights: {class_weights.numpy()}")
    
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device), label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=20,
        T_mult=2,
        eta_min=args.lr * 0.01,
    )
    
    scaler = GradScaler(enabled=(args.amp and device == 'cuda'))
    print(f"   AMP enabled: {args.amp and device == 'cuda'}")
    print(f"   Gradient accumulation: {args.accumulation_steps} steps")
    
    # Resume checkpoint if provided
    start_epoch = 0
    best_val_loss = float('inf')
    patience_counter = 0
    
    if args.checkpoint and os.path.exists(args.checkpoint):
        checkpoint = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint.get('epoch', 0)
        best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        print(f"   Resumed from checkpoint (epoch {start_epoch})")
    
    # Training loop
    print("\n" + "="*60)
    print("🚀 Starting training...")
    print("="*60 + "\n")
    
    metrics_log = []
    
    for epoch in range(start_epoch, args.epochs):
        model.train()
        total_loss = 0
        batch_count = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        
        for step, (imgs, masks) in enumerate(pbar):
            imgs = imgs.to(device, non_blocking=True)
            masks = masks.to(device, dtype=torch.long, non_blocking=True)
            
            if device == 'cuda':
                imgs = imgs.to(memory_format=torch.channels_last)
            
            # Forward pass with gradient accumulation
            with autocast(device_type='cuda' if device == 'cuda' else 'cpu', 
                         dtype=torch.float16, 
                         enabled=args.amp and device == 'cuda'):
                outputs = model(imgs)
                loss = criterion(outputs, masks)
                loss = loss / args.accumulation_steps
            
            # Backward pass
            if scaler.is_enabled():
                scaler.scale(loss).backward()
                if (step + 1) % args.accumulation_steps == 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)
            else:
                loss.backward()
                if (step + 1) % args.accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
            
            total_loss += loss.item() * args.accumulation_steps
            batch_count += 1
            
            # Update progress bar
            lr = optimizer.param_groups[0]['lr']
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'lr': f'{lr:.2e}',
                'gpu_mem': f'{torch.cuda.memory_allocated(device) / 1e9:.1f}GB' if device == 'cuda' else 'N/A'
            })
        
        avg_train_loss = total_loss / batch_count
        
        # Validation
        if (epoch + 1) % args.val_freq == 0 or epoch == 0:
            val_loss = validate(model, val_loader, criterion, device)
            
            print(f"\nEpoch {epoch+1:3d} | Train Loss: {avg_train_loss:.4f} | Val Loss: {val_loss:.4f}", end="")
            
            metrics_log.append({
                'epoch': epoch + 1,
                'train_loss': float(avg_train_loss),
                'val_loss': float(val_loss),
            })
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                
                # Save checkpoint
                checkpoint = {
                    'epoch': epoch + 1,
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'best_val_loss': best_val_loss,
                }
                torch.save(checkpoint, f"unet_acdc_best.pth")
                print(" ✓ BEST")
            else:
                patience_counter += 1
                print(f" | Patience: {patience_counter}/{args.early_stopping_patience}")
                
                if patience_counter >= args.early_stopping_patience:
                    print(f"\n⏸️ Early stopping at epoch {epoch+1}")
                    break
        else:
            print(f"Epoch {epoch+1:3d} | Train Loss: {avg_train_loss:.4f}")
        
        scheduler.step()
        
        if device == 'cuda':
            torch.cuda.empty_cache()
    
    # Final save
    torch.save(model.state_dict(), "unet_acdc.pth")
    print("\n✅ Training complete!")
    print(f"✅ Model saved as unet_acdc.pth")
    
    # Save metrics
    with open('training_metrics.json', 'w') as f:
        json.dump(metrics_log, f, indent=2)
    
    print(f"✅ Metrics saved to training_metrics.json")


if __name__ == "__main__":
    main()
