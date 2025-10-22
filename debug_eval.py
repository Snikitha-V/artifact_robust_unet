import torch
from torch.utils.data import DataLoader
from dataset.acdc_dataset import ACDCDataset
from models.unet import UNet
import torchio as tio
import numpy as np
import os

device = "cuda" if torch.cuda.is_available() else "cpu"

# Load trained model
model = UNet(in_ch=1, out_ch=4).to(device)
model.load_state_dict(torch.load('unet_acdc.pth', map_location=device))
model.eval()

print(f"Device: {device}\n")

# Test on TRAINING set
print("=== TRAINING SET ===")
train_transform = tio.Compose([tio.CropOrPad((256, 256, 1))])
train_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), 'data', 'training'))
train_ds = ACDCDataset(train_dir, transform=train_transform, return_meta=True, slice_policy='center')
train_loader = DataLoader(train_ds, batch_size=4, num_workers=0)

train_preds = []
train_masks = []
for imgs, masks, _ in train_loader:
    imgs = imgs.to(device)
    with torch.no_grad():
        out = model(imgs)
        preds = torch.argmax(out, dim=1)
    train_preds.append(preds.cpu().numpy())
    train_masks.append(masks.numpy())

train_preds = np.concatenate(train_preds, axis=0)
train_masks = np.concatenate(train_masks, axis=0)

print(f"Train preds shape: {train_preds.shape}, classes: {np.unique(train_preds)}")
print(f"Train masks shape: {train_masks.shape}, classes: {np.unique(train_masks)}")
print(f"Train FG pred %: {(train_preds > 0).mean()*100:.2f}%")
print(f"Train FG gt %: {(train_masks > 0).mean()*100:.2f}%")

# Compute simple per-class accuracy
for c in range(4):
    tp = np.sum((train_preds == c) & (train_masks == c))
    total_pred = np.sum(train_preds == c)
    total_gt = np.sum(train_masks == c)
    if total_gt > 0:
        recall = tp / total_gt
        print(f"  Class {c}: Recall = {recall:.4f} ({tp}/{total_gt})")

# Test on TEST set
print("\n=== TEST SET ===")
test_transform = tio.Compose([tio.CropOrPad((256, 256, 1))])
test_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), 'data', 'testing'))
test_ds = ACDCDataset(test_dir, transform=test_transform, return_meta=True, slice_policy='center')
test_loader = DataLoader(test_ds, batch_size=4, num_workers=0)

test_preds = []
test_masks = []
for imgs, masks, _ in test_loader:
    imgs = imgs.to(device)
    with torch.no_grad():
        out = model(imgs)
        preds = torch.argmax(out, dim=1)
    test_preds.append(preds.cpu().numpy())
    test_masks.append(masks.numpy())

test_preds = np.concatenate(test_preds, axis=0)
test_masks = np.concatenate(test_masks, axis=0)

print(f"Test preds shape: {test_preds.shape}, classes: {np.unique(test_preds)}")
print(f"Test masks shape: {test_masks.shape}, classes: {np.unique(test_masks)}")
print(f"Test FG pred %: {(test_preds > 0).mean()*100:.2f}%")
print(f"Test FG gt %: {(test_masks > 0).mean()*100:.2f}%")

# Compute per-class accuracy
for c in range(4):
    tp = np.sum((test_preds == c) & (test_masks == c))
    total_pred = np.sum(test_preds == c)
    total_gt = np.sum(test_masks == c)
    if total_gt > 0:
        recall = tp / total_gt
        print(f"  Class {c}: Recall = {recall:.4f} ({tp}/{total_gt})")
