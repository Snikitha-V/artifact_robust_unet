import torch
from torch.utils.data import DataLoader
from dataset.acdc_dataset import ACDCDataset
from models.unet import UNet
import torchio as tio
import torch.nn as nn
import torch.optim as optim
from torch.amp import autocast, GradScaler
import os

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")

# Create a small dataset for debugging
train_transform = tio.Compose([
    tio.CropOrPad((256, 256, 1)),
])
train_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), 'data', 'training'))
train_ds = ACDCDataset(train_dir, transform=train_transform)

# Just one batch for debugging
train_loader = DataLoader(
    train_ds,
    batch_size=2,
    shuffle=False,
    num_workers=0,
    pin_memory=True if device == 'cuda' else False,
)

model = UNet(in_ch=1, out_ch=4).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

print("\n=== Before training ===")
imgs, masks = next(iter(train_loader))
print(f"Input shape: {imgs.shape}, mask shape: {masks.shape}")
print(f"Mask classes: {masks.unique()}")

# Get model output before training
with torch.no_grad():
    out = model(imgs.to(device))
    pred_before = torch.argmax(out, dim=1)
    print(f"Pred classes before: {pred_before.unique()}")

# Train for 1 step
print("\n=== Training step ===")
model.train()
imgs, masks = imgs.to(device), masks.to(device, dtype=torch.long)
print(f"Masks dtype: {masks.dtype}, device: {masks.device}")

optimizer.zero_grad()
outputs = model(imgs)
print(f"Outputs shape: {outputs.shape}, Masks shape: {masks.shape}")
print(f"Outputs dtype: {outputs.dtype}, Masks dtype: {masks.dtype}")

loss = criterion(outputs, masks)
print(f"Loss: {loss.item():.4f}")

# Check gradients before backward
print(f"Grad before backward: {model.inc.conv[0].weight.grad}")

loss.backward()

# Check gradients after backward
print(f"Grad after backward (non-zero?): {model.inc.conv[0].weight.grad is not None and model.inc.conv[0].weight.grad.abs().sum() > 0}")

optimizer.step()

# Get model output after training
print("\n=== After 1 training step ===")
model.eval()
with torch.no_grad():
    out = model(imgs.to(device))
    pred_after = torch.argmax(out, dim=1)
    print(f"Pred classes after: {pred_after.unique()}")
    print(f"Output logits changed: {(out.abs() > 30).sum().item()} pixels with |logit| > 30")
