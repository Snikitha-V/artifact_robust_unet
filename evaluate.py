import argparse
import os
import torch
from torch.utils.data import DataLoader
from dataset.acdc_dataset import ACDCDataset
from models.unet import UNet
from models.attention_unet import AttUNet
import torchio as tio
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import directed_hausdorff
from scipy import ndimage as ndi
from skimage.segmentation import find_boundaries

# ----------------------------
# Artifact transforms
# ----------------------------
def make_artifact_transforms(args):
    return {
        "noise": tio.RandomNoise(mean=0, std=(args.noise_std, args.noise_std)),
        "motion": tio.RandomMotion(degrees=args.motion_deg, translation=args.motion_trans),
        "bias": tio.RandomBiasField(coefficients=args.bias_coeff),
        "slice_dropout": tio.RandomSpike(intensity=(args.spike_intensity_min, args.spike_intensity_max), num_spikes=args.spike_num),
    }

# ----------------------------
# Metrics
# ----------------------------
def dice_coef_binary(pred_bin: np.ndarray, target_bin: np.ndarray, smooth=1e-6):
    pred_flat = pred_bin.reshape(-1).astype(np.bool_)
    target_flat = target_bin.reshape(-1).astype(np.bool_)
    inter = np.sum(pred_flat & target_flat)
    return (2.0 * inter) / (pred_flat.sum() + target_flat.sum() + smooth)

def hd95_mm(pred_bin: np.ndarray, target_bin: np.ndarray, spacing_xy: np.ndarray):
    """Compute symmetric 95th percentile Hausdorff distance in millimeters using pixel spacing.
    pred_bin and target_bin are boolean 2D arrays; spacing_xy is (sx, sy).
    """
    pred_bin = pred_bin.astype(bool)
    target_bin = target_bin.astype(bool)
    if pred_bin.sum() == 0 or target_bin.sum() == 0:
        return np.nan
    s1 = find_boundaries(pred_bin, mode='inner')
    s2 = find_boundaries(target_bin, mode='inner')
    if s1.sum() == 0 or s2.sum() == 0:
        return np.nan
    # Distance maps (mm) using anisotropic spacing
    dt2 = ndi.distance_transform_edt(~s2, sampling=(spacing_xy[1], spacing_xy[0]))
    dt1 = ndi.distance_transform_edt(~s1, sampling=(spacing_xy[1], spacing_xy[0]))
    d12 = dt2[s1]
    d21 = dt1[s2]
    if d12.size == 0 or d21.size == 0:
        return np.nan
    all_d = np.concatenate([d12, d21])
    return np.percentile(all_d, 95)

def boundary_dice_binary(pred_bin: np.ndarray, target_bin: np.ndarray):
    pred_boundary = find_boundaries(pred_bin, mode='inner').astype(np.uint8)
    target_boundary = find_boundaries(target_bin, mode='inner').astype(np.uint8)
    if pred_boundary.sum() == 0 or target_boundary.sum() == 0:
        return np.nan
    intersection = (pred_boundary & target_boundary).sum()
    return (2.0 * intersection) / (pred_boundary.sum() + target_boundary.sum())

# ----------------------------
# Load model
# ----------------------------
def load_model(choice, device):
    if choice.lower() == "unet":
        model = UNet(in_ch=1, out_ch=4).to(device)
        path = "unet_acdc.pth"
    elif choice.lower() == "attention":
        model = AttUNet(in_ch=1, out_ch=4).to(device)
        path = "attention_unet_acdc.pth"
    else:
        raise ValueError("Choose either 'unet' or 'attention'")
    
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()
    return model

# ----------------------------
# Evaluation loop
# ----------------------------
def evaluate(model, loader, device, num_classes: int = 4):
    per_set_dice = []
    per_set_hd95 = []
    per_set_bdice = []
    fg_pred_ratios = []
    fg_gt_ratios = []

    for batch in tqdm(loader):
        # batch may be (imgs, masks, spacing)
        if len(batch) == 3:
            imgs, masks, spacings = batch
            spacings = spacings.numpy()
        else:
            imgs, masks = batch
            spacings = np.tile(np.array([1.0, 1.0], dtype=np.float32), (imgs.shape[0], 1))

        imgs, masks = imgs.to(device), masks.to(device)
        with torch.no_grad():
            outputs = model(imgs)
            preds = torch.argmax(outputs, dim=1)

        preds_np = preds.cpu().numpy().astype(np.int32)
        masks_np = masks.cpu().numpy().astype(np.int32)

        # per-image, per-class (exclude background=0)
        for i in range(preds_np.shape[0]):
            p = preds_np[i]
            m = masks_np[i]
            spacing_xy = spacings[i]

            # foreground ratios (diagnostic)
            fg_pred_ratios.append((p > 0).mean())
            fg_gt_ratios.append((m > 0).mean())

            dices = []
            hds = []
            bdices = []
            for c in range(1, num_classes):
                pb = (p == c)
                mb = (m == c)
                dices.append(dice_coef_binary(pb, mb))
                hds.append(hd95_mm(pb, mb, spacing_xy))
                bdices.append(boundary_dice_binary(pb, mb))

            # aggregate over classes for this image
            per_set_dice.append(np.nanmean(dices))
            per_set_hd95.append(np.nanmean(hds))
            per_set_bdice.append(np.nanmean(bdices))

    return (
        np.nanmean(per_set_dice),
        np.nanmean(per_set_hd95),
        np.nanmean(per_set_bdice),
        float(np.nanmean(fg_pred_ratios)) if len(fg_pred_ratios) else float('nan'),
        float(np.nanmean(fg_gt_ratios)) if len(fg_gt_ratios) else float('nan'),
    )

# ----------------------------
# Main
# ----------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate UNet/Attention U-Net on ACDC")
    parser.add_argument("--model", choices=["unet", "attention"], default="unet", help="Model to evaluate")
    parser.add_argument("--batch-size", type=int, default=2, help="Evaluation batch size")
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader workers")
    parser.add_argument("--size", type=int, nargs=2, default=[256, 256], help="Spatial size H W for CropOrPad")
    parser.add_argument("--debug-artifacts", action="store_true", help="Save a few samples for each artifact to verify transforms")
    parser.add_argument("--outdir", type=str, default="eval_debug", help="Output directory for debug samples")
    # Artifact severities
    parser.add_argument("--noise-std", type=float, default=0.03, help="Std for RandomNoise")
    parser.add_argument("--motion-deg", type=float, default=10.0, help="Degrees for RandomMotion")
    parser.add_argument("--motion-trans", type=float, default=5.0, help="Translation (px) for RandomMotion")
    parser.add_argument("--bias-coeff", type=float, default=0.3, help="Coefficient for RandomBiasField")
    parser.add_argument("--spike-intensity-min", type=float, default=0.8, help="Min intensity for RandomSpike")
    parser.add_argument("--spike-intensity-max", type=float, default=1.2, help="Max intensity for RandomSpike")
    parser.add_argument("--spike-num", type=int, default=5, help="Number of spikes for RandomSpike")
    return parser.parse_args()


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(args.model, device)

    # ----------------------------
    # Clean test set
    # ----------------------------
    print("\nEvaluating on CLEAN test set...")
    clean_transform = tio.Compose([
        tio.CropOrPad((args.size[0], args.size[1], 1)),
    ])
    clean_ds = ACDCDataset("data/testing", transform=clean_transform, return_meta=True, slice_policy='center')
    clean_loader = DataLoader(
        clean_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True if device == 'cuda' else False,
        persistent_workers=True if args.num_workers > 0 else False,
    )
    dice_clean, hd_clean, bd_clean, fg_pred_clean, fg_gt_clean = evaluate(model, clean_loader, device)
    
    # ----------------------------
    # Artifact test sets
    # ----------------------------
    results = {}
    artifact_transforms = make_artifact_transforms(args)
    for name, transform in artifact_transforms.items():
        print(f"\nEvaluating on {name.upper()} test set...")
        ds = ACDCDataset("data/testing", transform=tio.Compose([
            transform,
            tio.CropOrPad((args.size[0], args.size[1], 1)),
        ]), return_meta=True, slice_policy='center')
        loader = DataLoader(
            ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True if device == 'cuda' else False,
            persistent_workers=True if args.num_workers > 0 else False,
        )
        dice, hd, bd, fg_pred, fg_gt = evaluate(model, loader, device)
        results[name] = (dice, hd, bd, fg_pred, fg_gt)

        # Optional debug: save a couple of samples per artifact
        if args.debug_artifacts:
            os.makedirs(args.outdir, exist_ok=True)
            # Save first two transformed images and masks for quick visual check
            for i in range(min(2, len(ds))):
                img_t, msk_t, _ = ds[i]
                img_np = img_t.numpy().squeeze(0)
                msk_np = msk_t.numpy()
                plt.imsave(os.path.join(args.outdir, f"{name}_img_{i}.png"), img_np, cmap='gray')
                plt.imsave(os.path.join(args.outdir, f"{name}_mask_{i}.png"), msk_np, cmap='viridis')

    # ----------------------------
    # Print results table
    # ----------------------------
    print("\n================ PERFORMANCE COMPARISON ================")
    print(f"{'Test Set':<15} | {'Dice':<10} | {'HD95 (mm)':<12} | {'Boundary Dice':<15} | {'FG pred%':<9} | {'FG gt%':<7}")
    print("-"*60)
    print(f"{'Clean':<15} | {dice_clean:<10.4f} | {hd_clean:<12.2f} | {bd_clean:<15.4f} | {fg_pred_clean*100:<9.2f} | {fg_gt_clean*100:<7.2f}")
    for k, v in results.items():
        print(f"{k:<15} | {v[0]:<10.4f} | {v[1]:<12.2f} | {v[2]:<15.4f} | {v[3]*100:<9.2f} | {v[4]*100:<7.2f}")
    print("=======================================================")

if __name__ == "__main__":
    import torch.multiprocessing
    torch.multiprocessing.set_start_method('spawn', force=True)
    main()
