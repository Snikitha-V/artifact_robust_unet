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
import json


def make_artifact_transforms(args):
    """Create artifact transforms for robustness testing."""
    return {
        "noise": tio.RandomNoise(mean=0, std=(args.noise_std, args.noise_std)),
        "motion": tio.RandomMotion(degrees=args.motion_deg, translation=args.motion_trans),
        "bias": tio.RandomBiasField(coefficients=args.bias_coeff),
        "slice_dropout": tio.RandomSpike(intensity=(args.spike_intensity_min, args.spike_intensity_max), num_spikes=args.spike_num),
    }


def dice_coef_binary(pred_bin: np.ndarray, target_bin: np.ndarray, smooth=1e-6):
    """Compute Dice coefficient."""
    pred_flat = pred_bin.reshape(-1).astype(np.bool_)
    target_flat = target_bin.reshape(-1).astype(np.bool_)
    inter = np.sum(pred_flat & target_flat)
    return (2.0 * inter) / (pred_flat.sum() + target_flat.sum() + smooth)


def hd95_mm(pred_bin: np.ndarray, target_bin: np.ndarray, spacing_xy: np.ndarray):
    """Compute symmetric 95th percentile Hausdorff distance in millimeters."""
    pred_bin = pred_bin.astype(bool)
    target_bin = target_bin.astype(bool)
    if pred_bin.sum() == 0 or target_bin.sum() == 0:
        return np.nan
    s1 = find_boundaries(pred_bin, mode='inner')
    s2 = find_boundaries(target_bin, mode='inner')
    if s1.sum() == 0 or s2.sum() == 0:
        return np.nan
    dt2 = ndi.distance_transform_edt(~s2, sampling=(spacing_xy[1], spacing_xy[0]))
    dt1 = ndi.distance_transform_edt(~s1, sampling=(spacing_xy[1], spacing_xy[0]))
    d12 = dt2[s1]
    d21 = dt1[s2]
    if d12.size == 0 or d21.size == 0:
        return np.nan
    all_d = np.concatenate([d12, d21])
    return np.percentile(all_d, 95)


def boundary_dice_binary(pred_bin: np.ndarray, target_bin: np.ndarray):
    """Compute Boundary Dice."""
    pred_boundary = find_boundaries(pred_bin, mode='inner').astype(np.uint8)
    target_boundary = find_boundaries(target_bin, mode='inner').astype(np.uint8)
    if pred_boundary.sum() == 0 or target_boundary.sum() == 0:
        return np.nan
    intersection = (pred_boundary & target_boundary).sum()
    return (2.0 * intersection) / (pred_boundary.sum() + target_boundary.sum())


def load_model(choice, device):
    """Load model and weights."""
    if choice.lower() == "unet":
        model = UNet(in_ch=1, out_ch=4).to(device)
        path = "unet_acdc_best.pth" if os.path.exists("unet_acdc_best.pth") else "unet_acdc.pth"
    elif choice.lower() == "attention":
        model = AttUNet(in_ch=1, out_ch=4).to(device)
        path = "attention_unet_acdc_best.pth" if os.path.exists("attention_unet_acdc_best.pth") else "attention_unet_acdc.pth"
    else:
        raise ValueError("Choose either 'unet' or 'attention'")
    
    # Handle both dict and checkpoint formats
    checkpoint = torch.load(path, map_location=device)
    if isinstance(checkpoint, dict) and 'model' in checkpoint:
        model.load_state_dict(checkpoint['model'])
    else:
        model.load_state_dict(checkpoint)
    
    model.eval()
    return model


def evaluate(model, loader, device, num_classes: int = 4):
    """Evaluation loop with GPU optimization."""
    per_set_dice = []
    per_set_hd95 = []
    per_set_bdice = []
    fg_pred_ratios = []
    fg_gt_ratios = []

    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating"):
            # Handle batch format
            if len(batch) == 3:
                imgs, masks, spacings = batch
                spacings = spacings.numpy()
            else:
                imgs, masks = batch
                spacings = np.tile(np.array([1.0, 1.0], dtype=np.float32), (imgs.shape[0], 1))

            imgs = imgs.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            
            if device == 'cuda':
                imgs = imgs.to(memory_format=torch.channels_last)
            
            # Forward pass
            outputs = model(imgs)
            preds = torch.argmax(outputs, dim=1)

            preds_np = preds.cpu().numpy().astype(np.int32)
            masks_np = masks.cpu().numpy().astype(np.int32)

            # Compute metrics per image
            for i in range(preds_np.shape[0]):
                p = preds_np[i]
                m = masks_np[i]
                spacing_xy = spacings[i]

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


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate UNet with optimized GPU/CPU usage")
    parser.add_argument("--model", choices=["unet", "attention"], default="unet", help="Model to evaluate")
    parser.add_argument("--batch-size", type=int, default=16, help="Evaluation batch size (increased for efficiency)")
    parser.add_argument("--num-workers", type=int, default=8, help="DataLoader workers (CPU cores)")
    parser.add_argument("--size", type=int, nargs=2, default=[256, 256], help="Spatial size H W")
    parser.add_argument("--debug-artifacts", action="store_true", help="Save sample visualizations")
    parser.add_argument("--outdir", type=str, default="eval_debug", help="Output directory")
    
    # Artifact parameters
    parser.add_argument("--noise-std", type=float, default=0.03, help="Noise std")
    parser.add_argument("--motion-deg", type=float, default=10.0, help="Motion degrees")
    parser.add_argument("--motion-trans", type=float, default=5.0, help="Motion translation")
    parser.add_argument("--bias-coeff", type=float, default=0.3, help="Bias field coefficient")
    parser.add_argument("--spike-intensity-min", type=float, default=0.8, help="Spike min intensity")
    parser.add_argument("--spike-intensity-max", type=float, default=1.2, help="Spike max intensity")
    parser.add_argument("--spike-num", type=int, default=5, help="Number of spikes")
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    print("\n" + "="*70)
    print("🔍 IMPROVED EVALUATION WITH GPU/CPU OPTIMIZATION")
    print("="*70)
    
    # Device setup
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n📍 Device: {device}")
    if device == 'cuda':
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
        print(f"   VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        torch.cuda.reset_peak_memory_stats()
    
    print(f"   Batch size: {args.batch_size}")
    print(f"   Workers: {args.num_workers}")
    
    # Optimize PyTorch
    torch.backends.cudnn.benchmark = True
    
    # Load model
    print(f"\n🧠 Loading {args.model} model...")
    model = load_model(args.model, device)
    if device == 'cuda':
        model = model.to(memory_format=torch.channels_last)
    print("   ✓ Model loaded")
    
    # Evaluation function
    def evaluate_set(name, transform, description=""):
        print(f"\n📊 Evaluating on {name.upper()} test set... {description}")
        ds = ACDCDataset("data/testing", transform=transform, return_meta=True, slice_policy='center')
        loader = DataLoader(
            ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True if device == 'cuda' else False,
            persistent_workers=True if args.num_workers > 0 else False,
        )
        
        dice, hd, bd, fg_pred, fg_gt = evaluate(model, loader, device)
        return dice, hd, bd, fg_pred, fg_gt
    
    results = {}
    
    # Clean evaluation
    print("\n" + "-"*70)
    clean_transform = tio.Compose([
        tio.CropOrPad((args.size[0], args.size[1], 1)),
    ])
    dice_clean, hd_clean, bd_clean, fg_pred_clean, fg_gt_clean = evaluate_set(
        "clean", clean_transform, "(baseline - no artifacts)"
    )
    results['clean'] = (dice_clean, hd_clean, bd_clean, fg_pred_clean, fg_gt_clean)
    
    # Artifact evaluations
    print("\n" + "-"*70)
    artifact_transforms = make_artifact_transforms(args)
    for name, transform in artifact_transforms.items():
        full_transform = tio.Compose([
            transform,
            tio.CropOrPad((args.size[0], args.size[1], 1)),
        ])
        dice, hd, bd, fg_pred, fg_gt = evaluate_set(
            name, full_transform, f"(severity: {name})"
        )
        results[name] = (dice, hd, bd, fg_pred, fg_gt)
    
    # Print results table
    print("\n" + "="*70)
    print("📈 PERFORMANCE COMPARISON")
    print("="*70)
    print(f"{'Test Set':<18} | {'Dice':<10} | {'HD95 (mm)':<12} | {'Boundary Dice':<15}")
    print("-"*70)
    print(f"{'Clean':<18} | {dice_clean:<10.4f} | {hd_clean:<12.2f} | {bd_clean:<15.4f}")
    for name, (dice, hd, bd, _, _) in results.items():
        if name != 'clean':
            print(f"{name:<18} | {dice:<10.4f} | {hd:<12.2f} | {bd:<15.4f}")
    print("="*70)
    
    # Performance drop analysis
    print("\n📉 ROBUSTNESS ANALYSIS")
    print("-"*70)
    for name, (dice, hd, bd, _, _) in results.items():
        if name != 'clean':
            dice_drop = ((dice_clean - dice) / (dice_clean + 1e-8)) * 100
            hd_increase = ((hd - hd_clean) / (hd_clean + 1e-8)) * 100
            print(f"{name:<18} | Dice drop: {dice_drop:>6.1f}% | HD95 increase: {hd_increase:>6.1f}%")
    
    # Save results
    results_json = {
        'clean': {
            'dice': float(dice_clean),
            'hd95': float(hd_clean),
            'boundary_dice': float(bd_clean),
        }
    }
    for name, (dice, hd, bd, _, _) in results.items():
        if name != 'clean':
            results_json[name] = {
                'dice': float(dice),
                'hd95': float(hd),
                'boundary_dice': float(bd),
            }
    
    with open('evaluation_results.json', 'w') as f:
        json.dump(results_json, f, indent=2)
    
    print("\n✅ Results saved to evaluation_results.json")
    
    # GPU memory stats
    if device == 'cuda':
        print(f"\n💾 GPU Memory Usage:")
        print(f"   Peak: {torch.cuda.max_memory_allocated(device) / 1e9:.2f} GB")
        print(f"   Current: {torch.cuda.memory_allocated(device) / 1e9:.2f} GB")
    
    print("\n✅ Evaluation complete!")


if __name__ == "__main__":
    import torch.multiprocessing
    torch.multiprocessing.set_start_method('spawn', force=True)
    main()
