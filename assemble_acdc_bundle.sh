#!/usr/bin/env bash
set -e

# Directory to hold the assembled bundle
BUNDLE_DIR="${PWD}/acdc_bundle"
mkdir -p "$BUNDLE_DIR"
echo "Creating bundle at: $BUNDLE_DIR"

# 1) Clone repositories (if already cloned, skip or pull latest)
REPOS_DIR="${PWD}/repos"
mkdir -p "$REPOS_DIR"
cd "$REPOS_DIR"

if [ ! -d "Pytorch-UNet" ]; then
  git clone https://github.com/milesial/Pytorch-UNet.git
else
  echo "Pytorch-UNet already cloned."
fi

if [ ! -d "Attention-Unet-Pytorch" ]; then
  git clone https://github.com/YigitEkin/Attention-Unet-Pytorch.git
else
  echo "Attention-Unet-Pytorch already cloned."
fi

if [ ! -d "Image_Segmentation" ]; then
  git clone https://github.com/LeeJunHyun/Image_Segmentation.git
else
  echo "Image_Segmentation already cloned."
fi

# 2) Copy relevant folders into bundle
cd "$BUNDLE_DIR"

# Copy full repos to allow you to inspect examples & train scripts,
# but place them in separate subfolders to avoid file conflicts.
cp -r "$REPOS_DIR/Pytorch-UNet" ./unet_repo
cp -r "$REPOS_DIR/Attention-Unet-Pytorch" ./att_unet_repo
cp -r "$REPOS_DIR/Image_Segmentation" ./image_segmentation_repo

# 3) Add a requirements.txt to help create an env
cat > requirements.txt <<EOF
# PyTorch - install appropriate CUDA/cuDNN version for your system
torch
torchvision

# Medical/image libs
torchio
monai
nibabel
scikit-image

# Utilities
numpy
scipy
pandas
matplotlib
tqdm
opencv-python
EOF

# 4) Add a small README with instructions on adaptation for ACDC
cat > README_FOR_ACDC.md <<'EOF'
ACDC Bundle - quick guide
-------------------------

What was copied:
- unet_repo/             -> milesial Pytorch-UNet (clean UNet baseline + training scripts)
- att_unet_repo/         -> Attention U-Net implementation and examples
- image_segmentation_repo/ -> multiple variants (U-Net, R2U-Net, Attention variants)

Quick edits to run on ACDC:
1) Data loader:
   - Write or reuse a dataset loader that yields (image_tensor, mask_tensor).
   - ACDC images are single-channel (grayscale) slices. Ensure input channel = 1.
   - Masks: set up classes for background + LV + RV + MYO. Adjust num_classes/out_channels.

2) Change model outputs:
   - In model definition files, change final conv out_channels to match #classes (e.g. 4 = background + 3 structures).

3) Training:
   - Use CrossEntropyLoss (for integer label masks) combined with dice loss if desired.
   - Ensure training script uses the same transforms normalisation as your dataloader.

4) Artifacted test sets:
   - Use TorchIO / MONAI to create versions of the test set with noise, motion, bias-field, slice dropout. See TorchIO docs for RandomMotion, RandomBiasField, AdditiveRicianNoise.

5) Evaluation:
   - Save predictions as numpy arrays or nifti volumes.
   - Compute metrics: per-class Dice, HD95 (scipy.spatial or medpy), boundary Dice (morphological erosion).

Where to look in each repo:
- unet_repo/: model and train examples (entry points vary; inspect README in that folder)
- att_unet_repo/: look for model definition files (often named attention_unet.py or models/*.py)
- image_segmentation_repo/: many variants + example training loops

License & citation:
- These are public repos under their own licenses. Check each repo's LICENSE before using for publication.
EOF

echo "Bundle assembled at: $BUNDLE_DIR"
echo "To create environment: python -m venv venv && source venv/bin/activate && pip install -r $BUNDLE_DIR/requirements.txt"
echo "Done."
