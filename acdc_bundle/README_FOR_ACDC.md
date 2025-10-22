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
