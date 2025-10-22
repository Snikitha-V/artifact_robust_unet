import os
import torch
import nibabel as nib
import numpy as np
import torchio as tio
from torch.utils.data import Dataset

class ACDCDataset(Dataset):
    def __init__(self, root_dir, transform=None, return_meta: bool = False, slice_policy: str = 'random'):
        """
        slice_policy:
            - 'random': pick a random non-empty slice
            - 'center': pick the center slice among non-empty slices (deterministic)
        return_meta:
            - if True, returns (image, mask, spacing_xy) where spacing_xy=(sx, sy) in mm
        """
        self.root_dir = root_dir
        self.transform = transform
        self.return_meta = return_meta
        assert slice_policy in ('random', 'center', 'largest'), "slice_policy must be 'random', 'center', or 'largest'"
        self.slice_policy = slice_policy
        self.image_paths = []
        self.mask_paths = []

        for p in sorted(os.listdir(root_dir)):
            patient_path = os.path.join(root_dir, p)
            if not os.path.isdir(patient_path):
                continue
            for f in os.listdir(patient_path):
                if f.endswith(".nii.gz") and "gt" not in f:
                    img_path = os.path.join(patient_path, f)
                    gt_path = img_path.replace(".nii.gz", "_gt.nii.gz")
                    if os.path.exists(gt_path):
                        self.image_paths.append(img_path)
                        self.mask_paths.append(gt_path)

        print(f"Found {len(self.image_paths)} ACDC volumes.")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        mask_path = self.mask_paths[idx]

        img_nii = nib.load(img_path)
        img = img_nii.get_fdata().astype(np.float32)
        mask = nib.load(mask_path).get_fdata().astype(np.int64)
        zooms = img_nii.header.get_zooms()
        spacing_xy = np.array(zooms[:2], dtype=np.float32)  # (sx, sy)

        img = (img - np.mean(img)) / (np.std(img) + 1e-8)
        slices = []
        slice_masks = []
        for i in range(img.shape[2]):
            im = img[:, :, i]
            ms = mask[:, :, i]
            if np.max(ms) == 0:
                continue
            im = np.expand_dims(im, axis=0)
            ms = np.expand_dims(ms, axis=0)
            slices.append(im)
            slice_masks.append(ms)

        if self.slice_policy == 'random':
            idx = np.random.randint(0, len(slices))
        elif self.slice_policy == 'center':
            idx = len(slices) // 2
        else:  # 'largest'
            # choose slice with largest foreground area
            areas = [np.count_nonzero(ms) for ms in slice_masks]
            idx = int(np.argmax(areas)) if areas else 0
        img_slice, mask_slice = slices[idx], slice_masks[idx]

        sample = {
            'image': torch.tensor(img_slice, dtype=torch.float32),  # (1, H, W)
            'mask': torch.tensor(mask_slice, dtype=torch.long).squeeze(0)  # (H, W)
        }

        if self.transform:
            # TorchIO expects tensors of shape (C, X, Y, Z). For 2D slices, use Z=1.
            img_4d = sample['image'].unsqueeze(-1)        # (1, H, W, 1)
            mask_4d = sample['mask'].unsqueeze(0).unsqueeze(-1)  # (1, H, W, 1)

            tio_subject = tio.Subject(
                img=tio.ScalarImage(tensor=img_4d),
                mask=tio.LabelMap(tensor=mask_4d)
            )

            transformed = self.transform(tio_subject)

            # Back to training shapes: image -> (1, H, W); mask -> (H, W)
            sample['image'] = transformed.img.data.squeeze(-1)             # (1, H, W)
            sample['mask'] = transformed.mask.data.squeeze(-1).squeeze(0).long()  # (H, W) long

        if self.return_meta:
            return sample['image'], sample['mask'], torch.from_numpy(spacing_xy)
        else:
            return sample['image'], sample['mask']
