import torchio as tio
from dataset.acdc_dataset import ACDCDataset
from torch.utils.data import DataLoader

# ----------------------
# Artifact Transforms
# ----------------------
artifact_transforms = {
    "noise": tio.RandomNoise(mean=0, std=(0.01, 0.05)),                  # Gaussian noise
    "motion": tio.RandomMotion(degrees=10, translation=5),               # Patient motion
    "bias": tio.RandomBiasField(coefficients=0.3),                       # Intensity inhomogeneity
    "slice_dropout": tio.RandomSpike(intensity=(0.8, 1.2), num_spikes=5), # Slice dropout
    # ----------------------------
    # Advanced artifacts
    # ----------------------------
    "ghosting": tio.RandomGhosting(num_ghosts=(1, 3), axes=(0,1)),      # MRI ghosting artifacts
    "blur": tio.RandomBlur(std=(0.5, 1.5)),                              # Slight out-of-focus / motion blur
    "intensity_shift": tio.RandomBiasField(coefficients=0.6),            # Strong intensity inhomogeneity
    "spike_noise": tio.RandomSpike(num_spikes=(5,10), intensity=(0.7,1.2)), # Sudden slice corruption
    "combined": tio.Compose([                                            # Combination of 2-3 artifacts
        tio.RandomNoise(mean=0, std=(0.01,0.03)),
        tio.RandomMotion(degrees=5, translation=3),
        tio.RandomBiasField(coefficients=0.2)
    ])
}

# ----------------------
# Function to create artifact datasets and loaders
# ----------------------
def create_artifact_datasets(base_path="data/testing", transforms=artifact_transforms, batch_size=2):
    """
    Returns:
        datasets: dict of ACDCDataset instances
        loaders: dict of DataLoader instances
    """
    datasets = {}
    loaders = {}
    
    for name, transform in transforms.items():
        datasets[name] = ACDCDataset(base_path, transform=transform)
        loaders[name] = DataLoader(datasets[name], batch_size=batch_size, shuffle=False, num_workers=0)
    
    return datasets, loaders

# ----------------------
# Example usage
# ----------------------
if __name__ == "__main__":
    artifact_datasets, artifact_loaders = create_artifact_datasets()
    print("Artifact datasets created:", list(artifact_datasets.keys()))
