import torch
from torch.utils.data import Dataset


class MissingnessClassificationDataset(Dataset):
    """
    Wraps SyntheticSpectralDataset to add classification labels.

    Task:
        y = 1 if missing_fraction > threshold else 0
    """

    def __init__(self, base_dataset, threshold: float = 0.2):
        self.base_dataset = base_dataset
        self.threshold = threshold

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        sample = self.base_dataset[idx]

        missing_frac = sample["meta"]["missing_fraction"]
        y = int(missing_frac > self.threshold)

        sample["y"] = torch.tensor(y, dtype=torch.long)
        return sample
