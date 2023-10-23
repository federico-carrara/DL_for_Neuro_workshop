import torch
import h5py
from torch.utils.data import Dataset

"""
In this script we create a simplified version of our Datasets in which we directly load pre-processed data.
"""

class SEEDDataset(Dataset):
    def __init__(
            self,
            path_to_preprocessed: str,
            split: str,       
    ) -> None:
        super().__init__() 

        self.path_to_preprocessed = path_to_preprocessed
        self.split = split

        # Load preprocessed data to save time
        print(f"Loading preprocessed {self.split} data at {self.path_to_preprocessed}...")
        with h5py.File(self.path_to_preprocessed, "r") as file:
            print(f"Window length: {self.win_length}")
            self.data = torch.tensor(file["data"][:], dtype=torch.float32)
            self.labels = torch.tensor(file["labels"][:], dtype=torch.int64)
            self.trial_ids = torch.tensor(file["trial_ids"][:], dtype=torch.int64)
            self.N_samples = self.data.shape[0]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        sample = self.data[index]
        label = self.labels[index]
        trial_id = self.trial_ids[index]

        return sample, label, trial_id