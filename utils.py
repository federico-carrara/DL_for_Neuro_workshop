import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import string
from torch.utils.data import Dataset, DataLoader
from scipy.io import loadmat
from typing import List, Optional

#--------------------------------------------------------------------------
def load_eeg_data_file(
    path_to_file: str
) -> torch.tensor:
    raw = loadmat(path_to_file)
    dims = raw["djc_eeg1"].shape
    data = torch.empty((15, dims[0], dims[1]), dtype=torch.float64)
    for i in range(15):
        data[i] = raw[f"djc_eeg{i + 1}"]
    return data
#--------------------------------------------------------------------------

#--------------------------------------------------------------------------
class TrainDataset(Dataset):
    def __init__(
        self, 
        path_to_data_dir: str,
        path_to_labels: str,
        do_augmentation: Optional[bool] = True
    ):
        # Notice: N_subjects = 15, N_recordings=45 (N_subjects*3)
        self.N_subjects = 15
        self.N_recordings = 45

        # Load data store them in a torch tensor
        eeg_files = [fname for fname in os.listdir(path_to_data_dir) if fname[0] in string.digits]
        data = []
        for eeg_file in eeg_files:
            curr_data = load_eeg_data_file(os.path.join(path_to_data_dir, eeg_file))
            data.append(curr_data)
        self.data = torch.tensor(data)

        # Load labels
        labels_dict = loadmat(path_to_labels)
        labels = torch.tensor(labels_dict["label"])
        self.labels = labels.repeat(self.N_recordings, 1)
        
        # Boolean: if true apply transformation for data augmentation
        if do_augmentation:
            self.transform = transform
        else:
            self.transform = None

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        sample = self.data[index]
        label = self.labels[index]

        if self.transform:
            sample = self.transform(sample)

        return sample, label

    def transform(self):
        
#--------------------------------------------------------------------------

def plot_eeg_data(
    data: np.ndarray
) -> None:
    return 

def windowize_data(
    data: np.ndarray,
    length: int,
    overlap: int
) -> np.ndarray: 
    return

def find_bad_channels(
    data: np.ndarray
) -> List[int]: