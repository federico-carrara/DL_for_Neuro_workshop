import os
import re
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
    pattern = list(raw.keys())[4].split("_")[0]
    data = []
    for i in range(15):
        data.append(torch.tensor(raw[f"{pattern}_eeg{i + 1}"]))
    return data
#--------------------------------------------------------------------------

#--------------------------------------------------------------------------
def windowize_data(
    data: List[torch.tensor],
    length: int,
    overlap: int
) -> np.ndarray: 

    assert overlap < length, f"Window overlapping {overlap} cannot be greater than the window length {length}."

    windowized_data = []
    for signal in data:
        ptr = 0
        while ptr + length < signal.shape[1]:
            windowized_data.append(signal[:, ptr:(ptr + length)])
            ptr += length - overlap

    return windowized_data
#--------------------------------------------------------------------------



#--------------------------------------------------------------------------
class TrainDataset(Dataset):
    def __init__(
        self, 
        path_to_data_dir: str,
        path_to_labels: str,
        win_length: int,
        win_overlap: int,
        do_augmentation: Optional[bool] = True
    ):
        # Notice: N_subjects = 15, N_recordings=45 (N_subjects*3)
        self.N_subjects = 15
        self.N_recordings = 45
        self.win_length = win_length
        self.win_overlap = win_overlap

        # Load data store them in a torch tensor
        eeg_files = [fname for fname in os.listdir(path_to_data_dir) if fname[0] in string.digits]
        data = []
        for eeg_file in eeg_files:
            curr_data = load_eeg_data_file(os.path.join(path_to_data_dir, eeg_file))
            curr_windowized_data = windowize_data(curr_data, self.win_length, self.win_overlap)
            data.append(curr_windowized_data)
        self.data = torch.tensor(data)

        # Load labels
        labels_dict = loadmat(path_to_labels)
        labels = torch.tensor(labels_dict["label"])
        self.labels = labels.repeat(self.N_recordings, 1)
        
        # Boolean: if true apply transformation for data augmentation
        if do_augmentation:
            self.transform = add_gaussian_noise
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

    def add_gaussian_noise(self):
        return

#--------------------------------------------------------------------------

def plot_eeg_data(
    data: np.ndarray
) -> None:
    return 



def find_bad_channels(
    data: np.ndarray
) -> List[int]: