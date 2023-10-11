import os
import re
import string
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from random import random
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from scipy.io import loadmat
from typing import List, Optional, Tuple

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
def windowize_signal(
    data: List[torch.tensor],
    length: int,
    overlap: int
) -> Tuple[List[torch.tensor], List[int]]: 

    assert overlap < length, f"Window overlapping {overlap} cannot be greater than the window length {length}."

    windowized_data = []
    num_windows = []
    for signal in data:
        ptr = 0
        curr_windowized_data = []
        while ptr + length < signal.shape[1]:
            curr_windowized_data.append(signal[:, ptr:(ptr + length)])
            ptr += length - overlap
        num_windows.append(len(curr_windowized_data))
        windowized_data.extend(curr_windowized_data)

    return windowized_data, num_windows
#--------------------------------------------------------------------------

#--------------------------------------------------------------------------
def add_gaussian_noise(data: torch.tensor):
    noise = torch.normal(0, 0.2, size=(data.shape[0], data.shape[1]))
    return data + noise
#--------------------------------------------------------------------------

#--------------------------------------------------------------------------
def shuffle_channels(data: torch.tensor):
    permutation = torch.randperm(data.shape[0])
    return data[permutation, :]
#--------------------------------------------------------------------------

#--------------------------------------------------------------------------
class TrainDataset(Dataset):
    def __init__(
        self, 
        path_to_data_dir: str,
        path_to_labels: str,
        win_length: int,
        win_overlap: int,
        transform: Optional[bool] = True
    ):
        # Notice: N_subjects = 15, N_recordings=45 (N_subjects*3)
        self.N_subjects = 15
        self.N_recordings = 45
        self.N_channels = 62
        self.win_length = win_length
        self.win_overlap = win_overlap

        # Load data and labels, store them in torch tensors
        eeg_files = [fname for fname in os.listdir(path_to_data_dir) if fname[0] in string.digits]
        labels_dict = loadmat(path_to_labels)
        labels = labels_dict["label"].squeeze(0)
        data = []
        for eeg_file in tqdm(eeg_files, desc="Loading data files"):
            curr_data = load_eeg_data_file(os.path.join(path_to_data_dir, eeg_file))
            curr_windowized_data, num_windows = windowize_signal(curr_data, self.win_length, self.win_overlap)
            data.extend(curr_windowized_data)
            curr_labels = [torch.tensor([labels[i]] * num_windows[i]) for i in range(len(curr_data))]
            curr_labels = torch.concat(curr_labels, axis=0)
        data = [tens.unsqueeze(0) for tens in data]
        self.data = torch.concat(data, axis=0)
        self.labels = curr_labels.repeat(self.N_recordings)
        
        # Boolean: if true apply transformation for data augmentation
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        sample = self.data[index]
        label = self.labels[index]

        if self.transform:
            if random() > 0.2:
                sample = shuffle_channels(sample)
            if random() > 0.2:
                sample = add_gaussian_noise(sample)

        return sample, label
#--------------------------------------------------------------------------

#--------------------------------------------------------------------------
def plot_eeg_data(
    data: np.ndarray
) -> None:
    return 
#--------------------------------------------------------------------------

#--------------------------------------------------------------------------
class EEGNET(nn.Module):
    def __init__(
            self, 
            num_classes = 3,
            num_out_channels = 16,
            temporal_kernel_size = 64,
            spatial_kernel_size = 8, 
            separable_kernel_size = 16,
            pooling_size = (2, 5), 
            dropout_prob = 0.5, 
    ):
        super(EEGNET, self).__init__()
        
        self.temporal_conv = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=num_out_channels,
                kernel_size=[1, temporal_kernel_size],
                stride=1, 
                bias=True,
                padding="same"
            ), 
            nn.BatchNorm2d(num_features=num_out_channels),
        )
        self.spatial_conv = nn.Sequential(
            nn.Conv2d(
                in_channels=num_out_channels,
                out_channels=num_out_channels * 2,
                kernel_size=[spatial_kernel_size, 1],
                stride=1,
                bias=True,
                padding="valid"
            ),
            nn.BatchNorm2d(num_features=num_out_channels * 2),
            nn.ReLU(True),
        )

        self.avg_pool = nn.AvgPool2d(
            kernel_size=pooling_size, 
            stride=pooling_size, 
            padding=0
        )

        self.seperable_conv = nn.Sequential(
            nn.Conv2d(
                in_channels=num_out_channels * 2,
                out_channels=num_out_channels * 2,
                kernel_size=[1, separable_kernel_size],
                padding="same",
                bias=True
            ),
            nn.Conv2d(
                in_channels=num_out_channels * 2,
                out_channels=num_out_channels * 2,
                kernel_size=[1, 1], 
                padding="same",
                bias=True
            ),
            nn.BatchNorm2d(num_features=num_out_channels * 2),
            nn.ReLU(True),
        )

        self.dropout = nn.Dropout(dropout_prob)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Sequential(
            nn.Linear(, ),
            nn.ReLU(), 
        )
        self.fc2 = nn.Sequential(
            nn.Linear(, ),
        )

    def forward(self, x):
        x = self.temporal_conv(x)
        x = self.spatial_conv(x)
        x = self.avg_pool(x)
        x = self.dropout(x)
        x = self.seperable_conv(x)
        x = self.avg_pool(x)
        x = self.dropout(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.dropout(x)
        out = self.fc2(x)
        return out
