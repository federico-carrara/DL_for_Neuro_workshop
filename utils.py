import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.io import loadmat
from typing import List, Optional, Tuple

"""
Legend for tensors dimensions:
    - B = batch size 
    - C = number of channels in the network layers.
    - M = number of EEG channels.
    - W = number of time points in a signal window.
"""

#--------------------------------------------------------------------------
def load_eeg_data_file(
    path_to_file: str
) -> torch.Tensor:
    raw = loadmat(path_to_file)
    pattern = list(raw.keys())[4].split("_")[0]
    data = []
    for i in range(15):
        data.append(torch.tensor(raw[f"{pattern}_eeg{i + 1}"]))
    return data
#--------------------------------------------------------------------------

#--------------------------------------------------------------------------
def windowize_signal(
    data: List[torch.Tensor],
    length: int,
    overlap: int
) -> Tuple[List[torch.Tensor], List[int]]: 

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
def add_gaussian_noise(data: torch.Tensor):
    noise = torch.normal(0, 0.2, size=(data.shape))
    return data + noise
#--------------------------------------------------------------------------

#--------------------------------------------------------------------------
def shuffle_channels(data: torch.Tensor):
    permutation = torch.randperm(data.shape[0])
    return data[permutation, :]
#--------------------------------------------------------------------------

#--------------------------------------------------------------------------
def plot_eeg_data(
    data: np.ndarray
) -> None:
    
    return 
#--------------------------------------------------------------------------