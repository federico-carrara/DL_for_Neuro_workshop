import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.io import loadmat
from scipy.signal import butter, filtfilt
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
    path_to_file: str,
    num_keys: Optional[int] = 15
) -> torch.Tensor:
    raw = loadmat(path_to_file)
    pattern = list(raw.keys())[4].split("_")[0]
    data = []
    for i in range(num_keys):
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

#--------------------------------------------------------------------------
def set_butter_filter(
        lowcut: int, 
        highcut: int, 
        fs: int, 
        order: Optional[int] = 5
    ):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a
#--------------------------------------------------------------------------

#--------------------------------------------------------------------------
def bandpower_diff_entropy(
        data: torch.Tensor, 
        sampling_freq: int, 
        freq_band: Tuple[int, int],
        filter_order: Optional[int] = 5
):
    b, a = set_butter_filter(
        lowcut=freq_band[0], 
        highcut=freq_band[1], 
        fs=sampling_freq, 
        order=filter_order
    )
    filt_data = filtfilt(b, a, data, axis=1)
    std_devs = np.std(filt_data, axis=1)
    out_data = 1/2 * np.log(2 * np.pi * np.e * std_devs**2)
    out_data = torch.tensor(out_data, dtype=torch.float32)
    return out_data
#--------------------------------------------------------------------------