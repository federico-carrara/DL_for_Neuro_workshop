import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.io import loadmat
from scipy.signal import butter, filtfilt
from typing import List, Optional, Tuple, Dict

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

#--------------------------------------------------------------------------
def format_channel_location_dict(channel_list, location_list):
    location_list = np.array(location_list)
    output = {}
    for channel in channel_list:
        if len(np.argwhere(location_list == channel)):
            location = (np.argwhere(location_list == channel)[0]).tolist()
            output[channel] = location
        else:
            output[channel] = None
    return output
#--------------------------------------------------------------------------

#--------------------------------------------------------------------------
class ToGrid:
    def __init__(
            self,
            channel_name_list: List[str],
            channel_location_list: List[List[str]],
    ):
        self.channel_location_dict = format_channel_location_dict(
            channel_list=channel_name_list,
            location_list=channel_location_list
        )
        self.num_channels = len(channel_name_list)
        self.width = len(channel_location_list[0]) 
        self.height = len(channel_location_list)

    def apply(
            self, 
            input: np.ndarray
    ) -> np.ndarray:
        
        # input shape: (num_records, num_features, num_channels, )
        num_records = input.shape[0]
        num_features = input.shape[1]

        # output shape: (num_records, num_features, height, width, ) 
        outputs = np.zeros([num_records, num_features, self.height, self.width])
        for i, locs in enumerate(self.channel_location_dict.values()):
            if locs is None:
                continue
            (loc_y, loc_x) = locs
            outputs[:, :, loc_y, loc_x] = input[:, :, i]

        return outputs