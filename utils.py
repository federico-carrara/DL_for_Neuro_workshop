import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.io import loadmat
from typing import List, Optional

def load_eeg_data_per_subject(
    path_to_file: str
) -> np.ndarray:
    raw = loadmat(path_to_file)
    dims = raw["djc_eeg1"].shape
    data = np.empty((15, dims[0], dims[1]))
    for i in range(15):
        data[i] = raw[f"djc_eeg{i + 1}"]
    return data

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