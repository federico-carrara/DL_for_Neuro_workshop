import os
import string
import torch
from scipy.io import loadmat
from random import random
from tqdm import tqdm
from torch.utils.data import Dataset
from typing import Optional, List, Tuple
from utils import (
    load_eeg_data_file, 
    windowize_signal, 
    shuffle_channels, 
    add_gaussian_noise,
    bandpower_diff_entropy
)

#--------------------------------------------------------------------------
class TrainDataset(Dataset):
    def __init__(
        self, 
        path_to_data_dir: str,
        path_to_labels: str,
        win_length: int,
        win_overlap: int, 
        num_recordings: Optional[int] = None,
        num_movies: Optional[int] = 15,
        num_channels: Optional[int] = 62,
        sampling_frequency: Optional[int] = 200,
        data_augmentation: Optional[bool] = True,
        preprocess_de: Optional[bool] = True,
        band_frequencies: Optional[List[Tuple[int, int]]] = [(4, 7), (8, 13), (14, 30), (31, 50)] 
    ):
    
        if num_recordings:
            self.N_recordings = num_recordings
        else:
            self.N_recordings = len(os.listdir(path_to_data_dir))
        self.N_movies = num_movies
        self.N_channels = num_channels
        self.sampl_freq = sampling_frequency
        self.win_length = win_length
        self.win_overlap = win_overlap

        # Load data and labels, store them in torch tensors
        eeg_files = [fname for fname in os.listdir(path_to_data_dir) if fname[0] in string.digits]
        labels_dict = loadmat(path_to_labels)
        labels_unique = labels_dict["label"].squeeze(0)
        data = []
        labels = []
        for eeg_file in tqdm(eeg_files, desc="Loading training data files"):
            curr_data = load_eeg_data_file(
                path_to_file=os.path.join(path_to_data_dir, eeg_file),
                num_keys=self.N_movies,
            )
            curr_windowized_data, num_windows = windowize_signal(curr_data, self.win_length, self.win_overlap)
            data.extend(curr_windowized_data)
            curr_labels = [
                torch.tensor([labels_unique[i]] * num_windows[i])
                for i in range(len(curr_data))
            ]
            labels.append(torch.concat(curr_labels, axis=0))
        data = [tens.unsqueeze(0) for tens in data]
        self.data = torch.concat(data, axis=0).unsqueeze(1)
        self.data = self.data.type(torch.float32)
        self.labels = torch.concat(labels)
        self.labels += 1
        self.labels = self.labels.type(torch.int64)
        self.N_samples = self.data.shape[0]

        assert len(self.labels) == len(self.data), "Data and labels are not paired..."
        
        # Boolean: if true apply transformation for data augmentation
        self.data_augmentation = data_augmentation

        # Preprocessing: apply Differential Entropy on single channels
        if preprocess_de:
            # Stack eeg windows
            self.data = self.data.reshape(-1, self.win_length)
            self.band_freqs = band_frequencies
            self.data = torch.stack([
                bandpower_diff_entropy(self.data, self.sampl_freq, band_freq)
                for band_freq in tqdm(self.band_freqs, desc="Computing DE on band:")
            ])
            self.data = self.data.reshape(-1, self.N_samples, self.N_channels)
            self.data = self.data.swapaxes(0, 1)
            self.data = self.data.reshape(self.N_samples, -1)

            assert len(self.labels) == len(self.data), "Data and labels are not paired..."

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        sample = self.data[index]
        label = self.labels[index]

        if self.data_augmentation:
            if random() > 0.2:
                sample = shuffle_channels(sample)
            if random() > 0.2:
                sample = add_gaussian_noise(sample)

        return sample, label
#--------------------------------------------------------------------------

#--------------------------------------------------------------------------
class ValidationDataset(Dataset):
    def __init__(
        self, 
        path_to_data_dir: str,
        path_to_labels: str,
        win_length: int,
        win_overlap: int, 
        num_recordings: Optional[int] = None,
        num_movies: Optional[int] = 15,
        num_channels: Optional[int] = 62,
        sampling_frequency: Optional[int] = 200,
        preprocess_de: Optional[bool] = True,
        band_frequencies: Optional[List[Tuple[int, int]]] = [(4, 7), (8, 13), (14, 30), (31, 50)] 
    ):

        if num_recordings:
            self.N_recordings = num_recordings
        else:
            self.N_recordings = len(os.listdir(path_to_data_dir))
        self.N_movies = num_movies
        self.N_channels = num_channels
        self.sampl_freq = sampling_frequency
        self.win_length = win_length
        self.win_overlap = win_overlap

        # Load data and labels, store them in torch tensors
        eeg_files = [fname for fname in os.listdir(path_to_data_dir) if fname[0] in string.digits]
        labels_dict = loadmat(path_to_labels)
        labels_unique = labels_dict["label"].squeeze(0)
        data = []
        labels = []
        for eeg_file in tqdm(eeg_files, desc="Loading validation data files"):
            curr_data = load_eeg_data_file(
                path_to_file=os.path.join(path_to_data_dir, eeg_file),
                num_keys=self.N_movies,
            )
            curr_windowized_data, num_windows = windowize_signal(curr_data, self.win_length, self.win_overlap)
            data.extend(curr_windowized_data)
            curr_labels = [
                torch.tensor([labels_unique[i]] * num_windows[i])
                for i in range(len(curr_data))
            ]
            labels.append(torch.concat(curr_labels, axis=0))
        data = [tens.unsqueeze(0) for tens in data]
        self.data = torch.concat(data, axis=0).unsqueeze(1)
        self.data = self.data.type(torch.float32)
        self.labels = torch.concat(labels)
        self.labels += 1
        self.labels = self.labels.type(torch.int64)
        self.N_samples = self.data.shape[0]

        assert len(self.labels) == len(self.data), "Data and labels are not paired..."

        # Preprocessing: apply Differential Entropy on single channels
        if preprocess_de:
            # Stack eeg windows
            self.data = self.data.reshape(-1, self.win_length)
            self.band_freqs = band_frequencies
            self.data = torch.stack([
                bandpower_diff_entropy(self.data, self.sampl_freq, band_freq)
                for band_freq in tqdm(self.band_freqs, desc="Computing DE on band:")
            ])
            self.data = self.data.reshape(-1, self.N_samples, self.N_channels)
            self.data = self.data.swapaxes(0, 1)
            self.data = self.data.reshape(self.N_samples, -1)

            assert len(self.labels) == len(self.data), "Data and labels are not paired..."

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        sample = self.data[index]
        label = self.labels[index]

        return sample, label
#--------------------------------------------------------------------------

#--------------------------------------------------------------------------
class TestDataset(Dataset):
    def __init__(
        self, 
        path_to_data_dir: str,
        path_to_labels: str,
        win_length: int,
        win_overlap: int, 
        num_recordings: Optional[int] = None,
        num_movies: Optional[int] = 15,
        num_channels: Optional[int] = 62,
        sampling_frequency: Optional[int] = 200,
        preprocess_de: Optional[bool] = True,
        band_frequencies: Optional[List[Tuple[int, int]]] = [(4, 7), (8, 13), (14, 30), (31, 50)] 
    ):

        if num_recordings:
            self.N_recordings = num_recordings
        else:
            self.N_recordings = len(os.listdir(path_to_data_dir))
        self.N_movies = num_movies
        self.N_channels = num_channels
        self.sampl_freq = sampling_frequency
        self.win_length = win_length
        self.win_overlap = win_overlap

        # Load data and labels, store them in torch tensors
        eeg_files = [fname for fname in os.listdir(path_to_data_dir) if fname[0] in string.digits]
        labels_dict = loadmat(path_to_labels)
        labels_unique = labels_dict["label"].squeeze(0)
        data = []
        labels = []
        for eeg_file in tqdm(eeg_files, desc="Loading test data files"):
            curr_data = load_eeg_data_file(
                path_to_file=os.path.join(path_to_data_dir, eeg_file),
                num_keys=self.N_movies,
            )
            curr_windowized_data, num_windows = windowize_signal(curr_data, self.win_length, self.win_overlap)
            data.extend(curr_windowized_data)
            curr_labels = [
                torch.tensor([labels_unique[i]] * num_windows[i])
                for i in range(len(curr_data))
            ]
            labels.append(torch.concat(curr_labels, axis=0))
        data = [tens.unsqueeze(0) for tens in data]
        self.data = torch.concat(data, axis=0).unsqueeze(1)
        self.data = self.data.type(torch.float32)
        self.labels = torch.concat(labels)
        self.labels += 1
        self.labels = self.labels.type(torch.int64)
        self.N_samples = self.data.shape[0]

        assert len(self.labels) == len(self.data), "Data and labels are not paired..."

        # Preprocessing: apply Differential Entropy on single channels
        if preprocess_de:
            # Stack eeg windows
            self.data = self.data.reshape(-1, self.win_length)
            self.band_freqs = band_frequencies
            self.data = torch.stack([
                bandpower_diff_entropy(self.data, self.sampl_freq, band_freq)
                for band_freq in tqdm(self.band_freqs, desc="Computing DE on band:")
            ])
            self.data = self.data.reshape(-1, self.N_samples, self.N_channels)
            self.data = self.data.swapaxes(0, 1)
            self.data = self.data.reshape(self.N_samples, -1)

            assert len(self.labels) == len(self.data), "Data and labels are not paired..."

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        sample = self.data[index]
        label = self.labels[index]

        return sample, label
#--------------------------------------------------------------------------