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
        num_subjects: Optional[int] = 15,
        num_trials: Optional[int] = 3,
        num_channels: Optional[int] = 62,
        sampling_frequency: Optional[int] = 200,
        data_augmentation: Optional[bool] = True,
        preprocess_de: Optional[bool] = True,
        band_frequencies: Optional[List[Tuple[int, int]]] = [(4, 7), (8, 13), (14, 30), (31, 50)] 
    ):
        # Notice: N_subjects = 15, N_recordings=45 (N_subjects*3)
        self.N_subjects = num_subjects
        self.N_recordings = num_subjects * num_trials
        self.N_channels = num_channels
        self.sampl_freq = sampling_frequency
        self.win_length = win_length
        self.win_overlap = win_overlap

        # Load data and labels, store them in torch tensors
        eeg_files = [fname for fname in os.listdir(path_to_data_dir) if fname[0] in string.digits]
        labels_dict = loadmat(path_to_labels)
        labels = labels_dict["label"].squeeze(0)
        data = []
        for eeg_file in tqdm(eeg_files, desc="Loading training data files"):
            curr_data = load_eeg_data_file(os.path.join(path_to_data_dir, eeg_file))
            curr_windowized_data, num_windows = windowize_signal(curr_data, self.win_length, self.win_overlap)
            data.extend(curr_windowized_data)
            curr_labels = [torch.tensor([labels[i]] * num_windows[i]) for i in range(len(curr_data))]
            curr_labels = torch.concat(curr_labels, axis=0)
        data = [tens.unsqueeze(0) for tens in data]
        self.data = torch.concat(data, axis=0).unsqueeze(1)
        self.data = self.data.type(torch.float32)
        self.labels = curr_labels.repeat(self.N_recordings)
        self.labels += 1
        self.labels = self.labels.type(torch.int64)
        
        # Boolean: if true apply transformation for data augmentation
        self.data_augmentation = data_augmentation

        # Preprocessing: apply Differential Entropy on single channels
        if preprocess_de:
            print("-------------------------------------------------------")
            # Stack eeg windows one over the other
            self.data = self.data.reshape(-1, self.win_length)
            self.band_freqs = band_frequencies
            self.data = torch.stack([
                bandpower_diff_entropy(self.data, self.sampl_freq, band_freq)
                for band_freq in tqdm(self.band_freqs, desc="Computing DE on band:")
            ])

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
        num_subjects: Optional[int] = 15,
        num_trials: Optional[int] = 3,
        num_movies: Optional[int] = 15,
        num_channels: Optional[int] = 62,
        sampling_frequency: Optional[int] = 200,
        preprocess_de: Optional[bool] = True,
        band_frequencies: Optional[List[Tuple[int, int]]] = [(4, 7), (8, 13), (14, 30), (31, 50)] 
    ):
        # Notice: N_subjects = 15, N_recordings=45 (N_subjects*3)
        self.N_subjects = num_subjects
        self.N_recordings = num_subjects * num_trials
        self.N_channels = num_channels
        self.N_movies = num_movies
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
        print(f"Data shape before: {self.data.shape}")
        print(f"Labels shape before: {self.labels.shape}")

        # Preprocessing: apply Differential Entropy on single channels
        if preprocess_de:
            print("-------------------------------------------------------")
            # Stack eeg windows one over the other
            self.data = self.data.reshape(-1, self.win_length)
            print(f"Data shape stacked: {self.data.shape}")
            self.band_freqs = band_frequencies
            self.data = torch.stack([
                bandpower_diff_entropy(self.data, self.sampl_freq, band_freq)
                for band_freq in tqdm(self.band_freqs, desc="Computing DE on band:")
            ])
            print(f"Data shape mid through: {self.data.shape}")
            self.data = self.data.reshape(-1, self.N_channels * len(self.band_freqs))
            print(f"Data shape after: {self.data.shape}")

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
        num_subjects: Optional[int] = 15,
        num_trials: Optional[int] = 3,
        num_channels: Optional[int] = 62,
        sampling_frequency: Optional[int] = 200,
        data_augmentation: Optional[bool] = True,
        preprocess_de: Optional[bool] = True,
        band_frequencies: Optional[List[Tuple[int, int]]] = [(4, 7), (8, 13), (14, 30), (31, 50)] 
    ):
        # Notice: N_subjects = 15, N_recordings=45 (N_subjects*3)
        self.N_subjects = num_subjects
        self.N_recordings = num_subjects * num_trials
        self.N_channels = num_channels
        self.sampl_freq = sampling_frequency
        self.win_length = win_length
        self.win_overlap = win_overlap

        # Load data and labels, store them in torch tensors
        eeg_files = [fname for fname in os.listdir(path_to_data_dir) if fname[0] in string.digits]
        labels_dict = loadmat(path_to_labels)
        labels = labels_dict["label"].squeeze(0)
        data = []
        for eeg_file in tqdm(eeg_files, desc="Loading test data files"):
            curr_data = load_eeg_data_file(os.path.join(path_to_data_dir, eeg_file))
            curr_windowized_data, num_windows = windowize_signal(curr_data, self.win_length, self.win_overlap)
            data.extend(curr_windowized_data)
            curr_labels = [torch.tensor([labels[i]] * num_windows[i]) for i in range(len(curr_data))]
            curr_labels = torch.concat(curr_labels, axis=0)
        data = [tens.unsqueeze(0) for tens in data]
        self.data = torch.concat(data, axis=0).unsqueeze(1)
        self.data = self.data.type(torch.float32)
        self.labels = curr_labels.repeat(self.N_recordings)
        self.labels += 1
        self.labels = self.labels.type(torch.int64)

        # Preprocessing: apply Differential Entropy on single channels
        if preprocess_de:
            print("-------------------------------------------------------")
            # Stack eeg windows one over the other
            self.data = self.data.reshape(-1, self.win_length)
            self.band_freqs = band_frequencies
            self.data = torch.stack([
                bandpower_diff_entropy(self.data, self.sampl_freq, band_freq)
                for band_freq in tqdm(self.band_freqs, desc="Computing DE on band:")
            ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        sample = self.data[index]
        label = self.labels[index]

        return sample, label
#--------------------------------------------------------------------------