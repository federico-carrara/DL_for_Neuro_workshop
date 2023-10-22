import os
import string
import torch
import h5py
import warnings
from scipy.io import loadmat
from random import random
from tqdm import tqdm
from torch.utils.data import Dataset
from typing import Optional, List, Tuple, Literal
from utils import (
    load_eeg_data_file, 
    windowize_signal, 
    add_gaussian_noise,
    bandpower_diff_entropy,
    ToGrid, 
)

SEED_LOCATION_LIST = [
    ['-', '-', '-', 'FP1', 'FPZ', 'FP2', '-', '-', '-'],
    ['-', '-', '-', 'AF3', '-', 'AF4', '-', '-', '-'],
    ['F7', 'F5', 'F3', 'F1', 'FZ', 'F2', 'F4', 'F6', 'F8'],
    ['FT7', 'FC5', 'FC3', 'FC1', 'FCZ', 'FC2', 'FC4', 'FC6', 'FT8'],
    ['T7', 'C5', 'C3', 'C1', 'CZ', 'C2', 'C4', 'C6', 'T8'],
    ['TP7', 'CP5', 'CP3', 'CP1', 'CPZ', 'CP2', 'CP4', 'CP6', 'TP8'],
    ['P7', 'P5', 'P3', 'P1', 'PZ', 'P2', 'P4', 'P6', 'P8'],
    ['-', 'PO7', 'PO5', 'PO3', 'POZ', 'PO4', 'PO6', 'PO8', '-'],
    ['-', '-', 'CB1', 'O1', 'OZ', 'O2', 'CB2', '-', '-']
]

SEED_CHANNEL_LIST = [
    'FP1', 'FPZ', 'FP2', 'AF3', 'AF4', 'F7', 'F5', 'F3', 'F1', 'FZ', 'F2', 'F4',
    'F6', 'F8', 'FT7', 'FC5', 'FC3', 'FC1', 'FCZ', 'FC2', 'FC4', 'FC6', 'FT8',
    'T7', 'C5', 'C3', 'C1', 'CZ', 'C2', 'C4', 'C6', 'T8', 'TP7', 'CP5', 'CP3',
    'CP1', 'CPZ', 'CP2', 'CP4', 'CP6', 'TP8', 'P7', 'P5', 'P3', 'P1', 'PZ',
    'P2', 'P4', 'P6', 'P8', 'PO7', 'PO5', 'PO3', 'POZ', 'PO4', 'PO6', 'PO8',
    'CB1', 'O1', 'OZ', 'O2', 'CB2'
]

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
        preprocess_de: Optional[bool] = False,
        band_frequencies: Optional[List[Tuple[int, int]]] = [
            (4, 8), (8, 14), (14, 31), (31, 49)
        ],
        normalization: Optional[Literal["all", "trial"]] = "trial",
        to_grid: Optional[bool] = False,
    ):
        
        if not preprocess_de and to_grid:
            warnings.warn("You set `preprocess_de=False`, so cannot map eeg data to grid.")
            
        # Global parameters
        if num_recordings:
            self.N_recordings = num_recordings
        else:
            self.N_recordings = len(os.listdir(path_to_data_dir))
        self.N_movies = num_movies
        self.N_channels = num_channels
        self.sampl_freq = sampling_frequency
        self.trials_ids_unique = torch.arange(1, self.N_recordings + 1, step=1)
        # Windows parameters
        self.win_length = win_length
        self.win_overlap = win_overlap
        # Differential entropy parameters
        self.preprocess_de = preprocess_de
        self.band_frequencies = band_frequencies
        self.N_bands = len(self.band_frequencies)
        # Other parameters
        self.normalization = normalization
        self.to_grid = to_grid

        # Load data and labels, store them in torch tensors
        eeg_files = [fname for fname in os.listdir(path_to_data_dir) if fname[0] in string.digits]
        labels_dict = loadmat(path_to_labels)
        labels_unique = labels_dict["label"].squeeze(0)
        data = []
        labels = []
        trial_ids = []
        trial_id = 0
        for eeg_file in tqdm(eeg_files, desc="Loading training data files"):
            #Data
            curr_data = load_eeg_data_file(
                path_to_file=os.path.join(path_to_data_dir, eeg_file),
                num_keys=self.N_movies,
            )
            curr_windowized_data, num_windows = windowize_signal(curr_data, self.win_length, self.win_overlap)
            data.extend(curr_windowized_data)
            #Labels    
            curr_labels = [
                torch.tensor([labels_unique[i]] * num_windows[i])
                for i in range(len(curr_data))
            ]
            curr_labels = torch.concat(curr_labels, axis=0)
            labels.append(curr_labels)
            #Subject/trial id
            trial_id += 1
            trial_ids.append(torch.tensor([trial_id] * len(curr_labels)))
        
        data = [tens.unsqueeze(0) for tens in data]
        self.data = torch.concat(data, axis=0).unsqueeze(1)
        self.labels = torch.concat(labels)
        self.labels += 1
        self.trial_ids = torch.concat(trial_ids)

        # Set proper data type
        self.data = self.data.type(torch.float32)
        self.labels = self.labels.type(torch.int64)
        self.trial_ids = self.trial_ids.type(torch.int64)
        self.N_samples = self.data.shape[0]

        assert len(self.labels) == len(self.data), "Data and labels are not paired..."
        
        # Boolean: if true apply transformation for data augmentation
        self.data_augmentation = data_augmentation

        # Preprocessing: apply Differential Entropy on single channels
        if self.preprocess_de:
            
            # Exctract DE channel-wise
            self.data = self.compute_Diff_Entropy()
            
            # Normalize data according the method given above
            self.data = self.normalize_data()

            # If required, map data to a 2D grid made by electrodes positions
            if self.to_grid:
                self.data = self.map_to_grid()
            else:
                self.data = self.data.reshape(self.N_samples, -1)

            assert len(self.labels) == len(self.data), "Data and labels are not paired..."
    
    
    def map_to_grid(self):
        assert self.data.shape == (self.N_samples, self.N_bands, self.N_channels),  f"\
The shape of the input data {tuple(self.data.shape)} differs from the one expected {(self.N_samples, self.N_bands, self.N_channels)}"
        
        mapper = ToGrid(SEED_CHANNEL_LIST, SEED_LOCATION_LIST)
        self.data = mapper.apply(self.data)

        return self.data
    
    
    def normalize_data(self):
        assert self.data.shape == (self.N_samples, self.N_bands, self.N_channels),  f"\
The shape of the input data {tuple(self.data.shape)} differs from the one expected {(self.N_samples, self.N_bands, self.N_channels)}"
        
        if self.normalization == "all": # normalize across all instances
                mean_tensor = torch.mean(self.data, dim=0)
                std_tensor = torch.std(self.data, dim=0)
                self.data = (self.data - mean_tensor) / std_tensor
        elif self.normalization == "trial": # normalize data for each trial
            for trial_id in self.trials_ids_unique:
                mask = torch.argwhere(self.trial_ids == trial_id) 
                mean_tensor = torch.mean(self.data[mask, ...], dim=0)
                std_tensor = torch.std(self.data[mask, ...], dim=0)
                self.data[mask, ...] = (self.data[mask, ...] - mean_tensor) / std_tensor

        return self.data
    
    
    def compute_Diff_Entropy(self):
        assert self.data.shape == (self.N_samples, 1, self.N_channels, self.win_length),  f"\
The shape of the input data {tuple(self.data.shape)} differs from the one expected {(self.N_samples, 1, self.N_channels, self.win_length)}"
        
        # Legend: N=num_samples(split in windows), C=num_channels, W=window_length, F+num_band_frequencies
        # Stack eeg windows -- > input: (N, C, W)
        self.data = self.data.reshape(-1, self.win_length) # (N * C, W) 
        self.data = torch.stack([
            bandpower_diff_entropy(self.data, self.sampl_freq, band_freq)
            for band_freq in tqdm(self.band_frequencies, desc="Computing DE on band")
        ]) # (N * C, F)
        self.data = self.data.reshape(-1, self.N_samples, self.N_channels) # (F, N, C)
        self.data = self.data.swapaxes(0, 1) # (N, F, C)

        return self.data 
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        sample = self.data[index]
        label = self.labels[index]
        trial_id = self.trial_ids[index]

        if self.data_augmentation:
            if random() > 0.2:
                sample = add_gaussian_noise(sample)

        return sample, label, trial_id
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
        band_frequencies: Optional[List[Tuple[int, int]]] = [
            (4, 8), (8, 14), (14, 31), (31, 49)
        ],
        normalization: Optional[Literal["all", "trial"]] = "trial",
        to_grid: Optional[bool] = False,
    ):
        
        if not preprocess_de and to_grid:
            warnings.warn("You set `preprocess_de=False`, so cannot map eeg data to grid.")

        # Global parameters
        if num_recordings:
            self.N_recordings = num_recordings
        else:
            self.N_recordings = len(os.listdir(path_to_data_dir))
        self.N_movies = num_movies
        self.N_channels = num_channels
        self.sampl_freq = sampling_frequency
        self.trials_ids_unique = torch.arange(1, self.N_recordings + 1, step=1)
        # Windows parameters
        self.win_length = win_length
        self.win_overlap = win_overlap
        # Differential entropy parameters
        self.preprocess_de = preprocess_de
        self.band_frequencies = band_frequencies
        self.N_bands = len(self.band_frequencies)
        # Other parameters
        self.normalization = normalization
        self.to_grid = to_grid

        # Load data and labels, store them in torch tensors
        eeg_files = [fname for fname in os.listdir(path_to_data_dir) if fname[0] in string.digits]
        labels_dict = loadmat(path_to_labels)
        labels_unique = labels_dict["label"].squeeze(0)
        data = []
        labels = []
        trial_ids = []
        trial_id = 0
        for eeg_file in tqdm(eeg_files, desc="Loading validation data files"):
            #Data
            curr_data = load_eeg_data_file(
                path_to_file=os.path.join(path_to_data_dir, eeg_file),
                num_keys=self.N_movies,
            )
            curr_windowized_data, num_windows = windowize_signal(curr_data, self.win_length, self.win_overlap)
            data.extend(curr_windowized_data)
            #Labels    
            curr_labels = [
                torch.tensor([labels_unique[i]] * num_windows[i])
                for i in range(len(curr_data))
            ]
            curr_labels = torch.concat(curr_labels, axis=0)
            labels.append(curr_labels)
            #Subject/trial id
            trial_id += 1
            trial_ids.append(torch.tensor([trial_id] * len(curr_labels)))

        data = [tens.unsqueeze(0) for tens in data]
        self.data = torch.concat(data, axis=0).unsqueeze(1)
        self.labels = torch.concat(labels)
        self.labels += 1
        self.trial_ids = torch.concat(trial_ids)

        # Set proper data type
        self.data = self.data.type(torch.float32)
        self.labels = self.labels.type(torch.int64)
        self.trial_ids = self.trial_ids.type(torch.int64)
        self.N_samples = self.data.shape[0]

        assert len(self.labels) == len(self.data), "Data and labels are not paired..."

        # Preprocessing: apply Differential Entropy on single channels
        if self.preprocess_de:
            
            # Exctract DE channel-wise
            self.data = self.compute_Diff_Entropy()
            
            # Normalize data according the method given above
            self.data = self.normalize_data()

            # If required, map data to a 2D grid made by electrodes positions
            if self.to_grid:
                self.data = self.map_to_grid()
            else:
                self.data = self.data.reshape(self.N_samples, -1)

            assert len(self.labels) == len(self.data), "Data and labels are not paired..."
    
    def map_to_grid(self):
        assert self.data.shape == (self.N_samples, self.N_bands, self.N_channels),  f"\
The shape of the input data {tuple(self.data.shape)} differs from the one expected {(self.N_samples, self.N_bands, self.N_channels)}"
        
        mapper = ToGrid(SEED_CHANNEL_LIST, SEED_LOCATION_LIST)
        self.data = mapper.apply(self.data)

        return self.data
    
    
    def normalize_data(self):
        assert self.data.shape == (self.N_samples, self.N_bands, self.N_channels),  f"\
The shape of the input data {tuple(self.data.shape)} differs from the one expected {(self.N_samples, self.N_bands, self.N_channels)}"
        
        if self.normalization == "all": # normalize across all instances
                mean_tensor = torch.mean(self.data, dim=0)
                std_tensor = torch.std(self.data, dim=0)
                self.data = (self.data - mean_tensor) / std_tensor
        elif self.normalization == "trial": # normalize data for each trial
            for trial_id in self.trials_ids_unique:
                mask = torch.argwhere(self.trial_ids == trial_id) 
                mean_tensor = torch.mean(self.data[mask, ...], dim=0)
                std_tensor = torch.std(self.data[mask, ...], dim=0)
                self.data[mask, ...] = (self.data[mask, ...] - mean_tensor) / std_tensor

        return self.data
    
    
    def compute_Diff_Entropy(self):
        assert self.data.shape == (self.N_samples, 1, self.N_channels, self.win_length),  f"\
The shape of the input data {tuple(self.data.shape)} differs from the one expected {(self.N_samples, 1, self.N_channels, self.win_length)}"
        
        # Legend: N=num_samples(split in windows), C=num_channels, W=window_length, F+num_band_frequencies
        # Stack eeg windows -- > input: (N, C, W)
        self.data = self.data.reshape(-1, self.win_length) # (N * C, W) 
        self.data = torch.stack([
            bandpower_diff_entropy(self.data, self.sampl_freq, band_freq)
            for band_freq in tqdm(self.band_frequencies, desc="Computing DE on band")
        ]) # (N * C, F)
        self.data = self.data.reshape(-1, self.N_samples, self.N_channels) # (F, N, C)
        self.data = self.data.swapaxes(0, 1) # (N, F, C)

        return self.data 
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        sample = self.data[index]
        label = self.labels[index]
        trial_id = self.trial_ids[index]

        return sample, label, trial_id
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
        band_frequencies: Optional[List[Tuple[int, int]]] = [
            (4, 8), (8, 14), (14, 31), (31, 49)
        ],
        normalization: Optional[Literal["all", "trial"]] = "trial",
        to_grid: Optional[bool] = False,
    ):

        if not preprocess_de and to_grid:
            warnings.warn("You set `preprocess_de=False`, so cannot map eeg data to grid.")

        # Global parameters
        if num_recordings:
            self.N_recordings = num_recordings
        else:
            self.N_recordings = len(os.listdir(path_to_data_dir))
        self.N_movies = num_movies
        self.N_channels = num_channels
        self.sampl_freq = sampling_frequency
        self.trials_ids_unique = torch.arange(1, self.N_recordings + 1, step=1)
        # Windows parameters
        self.win_length = win_length
        self.win_overlap = win_overlap
        # Differential entropy parameters
        self.preprocess_de = preprocess_de
        self.band_frequencies = band_frequencies
        self.N_bands = len(self.band_frequencies)
        # Other parameters
        self.normalization = normalization
        self.to_grid = to_grid

        # Load data and labels, store them in torch tensors
        eeg_files = [fname for fname in os.listdir(path_to_data_dir) if fname[0] in string.digits]
        labels_dict = loadmat(path_to_labels)
        labels_unique = labels_dict["label"].squeeze(0)
        data = []
        labels = []
        trial_ids = []
        trial_id = 0
        for eeg_file in tqdm(eeg_files, desc="Loading test data files"):
            #Data
            curr_data = load_eeg_data_file(
                path_to_file=os.path.join(path_to_data_dir, eeg_file),
                num_keys=self.N_movies,
            )
            curr_windowized_data, num_windows = windowize_signal(curr_data, self.win_length, self.win_overlap)
            data.extend(curr_windowized_data)
            #Labels    
            curr_labels = [
                torch.tensor([labels_unique[i]] * num_windows[i])
                for i in range(len(curr_data))
            ]
            curr_labels = torch.concat(curr_labels, axis=0)
            labels.append(curr_labels)
            #Subject/trial id
            trial_id += 1
            trial_ids.append(torch.tensor([trial_id] * len(curr_labels)))

        data = [tens.unsqueeze(0) for tens in data]
        self.data = torch.concat(data, axis=0).unsqueeze(1)
        self.labels = torch.concat(labels)
        self.labels += 1
        self.trial_ids = torch.concat(trial_ids)

        # Set proper data type
        self.data = self.data.type(torch.float32)
        self.labels = self.labels.type(torch.int64)
        self.trial_ids = self.trial_ids.type(torch.int64)
        self.N_samples = self.data.shape[0]

        assert len(self.labels) == len(self.data), "Data and labels are not paired..."

        # Preprocessing: apply Differential Entropy on single channels
        if self.preprocess_de:
            
            # Exctract DE channel-wise
            self.data = self.compute_Diff_Entropy()
            
            # Normalize data according the method given above
            self.data = self.normalize_data()

            # If required, map data to a 2D grid made by electrodes positions
            if self.to_grid:
                self.data = self.map_to_grid()
            else:
                self.data = self.data.reshape(self.N_samples, -1)

            assert len(self.labels) == len(self.data), "Data and labels are not paired..."
    
    def map_to_grid(self):
        assert self.data.shape == (self.N_samples, self.N_bands, self.N_channels),  f"\
The shape of the input data {tuple(self.data.shape)} differs from the one expected {(self.N_samples, self.N_bands, self.N_channels)}"
        
        mapper = ToGrid(SEED_CHANNEL_LIST, SEED_LOCATION_LIST)
        self.data = mapper.apply(self.data)

        return self.data
    
    
    def normalize_data(self):
        assert self.data.shape == (self.N_samples, self.N_bands, self.N_channels),  f"\
The shape of the input data {tuple(self.data.shape)} differs from the one expected {(self.N_samples, self.N_bands, self.N_channels)}"
        
        if self.normalization == "all": # normalize across all instances
                mean_tensor = torch.mean(self.data, dim=0)
                std_tensor = torch.std(self.data, dim=0)
                self.data = (self.data - mean_tensor) / std_tensor
        elif self.normalization == "trial": # normalize data for each trial
            for trial_id in self.trials_ids_unique:
                mask = torch.argwhere(self.trial_ids == trial_id) 
                mean_tensor = torch.mean(self.data[mask, ...], dim=0)
                std_tensor = torch.std(self.data[mask, ...], dim=0)
                self.data[mask, ...] = (self.data[mask, ...] - mean_tensor) / std_tensor

        return self.data
    
    
    def compute_Diff_Entropy(self):
        assert self.data.shape == (self.N_samples, 1, self.N_channels, self.win_length),  f"\
The shape of the input data {tuple(self.data.shape)} differs from the one expected {(self.N_samples, 1, self.N_channels, self.win_length)}"
        
        # Legend: N=num_samples(split in windows), C=num_channels, W=window_length, F+num_band_frequencies
        # Stack eeg windows -- > input: (N, C, W)
        self.data = self.data.reshape(-1, self.win_length) # (N * C, W) 
        self.data = torch.stack([
            bandpower_diff_entropy(self.data, self.sampl_freq, band_freq)
            for band_freq in tqdm(self.band_frequencies, desc="Computing DE on band")
        ]) # (N * C, F)
        self.data = self.data.reshape(-1, self.N_samples, self.N_channels) # (F, N, C)
        self.data = self.data.swapaxes(0, 1) # (N, F, C)

        return self.data 
    

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        sample = self.data[index]
        label = self.labels[index]
        trial_id = self.trial_ids[index]

        return sample, label, trial_id
#--------------------------------------------------------------------------