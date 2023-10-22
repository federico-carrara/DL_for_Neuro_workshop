from torch.utils.data import DataLoader
from dataset import TrainDataset, ValidationDataset, TestDataset
from model import CCNN
import pytorch_lightning as pl

num_train_recordings = 36
num_valid_recordings = 6
num_test_recordings = 3

window_length = 2000
window_overlap = 200
batch_sz = 64
model_params = {
    "input_ch": 4, 
    "grid_size": (9, 9),
    "num_classes": 3,
    "output_ch":  64,   
    "kernel_size":  4, 
    "hidden_size":  1024,
    "dropout_prob": 0.5
}
training_params = {    
    "lr": 5e-4, 
    "betas": [0.9, 0.99], 
    "weight_decay": 1e-6, 
    "epochs": 20, 
    "lr_patience": 3
}

train_dataset = TrainDataset(
    path_to_data_dir=None, 
    path_to_labels=None,
    num_recordings=num_train_recordings,
    path_to_preprocessed=f"../data/processed_data_grid/train_data_processed_win{window_length}_grid.h5"
)
print("------------------------------------------------------------")
val_dataset = ValidationDataset(
    path_to_data_dir=None, 
    path_to_labels=None,
    num_recordings=num_valid_recordings,
    path_to_preprocessed=f"../data/processed_data_grid/valid_data_processed_win{window_length}_grid.h5"
)
print("------------------------------------------------------------")
test_dataset = TestDataset(
    path_to_data_dir=None, 
    path_to_labels=None,
    num_recordings=num_valid_recordings,
    path_to_preprocessed=f"../data/processed_data_grid/test_data_processed_win{window_length}_grid.h5"
)
print("------------------------------------------------------------")

train_dataloader = DataLoader(train_dataset, batch_size=batch_sz, shuffle=True, num_workers=8)
val_dataloader = DataLoader(val_dataset, batch_size=batch_sz, shuffle=False, num_workers=8)
test_dataloader = DataLoader(test_dataset, batch_size=batch_sz, shuffle=False, num_workers=8)

eeg_net = CCNN(
    model_parameters=model_params,
    **training_params
)

best_checkpoint_callback = pl.callbacks.ModelCheckpoint(
    save_top_k=1,
    monitor="val_loss",
    mode="min",
    filename="best_checkpoint",
)

last_checkpoint_callback = pl.callbacks.ModelCheckpoint(
    save_last=True,
    filename="last_checkpoint_at_{epoch:02d}",
)

lr_monitor = pl.callbacks.LearningRateMonitor(logging_interval='epoch')

trainer = pl.Trainer(
    accelerator="gpu",
    max_epochs=training_params["epochs"], 
    callbacks=[best_checkpoint_callback, last_checkpoint_callback, lr_monitor])

trainer.fit(eeg_net, train_dataloader, val_dataloader)