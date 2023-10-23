from torch.utils.data import DataLoader
from dataset_v2 import SEEDDataset
from model import EEGFeedForwardNet
import pytorch_lightning as pl

window_length = 2000
batch_sz = 64
model_params = {
    "input_size": 248,
    "num_classes": 3,
    "dropout_prob": 0.5, 
    "hidden_size": 64,
    "norm_method": "batch"
}
training_params = {    
    "lr": 5e-4, 
    "betas": [0.9, 0.99], 
    "weight_decay": 1e-6, 
    "epochs": 20, 
    "lr_patience": 3
}

train_dataset = SEEDDataset(
    path_to_preprocessed=f"../data/processed_data_grid/train_data_processed_win{window_length}_grid.h5",
    split="train"
)
print("------------------------------------------------------------")
val_dataset = SEEDDataset(
    path_to_preprocessed=f"../data/processed_data_grid/valid_data_processed_win{window_length}_grid.h5",
    split="validation"
)
print("------------------------------------------------------------")
test_dataset = SEEDDataset(
    path_to_preprocessed=f"../data/processed_data_grid/test_data_processed_win{window_length}_grid.h5",
    split="test"
)
print("------------------------------------------------------------")

train_dataloader = DataLoader(train_dataset, batch_size=batch_sz, shuffle=True, num_workers=8)
val_dataloader = DataLoader(val_dataset, batch_size=batch_sz, shuffle=False, num_workers=8)
test_dataloader = DataLoader(test_dataset, batch_size=batch_sz, shuffle=False, num_workers=8)

eeg_net = EEGFeedForwardNet(
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