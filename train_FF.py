from torch.utils.data import DataLoader
from dataset import TrainDataset, ValidationDataset, TestDataset
from model import EEGFeedForwardNet
import pytorch_lightning as pl

window_length = 1000
window_overlap = 100
batch_sz = 16
model_params = {
    "input_size": 248,
    "num_classes": 3,
    "dropout_prob": 0.3, 
    "hidden_size": 64,
    "norm_method": "batch"
}
training_params = {    
    "lr": 5e-4, 
    "betas": [0.9, 0.99], 
    "weight_decay": 1e-6, 
    "epochs": 100, 
}


train_dataset = TrainDataset(
    path_to_data_dir="../data/train/", 
    path_to_labels="../data/label.mat", 
    win_length=window_length, 
    win_overlap=window_overlap, 
    data_augmentation=False,
    preprocess_de=True
)
print("------------------------------------------------------------")
val_dataset = ValidationDataset(
    path_to_data_dir="../data/valid/", 
    path_to_labels="../data/label.mat", 
    win_length=window_length, 
    win_overlap=window_overlap,
    preprocess_de=True 
)
print("------------------------------------------------------------")
test_dataset = TestDataset(
    path_to_data_dir="../data/test/", 
    path_to_labels="../data/label.mat", 
    win_length=window_length, 
    win_overlap=window_overlap,
    preprocess_de=True
)
print("------------------------------------------------------------")
train_dataloader = DataLoader(train_dataset, batch_size=batch_sz, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=batch_sz, shuffle=False)
test_dataloader = DataLoader(test_dataset, batch_size=batch_sz, shuffle=False)

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