import torch
import torch.nn as nn
try:
    import pytorch_lightning as pl
except:
    import lightning as pl
from torcheval.metrics.functional import (
    multiclass_accuracy,
    multiclass_f1_score
)
from torch import optim
from typing import Tuple, Optional, Dict, Iterable, Literal

#--------------------------------------------------------------------------
class EEGNetModel(pl.LightningModule):
    def __init__(
            self, 
            input_size: Tuple[int, int],
            num_classes: Optional[int] = 3,
            num_out_channels: Optional[int] = 16,
            temporal_kernel_size: Optional[int] = 64,
            spatial_kernel_size: Optional[int] = 7, 
            separable_kernel_size: Optional[int] = 16,
            pooling_size: Optional[Tuple[int, int]] = (2, 5), 
            dropout_prob: Optional[float] = 0.5, 
            hidden_size: Optional[int] = 128,
    ):
        super().__init__()
        
        self.temporal_conv = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=num_out_channels,
                kernel_size=[1, temporal_kernel_size],
                stride=1, 
                bias=True,
                padding="same"
            ), 
            nn.BatchNorm2d(num_features=num_out_channels),
        )
        self.spatial_conv = nn.Sequential(
            nn.Conv2d(
                in_channels=num_out_channels,
                out_channels=num_out_channels * 2,
                kernel_size=[spatial_kernel_size, 1],
                stride=1,
                bias=True,
                padding="valid"
            ),
            nn.BatchNorm2d(num_features=num_out_channels * 2),
            nn.ReLU(True),
        )

        self.avg_pool = nn.AvgPool2d(
            kernel_size=pooling_size, 
            stride=pooling_size, 
            padding=0
        )

        self.seperable_conv = nn.Sequential(
            nn.Conv2d(
                in_channels=num_out_channels * 2,
                out_channels=num_out_channels * 2,
                kernel_size=[1, separable_kernel_size],
                padding="same",
                bias=True
            ),
            nn.Conv2d(
                in_channels=num_out_channels * 2,
                out_channels=num_out_channels * 2,
                kernel_size=[1, 1], 
                padding="same",
                bias=True
            ),
            nn.BatchNorm2d(num_features=num_out_channels * 2),
            nn.ReLU(True),
        )

        self.dropout = nn.Dropout(dropout_prob)
        self.flatten = nn.Flatten()
        flatten_size = int(
            num_out_channels * 2 * (input_size[1] - spatial_kernel_size  + 1) *\
            input_size[2] / pooling_size[0]**2 / pooling_size[1]**2 
        )
        self.fc1 = nn.Sequential(
            nn.Linear(in_features=flatten_size, out_features=hidden_size),
            nn.BatchNorm1d(num_features=hidden_size),
            nn.ReLU(True)
        )
        self.fc2 = nn.Linear(in_features=hidden_size, out_features=num_classes)   
    
    def forward(self, x):
        x = self.temporal_conv(x)
        x = self.spatial_conv(x)
        x = self.avg_pool(x)
        x = self.dropout(x)
        x = self.seperable_conv(x)
        x = self.avg_pool(x)
        x = self.dropout(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class EEGNet(pl.LightningModule):
    def __init__(
            self, 
            model_parameters: Dict,
            lr: Optional[float] = 1e-4, 
            betas: Optional[Iterable[float]] = [0.9, 0.99], 
            weight_decay: Optional[float] = 1e-6, 
            epochs: Optional[int] = 100, 
        ):
        '''
        '''
        super().__init__()

        self.save_hyperparameters()
        self.loss_fun = nn.CrossEntropyLoss()
        self.model = EEGNetModel(**model_parameters)
        self.lr = lr
        self.betas = betas
        self.weight_decay = weight_decay
        self.epochs = epochs

    def forward(
            self, 
            input: Tuple[torch.Tensor, int]
        ):
        '''
        Parameters:
        -----------
            input: Tuple[torch.Tensor, int]
                Tensor of size (B, 1, M, W), plus an integer label.

        Returns:
        --------
            x: (torch.Tensor)
                Tensor of size (B, 3, M)
        '''

        # extract input (x point cloud, y label)
        x, y = input
        x = self.model(x)
        return x
    

    def training_step(
            self, 
            train_batch: Tuple[torch.Tensor, torch.Tensor], 
            batch_idx: int
        ):
        '''
        Parameters:
        -----------
            train_batch: (Tuple[torch.Tensor, torch.Tensor])
                Tensor of size (B, 1, M, W) (the EEG signal) and tensor of size (B) (the labels).
        '''

        # extract input (x signal, y label)
        x, y = train_batch 
        
        # network output
        out = self.model(x)
        
        # compute loss & log it
        loss = self.loss_fun(out, y)
        
        # compute metrics & log them
        accuracy = multiclass_accuracy(input=out, target=y)
        f1_score = multiclass_f1_score(input=out, target=y, num_classes=out.shape[-1]) 

        # log loss and metrics
        self.log_dict(
            {'train_loss': loss, "train_accuracy": accuracy, "train_f1_score": f1_score},
            on_step=True, 
            on_epoch=False, 
            prog_bar=True,
            logger=True
        )
        
        return loss
    

    def validation_step(
            self, 
            val_batch: Tuple[torch.Tensor, torch.Tensor], 
            batch_idx: int
        ):
        '''
        Parameters:
        -----------
            val_batch: (Tuple[torch.Tensor, torch.Tensor])
                Tensor of size (B, 1, M, W) (the EEG signal) and tensor of size (B) (the labels).
        '''

        # extract input (x signal, y label)
        x, y = val_batch 
        
        # network output
        out = self.model(x)
        
        # compute loss
        loss = self.loss_fun(out, y)

        # compute metrics & log them
        accuracy = multiclass_accuracy(input=out, target=y)
        f1_score = multiclass_f1_score(input=out, target=y, num_classes=out.shape[-1]) 

        # log loss and metrics
        self.log_dict(
            {"val_loss": loss, "val_accuracy": accuracy, "val_f1_score": f1_score},
            on_step=False, 
            on_epoch=True, 
            prog_bar=True,
            logger=True
        )

        return loss
    
    def test_step(
            self, 
            test_batch: Tuple[torch.Tensor, torch.Tensor], 
            batch_idx: int
        ):
        '''
        Parameters:
        -----------
            test_batch: (Tuple[torch.Tensor, torch.Tensor])
                Tensor of size (B, 1, M, W) (the EEG signal) and tensor of size (B) (the labels).
        '''

        # extract input (x signal, y label)
        x, y = test_batch 
        
        # network output
        out = self.model(x)
        
        return self.loss_fun(out, y)
    

    def configure_optimizers(self):
        optimizer = optim.Adam(
            self.parameters(), lr=self.lr, betas=self.betas, weight_decay=self.weight_decay
        )
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer=optimizer, mode='min', factor=0.5, patience=15, min_lr=1e-7
        )
        return [optimizer], [{"scheduler": scheduler,"monitor": "val_loss", "interval": "epoch", "frequency": 1}]
#----------------------------------------------------------------------------------------------------------------

#----------------------------------------------------------------------------------------------------------------
class EEGFFNet(pl.LightningModule):
    def __init__(
            self,
            input_size: Optional[int] = 248,
            num_classes: Optional[int] = 3,
            hidden_size: Optional[int] = 128,
            norm_method: Optional[Literal["batch", "stratified"]] = "batch",
            dropout_prob: Optional[float] = 0.5, 
            

    ):
        super().__init__()

        assert norm_method in ["batch", "stratified"], f"\
            The chosen normalization method {norm_method} is not available. Please choose one from ['batch', 'stratified']."
        if norm_method == "batch":
            self.norm_layer = nn.BatchNorm1d
        elif norm_method == "stratified":
            NotImplementedError
            # self.norm_layer = StratifiedNorm

        self.input_norm = self.norm_layer(input_size)

        self.fc1 = nn.Sequential(
            nn.Linear(in_features=input_size, out_features=hidden_size),
            self.norm_layer(num_features=hidden_size),
            nn.ReLU(True)
        )
        self.drop1 = nn.Dropout(0.25)
        self.fc2 = nn.Sequential(
            nn.Linear(in_features=hidden_size, out_features=hidden_size // 2),
            self.norm_layer(num_features=hidden_size),
            nn.ReLU(True)
        )
        self.drop2 = nn.Dropout(0.25)
        self.fc3 = nn.Sequential(
            nn.Linear(in_features=input_size // 2, out_features=hidden_size // 2),
            self.norm_layer(num_features=hidden_size),
            nn.ReLU(True)
        )
        self.drop3 = nn.Dropout(0.25)
        self.fc4 = nn.Sequential(
            nn.Linear(in_features=hidden_size // 2, out_features=num_classes),
            self.norm_layer(num_features=hidden_size),
            nn.ReLU(True)
        )

    def forward(self, x, i):
        x = self.input_norm(x)
        x = self.fc1(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        x = self.fc3(x)
        x = self.drop3(x)
        x = self.fc4(x)
        return x
