import torch
import torch.nn as nn
import pytorch_lightning as pl
from torcheval.metrics.functional import (
    multiclass_accuracy,
    multiclass_f1_score
)
from torch import optim
from typing import Tuple, Optional, Dict, Iterable, Literal

#----------------------------------------------------------------------------------------------------------------
class EEGFeedForwardNetModel(pl.LightningModule):
    def __init__(
            self,
            input_size: Optional[int] = 248,
            num_classes: Optional[int] = 3,
            hidden_size: Optional[int] = 64,
            norm_method: Optional[Literal["batch", "stratified"]] = "batch",
            dropout_prob: Optional[float] = 0.5, 
    ):
        super().__init__()

        self.input_size = input_size
        self.num_classes = num_classes
        self.hidden_size = hidden_size
        self.norm_method = norm_method
        self.dropout_prob = dropout_prob

        assert self.norm_method in ["batch", "stratified"], f"\
            The chosen normalization method {norm_method} is not available. Please choose one from ['batch', 'stratified']."
        if self.norm_method == "batch":
            self.norm_layer = nn.BatchNorm1d
        elif self.norm_method == "stratified":
            NotImplementedError
            # self.norm_layer = StratifiedNorm

        self.input_norm = self.norm_layer(self.input_size)

        self.fc1 = nn.Sequential(
            nn.Linear(in_features=self.input_size, out_features=self.hidden_size),
            self.norm_layer(num_features=self.hidden_size),
            nn.ReLU(True)
        )
        self.drop1 = nn.Dropout(self.dropout_prob)
        self.fc2 = nn.Sequential(
            nn.Linear(in_features=self.hidden_size, out_features=self.hidden_size),
            self.norm_layer(num_features=self.hidden_size),
            nn.ReLU(True)
        )
        self.drop2 = nn.Dropout(self.dropout_prob)
        self.fc3 = nn.Sequential(
            nn.Linear(in_features=self.hidden_size, out_features=self.hidden_size),
            self.norm_layer(num_features=self.hidden_size),
            nn.ReLU(True)
        )
        self.drop3 = nn.Dropout(self.dropout_prob)
        self.fc4 = nn.Sequential(
            nn.Linear(in_features=self.hidden_size, out_features=self.num_classes),
            nn.ReLU(True)
        )

    def forward(self, x):
        x = self.input_norm(x)
        x = self.fc1(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        x = self.fc3(x)
        x = self.drop3(x)
        x = self.fc4(x)
        return x
#-------------------------------------------------------------------------------------------------------    

#-------------------------------------------------------------------------------------------------------
class EEGFeedForwardNet(pl.LightningModule):
    def __init__(
            self, 
            model_parameters: Dict,
            lr: Optional[float] = 1e-4, 
            betas: Optional[Iterable[float]] = [0.9, 0.99], 
            weight_decay: Optional[float] = 1e-6, 
            epochs: Optional[int] = 100,
            lr_patience: Optional[int] = 20,  
        ):
        super().__init__()

        self.save_hyperparameters()
        self.loss_fun = nn.CrossEntropyLoss()
        self.model = EEGFeedForwardNetModel(**model_parameters)
        self.lr = lr
        self.betas = betas
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.lr_patience = lr_patience

    def forward(
            self, 
            input: Tuple[torch.Tensor, int]
        ):
        '''
        Parameters:
        -----------
            input: Tuple[torch.Tensor, int, int]
                Tensor of size (B, M), plus an integer label, plus integer for trial_id

        Returns:
        --------
            x: (torch.Tensor)
                Tensor of size (B, 3)
        '''

        x, y, z = input
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
            train_batch: Tuple[torch.Tensor, int, int]
                Tensor of size (B, M), plus an integer label, plus integer for trial_id
        '''

        # extract input (x signal, y label)
        x, y, z = train_batch 
        
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
            on_step=False, 
            on_epoch=True, 
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
            val_batch: Tuple[torch.Tensor, int, int]
                Tensor of size (B, M), plus an integer label, plus integer for trial_id
        '''

        # extract input (x signal, y label)
        x, y, z = val_batch 
        
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
            test_batch: Tuple[torch.Tensor, int, int]
                Tensor of size (B, M), plus an integer label, plus integer for trial_id
        '''

        # extract input (x signal, y label)
        x, y, z = test_batch 
        
        # network output
        out = self.model(x)
        
        return self.loss_fun(out, y)
    

    def configure_optimizers(self):
        optimizer = optim.Adam(
            self.parameters(), lr=self.lr, betas=self.betas, weight_decay=self.weight_decay
        )
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer=optimizer, 
            mode='min', 
            factor=0.1, 
            patience=self.lr_patience, 
            min_lr=1e-7
        )
        return [optimizer], [{"scheduler": scheduler,"monitor": "val_loss", "interval": "epoch", "frequency": 1}]
#----------------------------------------------------------------------------------------------------------------
