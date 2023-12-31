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
class CCNNModel(pl.LightningModule):
    r'''
    Continuous Convolutional Neural Network (CCNN). 

    - Paper: Yang Y, Wu Q, Fu Y, et al. Continuous convolutional neural network with 3D input for EEG-based emotion recognition[C]//International Conference on Neural Information Processing. Springer, Cham, 2018: 433-443.
    - URL: https://link.springer.com/chapter/10.1007/978-3-030-04239-4_39
    - Related Project: https://github.com/ynulonger/DE_CNN

    Args:
        in_channels (int): The feature dimension of each electrode. (default: :obj:`4`)
        grid_size (tuple): Spatial dimensions of grid-like EEG representation. (default: :obj:`(9, 9)`)
        num_classes (int): The number of classes to predict. (default: :obj:`2`)
        dropout (float): Probability of an element to be zeroed in the dropout layers. (default: :obj:`0.25`)
    '''
    def __init__(
            self, 
            input_ch: Optional[int] = 4, 
            grid_size: Optional[Tuple[int, int]] = (9, 9),
            num_classes: Optional[int] = 3,
            output_ch: Optional[int] = 64,   
            kernel_size: Optional[int] = 4, 
            hidden_size: Optional[int] = 1024,
            dropout_prob: Optional[float] = 0.5
    ):
        super().__init__()
        self.input_ch = input_ch
        self.output_ch = output_ch
        self.grid_size = grid_size
        self.num_classes = num_classes
        self.kernel_size = kernel_size
        self.hidden_size = hidden_size
        self.dropout = dropout_prob

        self.conv1 = nn.Sequential(
            # nn.ZeroPad2d((1, 2, 1, 2)), 
            nn.Conv2d(
                in_channels=self.input_ch, 
                out_channels=self.output_ch, 
                kernel_size=kernel_size, 
                stride=1,
                padding="same"
            ),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            # nn.ZeroPad2d((1, 2, 1, 2)), 
            nn.Conv2d(
                in_channels=self.output_ch, 
                out_channels=self.output_ch * 2, 
                kernel_size=kernel_size, 
                stride=1,
                padding="same"
            ),
            nn.ReLU()
        )
        self.conv3 = nn.Sequential(
            # nn.ZeroPad2d((1, 2, 1, 2)), 
            nn.Conv2d(
                in_channels=self.output_ch * 2, 
                out_channels=self.output_ch * 4, 
                kernel_size=kernel_size, 
                stride=1,
                padding="same"
            ),
            nn.ReLU()
        )
        self.conv4 = nn.Sequential(
            # nn.ZeroPad2d((1, 2, 1, 2)), 
            nn.Conv2d(
                in_channels=self.output_ch * 4, 
                out_channels=self.output_ch, 
                kernel_size=kernel_size, 
                stride=1,
                padding="same"
            ),
            nn.ReLU()
        )

        self.lin1 = nn.Sequential(
            nn.Linear(self.grid_size[0] * self.grid_size[1] * self.output_ch, self.hidden_size),
            nn.SELU(), # Not mentioned in paper
            nn.Dropout2d(self.dropout)
        )
        self.lin2 = nn.Linear(self.hidden_size, self.num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r'''
        Args:
            x (torch.Tensor): EEG signal representation, the ideal input shape is :obj:`[n, 4, 9, 9]`. Here, :obj:`n` corresponds to the batch size, :obj:`4` corresponds to :obj:`in_channels`, and :obj:`(9, 9)` corresponds to :obj:`grid_size`.

        Returns:
            torch.Tensor[number of sample, number of classes]: the predicted probability that the samples belong to the classes.
        '''
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)

        x = x.flatten(start_dim=1)

        x = self.lin1(x)
        x = self.lin2(x)
        return x
#-------------------------------------------------------------------------------------------------------

#-------------------------------------------------------------------------------------------------------
class CCNN(pl.LightningModule):
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
        self.model = CCNNModel(**model_parameters)
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