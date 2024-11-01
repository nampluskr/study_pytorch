import sys

common_dir = "/mnt/d/github/study_pytorch/common"
if common_dir not in sys.path:
    sys.path.append(common_dir)

from trainer import Trainer, set_seed, EarlyStopping
from cifar10 import get_loaders

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


## Model
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(),
            nn.MaxPool2d(2),
        )
    def forward(self, x):
        x = self.conv_block(x)
        return x


class Encoder(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.conv_block1 = ConvBlock(3, 32)
        self.conv_block2 = ConvBlock(32, 64)
        self.conv_block3 = ConvBlock(64, 128)
        self.fc = nn.Linear(128 * 4 * 4, latent_dim)

    def forward(self, x):
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = x.view(-1, 128 * 4 * 4)
        x = self.fc(x)
        return x


def accuracy(y_pred, y_true):
    y_pred = y_pred.argmax(dim=1)   # int64 (long)
    return torch.eq(y_pred, y_true).float().mean()


if __name__ == "__main__":

    ## Set seed
    set_seed(42)

    ## Set hyper-parameters
    batch_size = 64
    learning_rate = 1e-3
    n_epochs = 100

    ## Data loaders
    data_dir = "/mnt/d/datasets/cifar10_178M/cifar-10-batches-py/"
    train_loader, test_loader = get_loaders(data_dir, 
            batch_size=batch_size, num_workers=4)

    ## Train model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Encoder(latent_dim=10).to(device)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    metrics = {"acc": accuracy}
    
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
    early_stopper = EarlyStopping(patience=3, min_delta=0.001)

    clf = Trainer(model, optimizer, loss_fn, metrics=metrics)
    clf.fit(train_loader, n_epochs, valid_loader=test_loader,
            stopper=early_stopper, scheduler=None)
    
    ## Evalueate model
    res = clf.evaluate(test_loader)

"""
Epoch[  1/100] loss: 1.353 acc: 0.515 | val_loss: 1.414 val_acc: 0.526                                                  
Epoch[  2/100] loss: 1.037 acc: 0.633 | val_loss: 1.118 val_acc: 0.613                                                  
Epoch[  3/100] loss: 0.923 acc: 0.678 | val_loss: 1.146 val_acc: 0.622                                                  
Epoch[  4/100] loss: 0.849 acc: 0.704 | val_loss: 0.883 val_acc: 0.695                                                  
Epoch[  5/100] loss: 0.801 acc: 0.720 | val_loss: 0.778 val_acc: 0.732                                                  
Epoch[  6/100] loss: 0.750 acc: 0.739 | val_loss: 0.869 val_acc: 0.708                                                  
Epoch[  7/100] loss: 0.718 acc: 0.749 | val_loss: 0.819 val_acc: 0.721                                                  
Epoch[  8/100] loss: 0.687 acc: 0.761 | val_loss: 0.817 val_acc: 0.717                                                  
Epoch[  9/100] loss: 0.659 acc: 0.771 | val_loss: 0.801 val_acc: 0.732                                                  
Epoch[ 10/100] loss: 0.631 acc: 0.781 | val_loss: 0.675 val_acc: 0.768                                                  
Epoch[ 11/100] loss: 0.609 acc: 0.790 | val_loss: 0.754 val_acc: 0.749                                                  
Epoch[ 12/100] loss: 0.596 acc: 0.793 | val_loss: 0.685 val_acc: 0.772                                                  
Epoch[ 13/100] loss: 0.573 acc: 0.803 | val_loss: 0.744 val_acc: 0.754                                                  
Epoch[ 14/100] loss: 0.550 acc: 0.810 | val_loss: 0.745 val_acc: 0.743                                                  
Epoch[ 15/100] loss: 0.538 acc: 0.815 | val_loss: 0.810 val_acc: 0.729                                                  
>> Early stopped! (best_loss: 0.675)
>> test_loss: 0.675 test_acc: 0.768   
"""