import sys

common_dir = "/mnt/d/github/study_pytorch/common"
if common_dir not in sys.path:
    sys.path.append(common_dir)

from trainer import VAETrainer, set_seed, EarlyStopping
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
        self.fc1 = nn.Linear(128 * 4 * 4, latent_dim)
        self.fc2 = nn.Linear(128 * 4 * 4, latent_dim)

    def forward(self, x):
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = x.view(-1, 128 * 4 * 4)
        mu, logvar = self.fc1(x), self.fc2(x)
        return mu, logvar 


class DeconvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.deconv_block = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, 
                kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(),
        )

    def forward(self, x):
        x = self.deconv_block(x)
        return x


class Decoder(nn.Module):
    def __init__(self, latent_dim, output_channels=3):
        super().__init__()
        self.fc = nn.Linear(latent_dim, 128 * 4 * 4)
        self.deconv1 = DeconvBlock(128, 64)
        self.deconv2 = DeconvBlock(64, 32)
        self.deconv3 = nn.ConvTranspose2d(32, output_channels, 
                        kernel_size=3, stride=2, padding=1, output_padding=1)

    def forward(self, z): 
        x = self.fc(z)
        x = x.view(-1, 128, 4, 4)
        x = self.deconv1(x)
        x = self.deconv2(x)
        x = self.deconv3(x)
        return x

class VAE(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def sample_z(self, mu, logvar):
        std = torch.exp(0.5 * logvar)  
        eps = torch.randn_like(mu)
        return mu + std * eps

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.sample_z(mu, logvar)        
        x = self.decoder(z)
        x = torch.sigmoid(x)
        return x, mu, logvar

def binary_accuracy(x_pred, x_true):
    return torch.eq(x_pred.round(), x_true.round()).float().mean()  

def bce_kld_loss(x_pred, x, mu, logvar):
    bce = nn.BCELoss(reduction='sum')(x_pred, x)
    kdl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return bce + kdl


if __name__ == "__main__":

    ## Set seed
    set_seed(42)

    ## Set hyper-parameters
    batch_size = 64
    learning_rate = 1e-3
    n_epochs = 100
    latent_dim = 64

    ## Data loaders
    data_dir = "/mnt/d/datasets/cifar10_178M/cifar-10-batches-py/"
    train_loader, test_loader = get_loaders(data_dir, batch_size=batch_size,
                                            num_workers=8)

    ## Train model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder = Encoder(latent_dim)
    decoder = Decoder(latent_dim)
    model = VAE(encoder, decoder).to(device)

    loss_fn = bce_kld_loss
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    metrics = {"bce": nn.BCELoss(), 
           "mse": nn.MSELoss(), 
           "L1": nn.L1Loss(), 
           "acc": binary_accuracy}
    
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
    early_stopper = EarlyStopping(patience=3, min_delta=100)

    vae = VAETrainer(model, optimizer, loss_fn, metrics=metrics)
    vae.fit(train_loader, n_epochs, valid_loader=test_loader,
            stopper=early_stopper, scheduler=None)
    
    ## Evalueate model
    res = vae.evaluate(test_loader)

"""
Epoch[  1/100] loss: 120160.935 bce: 0.601 mse: 0.022 L1: 0.112 acc: 0.814 | val_loss: 117570.433 val_bce: 0.588 val_mse: 0.016 val_L1: 0.096 val_acc: 0.843
Epoch[  2/100] loss: 117674.858 bce: 0.587 mse: 0.016 L1: 0.095 acc: 0.845 | val_loss: 117065.031 val_bce: 0.585 val_mse: 0.015 val_L1: 0.091 val_acc: 0.850
Epoch[  3/100] loss: 117357.197 bce: 0.585 mse: 0.015 L1: 0.092 acc: 0.851 | val_loss: 116903.082 val_bce: 0.585 val_mse: 0.015 val_L1: 0.090 val_acc: 0.852
Epoch[  4/100] loss: 117151.624 bce: 0.583 mse: 0.014 L1: 0.090 acc: 0.854 | val_loss: 116793.546 val_bce: 0.583 val_mse: 0.014 val_L1: 0.089 val_acc: 0.855
Epoch[  5/100] loss: 117024.034 bce: 0.582 mse: 0.014 L1: 0.088 acc: 0.856 | val_loss: 116470.928 val_bce: 0.582 val_mse: 0.014 val_L1: 0.086 val_acc: 0.858
Epoch[  6/100] loss: 116941.505 bce: 0.582 mse: 0.014 L1: 0.088 acc: 0.857 | val_loss: 116524.899 val_bce: 0.582 val_mse: 0.014 val_L1: 0.086 val_acc: 0.858
Epoch[  7/100] loss: 116855.575 bce: 0.581 mse: 0.014 L1: 0.087 acc: 0.859 | val_loss: 116405.599 val_bce: 0.581 val_mse: 0.013 val_L1: 0.085 val_acc: 0.861
Epoch[  8/100] loss: 116811.395 bce: 0.581 mse: 0.014 L1: 0.086 acc: 0.859 | val_loss: 116363.894 val_bce: 0.581 val_mse: 0.013 val_L1: 0.085 val_acc: 0.860
Epoch[  9/100] loss: 116749.412 bce: 0.580 mse: 0.013 L1: 0.086 acc: 0.860 | val_loss: 116411.507 val_bce: 0.581 val_mse: 0.013 val_L1: 0.085 val_acc: 0.861
Epoch[ 10/100] loss: 116717.798 bce: 0.580 mse: 0.013 L1: 0.085 acc: 0.861 | val_loss: 116236.866 val_bce: 0.580 val_mse: 0.013 val_L1: 0.084 val_acc: 0.862
Epoch[ 11/100] loss: 116679.721 bce: 0.580 mse: 0.013 L1: 0.085 acc: 0.862 | val_loss: 116341.724 val_bce: 0.581 val_mse: 0.013 val_L1: 0.085 val_acc: 0.861
Epoch[ 12/100] loss: 116646.087 bce: 0.580 mse: 0.013 L1: 0.085 acc: 0.862 | val_loss: 116188.655 val_bce: 0.580 val_mse: 0.013 val_L1: 0.083 val_acc: 0.863
Epoch[ 13/100] loss: 116618.639 bce: 0.580 mse: 0.013 L1: 0.084 acc: 0.862 | val_loss: 116176.464 val_bce: 0.580 val_mse: 0.013 val_L1: 0.083 val_acc: 0.864
>> Early stopped! (best_loss: 116236.866)
>> test_loss: 116230.662 test_bce: 0.580 test_mse: 0.013 test_L1: 0.084 test_acc: 0.862  
"""