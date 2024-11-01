import sys

common_dir = "/mnt/d/github/study_pytorch/common"
if common_dir not in sys.path:
    sys.path.append(common_dir)

from trainer import AETrainer, set_seed, EarlyStopping, binary_accuracy
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


class AutoEncoder(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        x = torch.sigmoid(x)
        return x


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
    model = AutoEncoder(encoder, decoder).to(device)

    loss_fn = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    metrics = {"L1": nn.L1Loss(), "acc": binary_accuracy}
    
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
    early_stopper = EarlyStopping(patience=5, min_delta=0.001)

    ae = AETrainer(model, optimizer, loss_fn, metrics=metrics)
    ae.fit(train_loader, n_epochs, valid_loader=test_loader,
            stopper=early_stopper, scheduler=None)
    
    ## Evalueate model
    res = ae.evaluate(test_loader)

"""
Epoch[  1/100] loss: 0.592 L1: 0.099 | val_loss: 0.577 val_L1: 0.078                                                    
Epoch[  2/100] loss: 0.574 L1: 0.076 | val_loss: 0.572 val_L1: 0.071                                                    
Epoch[  3/100] loss: 0.571 L1: 0.071 | val_loss: 0.571 val_L1: 0.068                                                    
Epoch[  4/100] loss: 0.570 L1: 0.069 | val_loss: 0.570 val_L1: 0.068                                                    
Epoch[  5/100] loss: 0.570 L1: 0.068 | val_loss: 0.569 val_L1: 0.065                                                    
Epoch[  6/100] loss: 0.569 L1: 0.067 | val_loss: 0.570 val_L1: 0.066                                                    
Epoch[  7/100] loss: 0.569 L1: 0.067 | val_loss: 0.569 val_L1: 0.065                                                    
Epoch[  8/100] loss: 0.569 L1: 0.066 | val_loss: 0.569 val_L1: 0.065                                                    
Epoch[  9/100] loss: 0.569 L1: 0.066 | val_loss: 0.569 val_L1: 0.065                                                    
Epoch[ 10/100] loss: 0.568 L1: 0.066 | val_loss: 0.568 val_L1: 0.064                                                    
>> Early stopped! (best_loss: 0.569)
>> test_loss: 0.569 test_L1: 0.065   
"""
