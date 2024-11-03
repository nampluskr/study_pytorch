import sys
sys.path.insert(0, '..') ## '../..' for parent-parent directory

import torch
import torch.nn as nn

import os
import argparse
import numpy as np
import random

from cifar10 import load_cifar10
import pt_gan_trainer as my


class Dataset(torch.utils.data.Dataset):
    def __init__(self, data):
        super().__init__()
        self.images, self.labels = data

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        image = torch.tensor(image).float()/255.
        label = self.labels[idx]
        label = torch.tensor(label).long()
        return image.permute(2, 0, 1), label


def get_dataloader(data, batch_size, training=True, use_cuda=False):
    kwargs = {'num_workers': 12, 'pin_memory': False} if use_cuda else {}
    dataloader = torch.utils.data.DataLoader(dataset=Dataset(data),
                                             batch_size=batch_size,
                                             shuffle=training, **kwargs)
    return dataloader


class Generator(nn.Module):
    """ Generator for CIFAR10 (input [N, 100], output [N, 3, 32, 32]) """

    def __init__(self, noise_dim=100):
        super().__init__()
        self.noise_dim = noise_dim

        self.reshape = nn.Sequential(
            nn.Linear(noise_dim, 4*4*256, bias=False),
            nn.ReLU(inplace=True),
            nn.Unflatten(dim=1, unflattened_size=(256, 4, 4)),)

        self.model = nn.Sequential(
            nn.ConvTranspose2d(256, 128, (4, 4), stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128, momentum=0.9),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(128, 64, (4, 4), stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64, momentum=0.9),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(64, 3, (4, 4), stride=2, padding=1, bias=False),
            nn.Sigmoid(),)

    def forward(self, inputs):
        noises = self.reshape(inputs)
        outputs = self.model(noises)
        return outputs


class Discriminator(nn.Module):
    """ Discriminator for CIFAR10 (input [N, 3, 32, 32], output [N, 1]) """

    def __init__(self):
        super().__init__()

        self.model = nn.Sequential(
            nn.Conv2d(3, 64, (3, 3), stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(64, 128, (3, 3), stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(128, 256, (3, 3), stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),)

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256*4*4, 1),)

    def forward(self, inputs):
        h = self.model(inputs)
        y = self.fc(h)
        return y


def make_noises(noise_size, noise_dim):
    noises = torch.randn(noise_size, noise_dim)
    return noises.to(device)


if __name__ == "__main__":

    ## Parameters:
    p = argparse.ArgumentParser()
    p.add_argument("--image_shape", type=tuple, default=(3, 32, 32))
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--noise_dim", type=int, default=100)
    p.add_argument("--n_epochs", type=int, default=50)
    p.add_argument("--log_interval", type=int, default=5)
    p.add_argument("--cpu", action='store_const', const=True, default=False)
    p.add_argument("--log_dir", type=str, default="log_pt_cifar10_gan")
    args = p.parse_args()

    manual_seed = 42
    random.seed(manual_seed)
    np.random.seed(manual_seed)

    use_cuda = not args.cpu and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    if use_cuda:
        torch.cuda.manual_seed(manual_seed)
    else:
        torch.manual_seed(manual_seed)

    ## Dataset and Data Loaders:
    data_dir = '../../datasets/cifar10'
    train_data, valid_data, class_names = load_cifar10(data_dir, download=True)

    train_loader = get_dataloader(train_data, args.batch_size, training=True,
                                  use_cuda=use_cuda)
    valid_loader = get_dataloader(valid_data, args.batch_size, training=False,
                                  use_cuda=use_cuda)

    ## Modeling and Training:
    G = Generator().to(device)
    D = Discriminator().to(device)

    gan = my.GAN(G, D, noise_dim=args.noise_dim)
    gan.compile(g_optim=torch.optim.Adam(G.parameters(), lr=2e-4, betas=(0.5, 0.999)),
                d_optim=torch.optim.Adam(D.parameters(), lr=2e-4, betas=(0.5, 0.999)),
                loss_fn=nn.BCEWithLogitsLoss())

    sample_noises = make_noises(noise_size=50, noise_dim=args.noise_dim)
    hist = my.train(gan, train_loader, args, sample_inputs=sample_noises)
    my.plot_progress(hist, args)

    ## Evaluation:
    gen_weights = os.path.join(args.log_dir, args.log_dir + "_gen_weights.pth")
    dis_weights = os.path.join(args.log_dir, args.log_dir + "_dis_weights.pth")

    trained_G = Generator().to(device)
    trained_D = Discriminator().to(device)

    trained_G.load_state_dict(torch.load(gen_weights))
    trained_D.load_state_dict(torch.load(dis_weights))