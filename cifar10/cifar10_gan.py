import sys

common_dir = "/mnt/d/github/study_pytorch/common"
if common_dir not in sys.path:
    sys.path.append(common_dir)

from trainer import set_seed
from cifar10 import get_loaders

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision.utils import  save_image
from tqdm import tqdm


# 생성자 네트워크
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose2d(100, 512, 4, 1, 0, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 3, 4, 2, 1, bias=False),
            # nn.Tanh()
        )

    def forward(self, input):
        x = self.main(input)
        x = torch.tanh(x)
        return x

# 판별자 네트워크
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(3, 128, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 512, 4, 2, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(512, 1, 4, 1, 0, bias=False),
            # nn.Sigmoid()
        )

    def forward(self, input):
        y = self.main(input)
        y = y.view(-1)
        y = torch.sigmoid(y)
        return y

if __name__ == '__main__':

    ## Set seed
    set_seed(42)

    ## Set hyper-parameters
    batch_size = 64
    learning_rate = 1e-3
    n_epochs = 100
    latent_dim = 100

    ## Data loaders
    data_dir = "/mnt/d/datasets/cifar10_178M/cifar-10-batches-py/"
    train_loader, test_loader = get_loaders(data_dir,
        batch_size=batch_size, num_workers=4)

    ## Train model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    generator = Generator().to(device)
    discriminator = Discriminator().to(device)
    loss_fn = nn.BCELoss()
    optimizerD = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    optimizerG = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))

    fixed_noise = torch.randn(batch_size, latent_dim, 1, 1).to(device)

    # GAN 학습
    n_epochs = 50
    for e in range(n_epochs):
        with tqdm(train_loader, leave=False, file=sys.stdout, ascii=True, unit=" batch") as pbar:
            for i, (real_images, _) in enumerate(pbar):
                batch_size = real_images.size(0)
                real = torch.ones(batch_size).to(device)
                fake = torch.zeros(batch_size).to(device)
                noise = torch.randn(batch_size, latent_dim, 1, 1).to(device)
                real_images = real_images.to(device)
                fake_images = generator(noise)

                ## Training Discriminator
                discriminator.zero_grad()

                pred_r = discriminator(real_images)
                loss_r = loss_fn(pred_r, real)
                loss_r.backward()

                pred_f = discriminator(fake_images.detach())
                loss_f = loss_fn(pred_f, fake)
                loss_f.backward()
                optimizerD.step()

                # Training Generator
                generator.zero_grad()
                pred_g = discriminator(fake_images)
                loss_g = loss_fn(pred_g, real)
                loss_g.backward()
                optimizerG.step()

                if i % 10 == 0:
                    desc = f"[{e+1}/{n_epochs}] loss_r: {loss_r.item():.4f} " \
                           f"loss_f: {loss_f.item():.4f} loss_g: {loss_g.item():.4f}"
                    pbar.set_description(desc)

            # 생성된 이미지 저장
            print(desc)
            gen_images = generator(fixed_noise)
            save_image(gen_images,
                       f'/mnt/d/github/study_pytorch/cifar10/output_gan/fake_images_epoch_{e + 1}.png',
                       normalize=True)