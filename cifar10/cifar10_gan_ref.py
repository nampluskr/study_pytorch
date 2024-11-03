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
from torchvision.utils import  save_image

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
            nn.Tanh()
        )

    def forward(self, input):
        return self.main(input)

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
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input)
    
if __name__ == '__main__':
    
    ## Set seed
    set_seed(42)
    
    ## Set hyper-parameters
    batch_size = 64
    learning_rate = 1e-3
    n_epochs = 100
    latent_dim = 64
    
    ## Data loaders
    data_dir = "/mnt/d/datasets/cifar10_178M/cifar-10-batches-py/"
    train_loader, test_loader = get_loaders(data_dir, 
        batch_size=batch_size, num_workers=4)
    
    ## Train model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    netG = Generator().to(device)
    netD = Discriminator().to(device)
    criterion = nn.BCELoss()
    optimizerD = optim.Adam(netD.parameters(), lr=0.0002, betas=(0.5, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=0.0002, betas=(0.5, 0.999))
    
    # GAN 학습
    n_epochs = 5
    for epoch in range(n_epochs):
        for i, (real_images, _) in enumerate(train_loader, 0):
            real_images = real_images.to(device)
            batch_size = real_images.size(0)

            # 판별자 학습: 실제 데이터
            netD.zero_grad()
            labels = torch.full((batch_size,), 1, dtype=torch.float, device=device)
            output = netD(real_images).view(-1)
            errD_real = criterion(output, labels)
            errD_real.backward()

            # 판별자 학습: 가짜 데이터
            noise = torch.randn(batch_size, 100, 1, 1, device=device)
            fake_images = netG(noise)
            labels.fill_(0)
            output = netD(fake_images.detach()).view(-1)
            errD_fake = criterion(output, labels)
            errD_fake.backward()
            optimizerD.step()

            # 생성자 학습
            netG.zero_grad()
            labels.fill_(1)
            output = netD(fake_images).view(-1)
            errG = criterion(output, labels)
            errG.backward()
            optimizerG.step()

            # 진행 상황 출력
            if i % 100 == 0:
                print(f'[{epoch + 1}/{n_epochs}][{i}/{len(train_loader)}] Loss_D: {errD_real.item() + errD_fake.item():.4f} Loss_G: {errG.item():.4f}')

            # 생성된 이미지 저장
            if (i % 500 == 0) or ((epoch == n_epochs - 1) and (i == len(train_loader) - 1)):
                save_image(fake_images[:64], 
                           f'/mnt/d/github/study_pytorch/cifar10/output_gan/fake_images_epoch_{epoch + 1}_batch_{i}.png', 
                           normalize=True)