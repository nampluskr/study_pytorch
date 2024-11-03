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
        self.label_emb = nn.Embedding(10, 10)
        self.main = nn.Sequential(
            nn.ConvTranspose2d(110, 512, 4, 1, 0, bias=False),
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

    def forward(self, noise, labels):
        label_embedding = self.label_emb(labels).unsqueeze(2).unsqueeze(3)
        input = torch.cat((noise, label_embedding), dim=1)
        return self.main(input)

# 판별자 네트워크
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.label_emb = nn.Embedding(10, 10)
        self.main = nn.Sequential(
            nn.Conv2d(13, 128, 4, 2, 1, bias=False),
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

    def forward(self, img, labels):
        label_embedding = self.label_emb(labels).unsqueeze(2).unsqueeze(3)
        label_embedding = label_embedding.repeat(1, 1, img.size(2), img.size(3))
        input = torch.cat((img, label_embedding), dim=1)
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
        for i, data in enumerate(train_loader, 0):
            real_images, labels = data
            real_images = real_images.to(device)
            labels = labels.to(device)
            batch_size = real_images.size(0)

            # 판별자 학습
            netD.zero_grad()
            labels_real = torch.full((batch_size,), 1, dtype=torch.float, device=device)
            output_real = netD(real_images, labels).view(-1)
            lossD_real = criterion(output_real, labels_real)
            lossD_real.backward()

            noise = torch.randn(batch_size, 100, 1, 1, device=device)
            fake_images = netG(noise, labels)
            labels_fake = torch.full((batch_size,), 0, dtype=torch.float, device=device)
            output_fake = netD(fake_images.detach(), labels).view(-1)
            lossD_fake = criterion(output_fake, labels_fake)
            lossD_fake.backward()
            optimizerD.step()

            # 생성자 학습
            netG.zero_grad()
            labels_gen = torch.full((batch_size,), 1, dtype=torch.float, device=device)
            output_gen = netD(fake_images, labels).view(-1)
            lossG = criterion(output_gen, labels_gen)
            lossG.backward()
            optimizerG.step()

            # 진행 상황 출력
            if i % 100 == 0:
                print(f'[{epoch + 1}/{n_epochs}][{i}/{len(train_loader)}] Loss_D: {lossD_real.item() + lossD_fake.item():.4f} Loss_G: {lossG.item():.4f}')

            # 생성된 이미지 저장
            if (i % 500 == 0) or ((epoch == n_epochs - 1) and (i == len(train_loader) - 1)):
                save_image(fake_images[:64], 
                           f'/mnt/d/github/study_pytorch/cifar10/output_cgan/fake_images_epoch_{epoch + 1}_batch_{i}.png', 
                           normalize=True)