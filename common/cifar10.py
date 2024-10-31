import os
import pickle
import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms


def unpickle(filename):
    # tar -zxvf cifar-10-python.tar.gz
    with open(filename, 'rb') as f:
        data = pickle.load(f, encoding='bytes')

    x = np.array(data[b'data']).reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    y = np.array(data[b'labels'])
    return x, y


def load_cifar10(data_dir):
    batch_files = [os.path.join(data_dir, f"data_batch_{i+1}") for i in range(5)]
    test_file = os.path.join(data_dir, "test_batch")

    images, labels = [], []
    for filename in batch_files:
        x, y = unpickle(filename)
        images.append(x)
        labels.append(y)

    x_train = np.concatenate(images, axis=0)
    y_train = np.concatenate(labels, axis=0)

    x_test, y_test = unpickle(test_file)
    return (x_train, y_train), (x_test, y_test)


class CIFAR10(Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        label = torch.tensor(label).long()
        return image, label


def get_loaders(data_dir, batch_size=64, num_workers=4):
    transform_train = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomHorizontalFlip(0.3),
        transforms.RandomVerticalFlip(0.3),
        transforms.ToTensor(),
        # transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        # transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])

    (x_train, y_train), (x_test, y_test) = load_cifar10(data_dir)
    train_dataset = CIFAR10(x_train, y_train, transform=transform_train)
    test_dataset = CIFAR10(x_test, y_test, transform=transform_test)

    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size,
                             shuffle=False, num_workers=num_workers)
    return train_loader, test_loader


if __name__ == "__main__":

    data_dir = "/mnt/d/datasets/cifar10_178M/cifar-10-batches-py/"
    train_loader, test_loader = get_loaders(data_dir)

    x, y = next(iter(train_loader))
    print(x.shape, y.shape)     # torch.Size([64, 3, 32, 32]) torch.Size([64]) 