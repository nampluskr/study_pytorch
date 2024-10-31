import os
import gzip
import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader


def load_mnist_images(data_dir, filename):
    data_path = os.path.join(data_dir, filename)
    with gzip.open(data_path, 'rb') as f:
        data = np.frombuffer(f.read(), np.uint8, offset=16)
    return data.reshape(-1, 28, 28)


def load_mnist_labels(data_dir, filename):
    data_path = os.path.join(data_dir, filename)
    with gzip.open(data_path, 'rb') as f:
        data = np.frombuffer(f.read(), np.uint8, offset=8)
    return data


def load_mnist(data_dir):
    x_train = load_mnist_images(data_dir, "train-images-idx3-ubyte.gz")
    y_train = load_mnist_labels(data_dir, "train-labels-idx1-ubyte.gz")
    x_test = load_mnist_images(data_dir, "t10k-images-idx3-ubyte.gz")
    y_test = load_mnist_labels(data_dir, "t10k-labels-idx1-ubyte.gz")
    return (x_train, y_train), (x_test, y_test)


class MNIST(Dataset):
    def __init__(self, images, labels):
        self.images = images / 255.0
        self.labels = labels

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        image = torch.tensor(image).float().unsqueeze(dim=0)    # (N, 1, 28, 28)
        label = torch.tensor(label).long()
        return image, label
    

def get_loaders(data_dir, batch_size=64, num_workers=4):
    (x_train, y_train), (x_test, y_test) = load_mnist(data_dir)
    train_dataset = MNIST(x_train, y_train)
    test_dataset = MNIST(x_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size,
                             shuffle=False, num_workers=num_workers)
    return train_loader, test_loader


if __name__ == "__main__":

    data_dir = "/mnt/d/datasets/mnist_11M/"
    train_loader, test_loader = get_loaders(data_dir)

    x, y = next(iter(train_loader))
    print(x.shape, y.shape)     # torch.Size([64, 1, 28, 28]) torch.Size([64])