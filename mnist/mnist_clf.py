import sys

common_dir = "/mnt/d/github/study_pytorch/common"
if common_dir not in sys.path:
    sys.path.append(common_dir)

from trainer import Trainer, set_seed, EarlyStopping
from mnist import get_loaders

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


## Model
class MLP(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        self.flatten = nn.Flatten()
        self.fc = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.LayerNorm(256),
            nn.Linear(256, n_classes),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.fc(x)
        return logits


class CNN(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, n_classes)

    def forward(self, x):
        x = x.view(-1, 1, 28, 28)
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        logits = self.fc3(x)
        return logits


def accuracy(y_pred, y_true):
    y_pred = y_pred.argmax(dim=1)   # int64 (long)
    return torch.eq(y_pred, y_true).float().mean()


if __name__ == "__main__":

    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ## Hyper-parameters
    learning_rate = 1e-3
    batch_size = 64
    n_epochs = 100

    ## Data loaders
    # data_dir = "/mnt/d/datasets/mnist_11M/"
    data_dir = "/mnt/d/datasets/fashion_mnist_29M/"
    
    train_loader, test_loader = get_loaders(data_dir, batch_size=batch_size,
                                            num_workers=4)

    ## Train model
    model = MLP(n_classes=10).to(device)
    # model = CNN(n_classes=10).to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    metrics = {"acc": accuracy}
    
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
    early_stopper = EarlyStopping(patience=5, min_delta=0.001)

    clf = Trainer(model, optimizer, loss_fn, metrics)
    clf.fit(train_loader, n_epochs, valid_loader=test_loader,
            stopper=early_stopper, scheduler=scheduler)
    res = clf.evaluate(test_loader)

