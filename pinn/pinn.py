import sys
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


class Dataset(torch.utils.data.Dataset):
    def __init__(self, t, transform=None):
        self.t = t
        if t.ndim == 1:
            self.t = self.t.reshape(-1, 1)
        self.transform = transform

    def __len__(self):
        return len(self.t)

    def __getitem__(self, idx):
        t = torch.tensor(self.t[idx]).float()
        if self.transform:
            t = self.transform(t)
        return t

class PINN(nn.Module):
    def __init__(self, layers_dim=[1, 100, 1], activation="tanh"):
        super().__init__()
        functions = {"tanh": nn.Tanh(), "sigmoid": nn.Sigmoid(),
                     "swish": nn.SiLU(), "silu": nn.SiLU(),
                     "elu": nn.ELU(), "gelu": nn.GELU(),
                     "relu": nn.ReLU(), "leakyrelu": nn.LeakyReLU()}
        layers = []
        for i in range(len(layers_dim) - 2):
            layers.append(nn.Linear(layers_dim[i], layers_dim[i+1]))
            layers.append(functions[activation.lower()])
        layers.append(nn.Linear(layers_dim[-2], layers_dim[-1]))

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

def to_torch(x: np.ndarray):
    tensor = torch.from_numpy(x).float()
    if x.ndim == 1:
        tensor = tensor.view(-1, 1)
    return tensor

def to_numpy(x: torch.Tensor):
    return x.detach().cpu().numpy()

def gradient(y: torch.Tensor, x: torch.Tensor):
    return torch.autograd.grad(y, x,
                grad_outputs=torch.ones_like(x),
                create_graph = True,
                only_inputs=True)[0]

@torch.no_grad()
def predict(model, x: np.ndarray):
    model.eval()
    pred = model(to_torch(x))
    return to_numpy(pred)

def init_normal(m):
    if type(m) == nn.Linear:
        nn.init.kaiming_normal_(m.weight)

class Trainer:
    def __init__(self, model, optimizer, loss_fn):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.device = next(model.parameters()).device

    def init(self):
        self.model.apply(init_normal)
        return self

    def mse_loss(self, x, y):
        return torch.mean((self.model(x) - y)**2)

    def fit(self, t, n_epochs, target=None, scheduler=None):
        self.model.train()
        t = to_torch(t).to(self.device)

        losses = {"total": [], "eqn": []}
        if target is not None:
            for name in target:
                losses[name] = []
                target[name][0] = to_torch(target[name][0]).to(self.device)
                target[name][1] = to_torch(target[name][1]).to(self.device)

        with tqdm(range(1, n_epochs + 1), file=sys.stdout, ascii=True) as pbar:
            for epoch in pbar:
                loss_total = loss_eqn = self.loss_fn(self.model, t)
                losses["eqn"].append(loss_eqn.item())

                if target is not None:
                    for name in target:
                        loss_data = self.mse_loss(target[name][0], target[name][1])
                        loss_total += loss_data
                        losses[name].append(loss_data.item())

                losses["total"].append(loss_total.item())

                self.optimizer.zero_grad()
                loss_total.backward()
                self.optimizer.step()

                desc = f"Epoch[{epoch}/{n_epochs}] "
                if scheduler is not None:
                    desc += f"lr: {scheduler.get_last_lr()[0]:.2e} "
                    scheduler.step()

                if epoch % 100 == 0:
                    desc += ' '.join([f"{name}: {values[-1]:.2e}" for name, values in losses.items()])
                    pbar.set_description(desc)
        return losses


class BatchTrainer(Trainer):
    def fit(self, t, n_epochs, batch_ratio=1.0, target=None, scheduler=None):
        self.model.train()
        batch_size = int(len(t) * batch_ratio)
        batch_loader = DataLoader(Dataset(t), batch_size=batch_size, shuffle=True)

        losses = {"total": [], "eqn": []}
        if target is not None:
            for name in target:
                losses[name] = []
                target[name][0] = to_torch(target[name][0]).to(self.device)
                target[name][1] = to_torch(target[name][1]).to(self.device)

        with tqdm(range(1, n_epochs + 1), file=sys.stdout, ascii=True) as pbar:
            for epoch in pbar:
                batch_losses = {name: 0 for name in losses}
                for i, t in enumerate(batch_loader):
                    t = t.to(self.device)
                    loss_total = loss_eqn = self.loss_fn(self.model, t)
                    batch_losses["eqn"] += loss_eqn.item()

                    if target is not None:
                        for name in target:
                            loss_data = self.mse_loss(target[name][0], target[name][1])
                            loss_total += loss_data
                            batch_losses[name] += loss_data.item()

                    batch_losses["total"] += loss_total.item()

                    self.optimizer.zero_grad()
                    loss_total.backward()
                    self.optimizer.step()

                for name in losses:
                    losses[name].append(batch_losses[name] / (i + 1))

                desc = f"Epoch[{epoch}/{n_epochs}] "
                if scheduler is not None:
                    desc += f"(lr: {scheduler.get_last_lr()[0]:.2e}) "
                    scheduler.step()

                if epoch % 100 == 0:
                    desc += ' '.join([f"{name}: {values[-1]:.2e}" for name, values in losses.items()])
                    pbar.set_description(desc)
        return losses