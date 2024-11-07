import sys
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn

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

    def fit(self, t, n_epochs, target=None):
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

                if epoch % 100 == 0:
                    desc = ' '.join([f"{name}: {values[-1]:.2e}" for name, values in losses.items()])
                    pbar.set_description(f"Epoch[{epoch}/{n_epochs}] " + desc)
        return losses