import sys
from tqdm import tqdm
import torch
import torch.nn as nn


def gradient(y, x):
    return torch.autograd.grad(y, x, grad_outputs=torch.ones_like(y),
                create_graph=True, retain_graph=True)[0]


class PINN(nn.Module):
    def __init__(self, layers_dim, activation="tanh"):
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

        self.pinn = nn.Sequential(*layers)

    def forward(self, *inputs):
        inputs = torch.cat(inputs, dim=1)
        return self.pinn(inputs)


class Trainer:
    def __init__(self, model, optimizer, loss_functions={}, targets={}):
        self.model = model
        self.optimizer = optimizer
        self.loss_functions = loss_functions
        self.targets = targets
        self.mse = nn.MSELoss()
        # self.device = next(model.parameters()).device

    def fit(self, inputs, n_epochs, scheduler=None):
        losses = {"total": []}
        for name in self.loss_functions:
            losses[name] = []
        for name in self.targets:
            losses[name] = []

        with tqdm(range(1, n_epochs+1), file=sys.stdout, ascii=True, ncols=150) as pbar:
            for epoch in pbar:
                total_loss = 0

                for name in self.loss_functions:
                    loss_value = self.loss_functions[name](self.model, *inputs)
                    losses[name].append(loss_value.item())
                    total_loss += loss_value

                for name in self.targets:
                    target_inputs, target_output = self.targets[name]
                    loss_target = self.mse(self.model(*target_inputs), target_output)
                    losses[name].append(loss_target.item())
                    total_loss += loss_target

                losses["total"].append(total_loss.item())

                self.optimizer.zero_grad()
                total_loss.backward()
                self.optimizer.step()

                desc = f"Epoch[{epoch}/{n_epochs}] "
                if scheduler is not None:
                    desc += f"(lr: {scheduler.get_last_lr()[0]:.2e}) "
                    scheduler.step()

                if epoch % 50 == 0:
                    desc += ' '.join([f'{k.upper()}: {v[-1]:.2e}' for k, v in losses.items()])
                    pbar.set_description(desc)

        return losses

    @torch.no_grad()
    def predict(self, inputs):
        self.model.eval()
        pred = self.model(*inputs)
        return pred.detach().cpu().numpy()