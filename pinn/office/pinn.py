import sys
from tqdm import tqdm
import torch
import torch.nn as nn
import matplotlib.pyplot as plt


def gradient(y, x):
    """ return dy/dx """
    return torch.autograd.grad(y, x, grad_outputs=torch.ones_like(y),
                create_graph=True, retain_graph=True)[0]

def to_tensor(x):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if x.ndim == 1:
        return torch.from_numpy(x).float().view(-1, 1).to(device)
    else:
        return torch.from_numpy(x).float().to(device)

def tensor_like(x, value):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.full_like(x, value).to(device)

def to_numpy(x):
    return x.detach().cpu().squeeze().numpy()


class PINN(nn.Module):
    def __init__(self, layers_dim, activation="tanh"):
        super().__init__()
        self.input_dim = layers_dim[0]
        activation_functions = {"tanh": nn.Tanh(), "sigmoid": nn.Sigmoid(),
                     "swish": nn.SiLU(), "silu": nn.SiLU(),
                     "elu": nn.ELU(), "gelu": nn.GELU(),
                     "relu": nn.ReLU(), "leakyrelu": nn.LeakyReLU()}
        layers = []
        for i in range(len(layers_dim) - 2):
            layers.append(nn.Linear(layers_dim[i], layers_dim[i+1]))
            layers.append(activation_functions[activation.lower()])
        layers.append(nn.Linear(layers_dim[-2], layers_dim[-1]))

        self.pinn = nn.Sequential(*layers)

    def forward(self, inputs):
        if self.input_dim > 1:
            inputs = torch.hstack(inputs)
            # inputs = torch.cat(inputs, dim=1)
        return self.pinn(inputs)


class Trainer:
    def __init__(self, model, optimizer, loss_functions={}, targets={}):
        self.model = model
        self.optimizer = optimizer
        self.loss_functions = loss_functions
        self.targets = targets
        self.mse = nn.MSELoss()
        # self.device = next(model.parameters()).device

        self.history = {"total": []}
        for name in self.loss_functions:
            self.history[name] = []
        for name in self.targets:
            self.history[name] = []

    def fit(self, inputs, n_epochs, scheduler=None, update_step=10):
        with tqdm(range(1, n_epochs+1), file=sys.stdout, ascii=True, ncols=200) as pbar:
            for epoch in pbar:
                total_loss = 0

                for name in self.loss_functions:
                    loss_value = self.loss_functions[name](self.model, inputs)
                    self.history[name].append(loss_value.item())
                    total_loss += loss_value

                for name in self.targets:
                    target_inputs, target_output = self.targets[name]
                    loss_target = self.mse(self.model(target_inputs), target_output)
                    self.history[name].append(loss_target.item())
                    total_loss += loss_target

                self.history["total"].append(total_loss.item())
                self.optimizer.zero_grad()
                total_loss.backward()
                self.optimizer.step()

                desc = f"Epoch[{epoch}/{n_epochs}] "
                if scheduler is not None:
                    desc += f"(lr: {scheduler.get_last_lr()[0]:.2e}) "
                    scheduler.step()

                if epoch % update_step == 0:
                    desc += ', '.join([f'{k.upper()}: {v[-1]:.2e}' for k, v in self.history.items()])
                    pbar.set_description(desc)
        return self.history

    @torch.no_grad()
    def predict(self, inputs):
        self.model.eval()
        pred = self.model(inputs)
        return pred.detach().cpu().squeeze().numpy()

    def show_history(self):
        fig, ax = plt.subplots(figsize=(5, 3))
        for name in self.history:
            epochs = range(1, len(self.history["total"]) + 1)
            ax.semilogy(epochs[::10], self.history[name][::10],
                        label=name.upper())
        ax.legend()
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        fig.tight_layout()
        plt.show()
