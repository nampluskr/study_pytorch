import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim

import sys
from tqdm import tqdm

class PINN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.hidden = nn.Sequential(
            nn.Linear(input_dim, 20),  nn.Tanh(),
            nn.Linear(20, 20), nn.Tanh(),
            nn.Linear(20, 20), nn.Tanh(),
            nn.Linear(20, 20), nn.Tanh(),
            nn.Linear(20, output_dim),
        )

    def forward(self, x, t):
        inputs = torch.cat([x, t], dim=1)
        return self.hidden(inputs)

def gradient(y, x):
    return torch.autograd.grad(y, x,
                grad_outputs=torch.ones_like(y),
                create_graph=True,
                retain_graph=True)[0]

if __name__ == "__main__":

    def residual(model, x, t):
        x.requires_grad = True
        t.requires_grad = True
        u = model(x, t)
        u_x, u_t = gradient(u, x), gradient(u, t)
        u_xx = gradient(u_x, x)
        return u_t + u * u_x - 0.01/np.pi * u_xx

    def ic(x):
        return -torch.sin(np.pi * x)

    def bc_left(t):
        return torch.zeros_like(t)

    def bc_right(t):
        return torch.zeros_like(t)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PINN(input_dim=2, output_dim=1).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    x_min, x_max, x_size = -1, 1, 200
    t_min, t_max, t_size = 0, 1, 100

    x = torch.linspace(x_min, x_max, x_size).view(-1, 1)
    t = torch.linspace(t_min, t_max, t_size).view(-1, 1)
    x_train, t_train = torch.meshgrid(x.squeeze(), t.squeeze(), indexing="xy")
    x_train = x_train.reshape(-1, 1)
    t_train = t_train.reshape(-1, 1)


    x, t = x.to(device), t.to(device)
    x_train, t_train = x_train.to(device), t_train.to(device)

    n_epochs = 10000
    with tqdm(range(1, n_epochs+1), file=sys.stdout, desc="Training", unit="epoch",
            ascii=True) as pbar:
        for epoch in pbar:

            loss_res = torch.mean(residual(model, x_train, t_train)**2)

            x0, t0, u0 = x, torch.zeros_like(x), ic(x)
            loss_ic = torch.mean((model(x0, t0) - u0)**2)

            xb, tb, ub = torch.full_like(t, -1), t, bc_left(t)
            loss_bc_left = torch.mean((model(xb, tb) - ub)**2)

            xb, tb, ub = torch.full_like(t, 1), t, bc_right(t)
            loss_bc_right = torch.mean((model(xb, tb) - ub)**2)

            total_loss = loss_res + loss_ic + loss_bc_left + loss_bc_right

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            if epoch % 20 == 0:
                pbar.set_postfix({"Loss": f'{total_loss.item():.2e}',
                                "Res": f'{loss_res.item():.2e}',
                                "IC": f'{loss_ic.item():.2e}',
                                "BC": f'{(loss_bc_left.item() + loss_bc_right.item()):.2e}'})