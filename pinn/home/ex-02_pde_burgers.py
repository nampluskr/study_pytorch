import random
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.optim as optim

from pinn import gradient, PINN, Trainer


## Solve Burgers's Equation: u(x, t)
## du/dt + u * du/dx - (0.01/pi) * d2u/dxdx, x in [-1, 1], t in [0, 1],
## u(x, 0) = -sin(pi*x)         (initial condition)
## u(-1, t) = 0, u(1, t) = 0    (boundary conditions)


def residual_loss(model, inputs):
    x, t = inputs
    x.requires_grad = True
    t.requires_grad = True
    u = model([x, t])
    u_x = gradient(u, x)        # du/dx
    u_t = gradient(u, t)        # du/dt
    u_xx = gradient(u_x, x)     # d2u/dxdx
    residual = u_t + u * u_x - (0.01/np.pi) * u_xx
    return torch.mean(residual**2)


def ic_loss(model, inputs):
    x, t = inputs
    x.requires_grad = True
    t0 = torch.full_like(x, 0)          # t = 0
    u = model([x, t0])
    return torch.mean((u - ic(x))**2)


def bc_left_loss(model, inputs):
    x, t = inputs
    t.requires_grad = True
    x_left = torch.full_like(t, -1.0)   # x = -1
    u = model([x_left, t])
    return torch.mean((u - bc_left(t))**2)


def bc_right_loss(model, inputs):
    x, t = inputs
    t.requires_grad = True
    x_right = torch.full_like(t, 1.0)   # x = 1
    u = model([x_right, t])
    return torch.mean((u - bc_right(t))**2)


def ic(x):
    return -torch.sin(np.pi * x)    # u(x, 0) = -sin(pi*x)

def bc_left(t):
    return torch.full_like(t, 0.0)  # u(-1, t) = 0

def bc_right(t):
    return torch.full_like(t, 0.0)  # u(1, t) = 0


if __name__ == "__main__":

    ## Hyperparameters
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    n_epochs = 10000
    learning_rate = 1e-3
    layers = [2, 50, 50, 50, 1]


    ## Training Data
    x_min, x_max, x_size = -1, 1, 201
    t_min, t_max, t_size = 0, 1, 101

    x_np = np.linspace(x_min, x_max, x_size)
    t_np = np.linspace(t_min, t_max, t_size)

    x_train_np, t_train_np = np.meshgrid(x_np, t_np, indexing="xy")
    x_train_np, t_train_np = x_train_np.flatten(), t_train_np.flatten()

    ## numpy array to torch tensor
    x = torch.from_numpy(x_np).float().view(-1, 1).to(device)
    t = torch.from_numpy(t_np).float().view(-1, 1).to(device)
    x_train = torch.from_numpy(x_train_np).float().view(-1, 1).to(device)
    t_train = torch.from_numpy(t_train_np).float().view(-1, 1).to(device)


    ## Modeling and Training
    model = PINN(layers_dim=layers, activation="tanh").to(device)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.95)

    loss_functions = {}
    loss_functions["residual"] = residual_loss
    loss_functions["ic"] = ic_loss
    loss_functions["bc_left"] = bc_left_loss
    loss_functions["bc_right"] = bc_right_loss

    targets = {}
    t0 = torch.full_like(x, 0)          # t = 0
    x_left = torch.full_like(t, -1.0)   # x = -1
    x_right = torch.full_like(t, 1.0)   # x = 1

    # targets["ic"] = [x, t0], ic(x)
    # targets["left"] = [x_left, t], bc_left(t)
    # targets["right"] = [x_right, t], bc_right(t)

    burgers = Trainer(model, optimizer, loss_functions, targets)
    losses = burgers.fit([x_train, t_train], n_epochs, scheduler=scheduler)


    ## Results
    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(10, 4))
    for name in losses:
        epochs = range(1, n_epochs + 1)[::10]
        ax1.semilogy(epochs, losses[name][::10], label=name.upper())
    ax1.legend(); ax1.set_xlabel("Epoch"); ax1.set_ylabel("Loss")

    # ax2.plot(t, solution(t), 'k:', label="Exact")
    # ax2.plot(t_data, solution(t_data), 'ko', label="Data")
    # ax2.plot(t, predict(model, t), 'r', label="Prediction")
    # ax2.legend()
    fig.tight_layout()
    plt.show()
