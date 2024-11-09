import random
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.optim as optim

from pinn import PINN, Trainer, gradient


## Solve systems of ODEs: x(t), y(t)
## dx/dt + 2*x + y = 0
## dy/dt + x + 2*y = 0
## x(0) = 1, y(0) = 0, t in [0, 5]    (initial condition)


def solution_x(t):
    return np.exp(-t)/2 + np.exp(-3*t)/2


def solution_y(t):
    return -np.exp(-t)/2 + np.exp(-3*t)/2


def residual_loss(model, t):
    t.requires_grad = True
    output = model(t)
    x, y = output[:, 0:1], output[:, 1:2]   # ndim == 2
    x_t = gradient(x, t)        # dx/dt
    y_t = gradient(y, t)        # dx/dt
    residual_x = x_t + 2*x + y
    residual_y = y_t + x + 2*y
    return torch.mean(residual_x**2) + torch.mean(residual_y**2)


def ic_loss(model, t):
    t.requires_grad = True
    output = model(t)
    x, y = output.T       # ndim == 1
    ic_x_loss = x[0] - 1.0
    ic_y_loss = y[0] - 0.0
    return torch.mean((ic_x_loss)**2) + torch.mean((ic_y_loss)**2)


if __name__ == '__main__':

    ## Hyperparameters
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    n_epochs = 5000
    learning_rate = 1e-3
    layers = [1, 64, 64, 2]
    
    ## Training Data
    t_min, t_max, t_size = 0, 5, 1001
    t_np = np.linspace(t_min, t_max, t_size)
    t = torch.from_numpy(t_np).float().view(-1, 1).to(device)

   
    ## Modeling and Training
    model = PINN(layers_dim=layers, activation="tanh").to(device)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.95)

    loss_functions = {}
    loss_functions["res"] = residual_loss
    loss_functions["ic"] = ic_loss

    targets = {}

    odes = Trainer(model, optimizer, loss_functions, targets)
    history = odes.fit(t, n_epochs, scheduler=scheduler)
    
    ## Results
    t_test_np = np.linspace(t_min, t_max, 1001)
    t_test = torch.from_numpy(t_test_np).float().view(-1, 1).to(device)
    pred_x_np, pred_y_np = odes.predict(t_test).T

    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(10, 4))
    for name in history:
        epochs = range(1, n_epochs + 1)
        ax1.semilogy(epochs[::10], history[name][::10], label=name.upper())
    ax1.legend()
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")

    ax2.plot(t_test_np, solution_x(t_test_np), 'b:', label="Exact x(t)")
    ax2.plot(t_test_np, pred_x_np, 'b', label="PINN x(t)")
    ax2.plot(t_test_np, solution_y(t_test_np), 'r:', label="Exact y(t)")
    ax2.plot(t_test_np, pred_y_np, 'r', label="PINN y(t)")
    ax2.legend()
    ax2.set_xlabel("t")
    fig.tight_layout()
    plt.show()
    
    