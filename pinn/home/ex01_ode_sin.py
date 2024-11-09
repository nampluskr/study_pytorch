import random
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.optim as optim

from pinn import PINN, Trainer, gradient


## Solve 1st Order IVP: u(t)
## du/dt = cos(2*pi*t), u(0) = 1, t in [0, 2]

def solution(t):
    return np.sin(2*np.pi*t)/(2*np.pi) + 1


def residual_loss(model, t):
    t.requires_grad = True
    u = model(t)
    u_t = gradient(u, t)        # du/dt
    residual = u_t - torch.cos(2 * np.pi * t)
    return torch.mean(residual**2)


def ic_loss(model, t):
    t.requires_grad = True
    u = model(t)
    return torch.mean((u[0] - 1)**2)    # scalar


if __name__ == '__main__':

    ## Hyperparameters
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    n_epochs = 10000
    learning_rate = 1e-3
    layers = [1, 20, 20, 1]

    # Training data
    t_min, t_max, t_size = 0, 2, 101
    t_np = np.linspace(t_min, t_max, t_size)

    n_data = 20
    t_data_np = np.linspace(0.25, 1, n_data) + np.random.randn(n_data) * 0.1
    u_data_np = solution(t_data_np) +  np.random.randn(n_data) * 0.01


    ## numpy array to torch tensor
    t = torch.from_numpy(t_np).float().view(-1, 1).to(device)
    t_data = torch.from_numpy(t_data_np).float().view(-1, 1).to(device)
    u_data = torch.from_numpy(u_data_np).float().view(-1, 1).to(device)

    loss_functions = {}
    loss_functions["res"] = residual_loss
    loss_functions["ic"] = ic_loss

    targets = {}
    # targets["ic"] = [torch.full_like(t, t_min)], torch.full_like(t, 1)
    targets["data"] = [t_data], u_data

    # Modeling and Training
    model = PINN(layers_dim=layers, activation="tanh").to(device)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.95)

    ivp = Trainer(model, optimizer, loss_functions, targets)
    losses = ivp.fit(inputs=[t], n_epochs=n_epochs, scheduler=scheduler)


    ## Results
    t_test_np = np.linspace(t_min, t_max, 1001)
    t_test = torch.from_numpy(t_test_np).float().view(-1, 1).to(device)

    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(10, 4))
    for name in losses:
        ax1.semilogy(range(1, n_epochs + 1)[::10], losses[name][::10], label=name)
    ax1.legend()
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")

    ax2.plot(t_test_np, solution(t_test_np), 'k:', label="Exact")
    ax2.plot(t_data_np, solution(t_data_np), 'ko', label="Data")
    ax2.plot(t_test_np, ivp.predict([t_test]), 'r', label="Prediction")
    ax2.legend()
    ax2.set_xlabel("t")
    ax2.set_ylabel("u(t)")
    fig.tight_layout()
    plt.show()
