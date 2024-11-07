import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim

from pinn import Trainer, to_torch, to_numpy, gradient, predict


def get_model(input_dim, output_dim, hidden_dim=100):
    model = nn.Sequential(
        nn.Linear(input_dim, hidden_dim), nn.Tanh(),
        nn.Linear(hidden_dim, hidden_dim), nn.Tanh(),
        nn.Linear(hidden_dim, output_dim)
    )
    return model



if __name__ == "__main__":

    seed = 42
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def solution(t):
        return np.sin(2*np.pi*t)/(2*np.pi) + 1

    def loss_fn(model, t):
        t.requires_grad = True
        u = model(t)
        u_t = gradient(u, t)
        eqn = u_t - torch.cos(2 * np.pi * t)
        return torch.mean(eqn**2)

    # Hyperparameters
    learning_rate = 1e-4
    n_epochs = 10000
    n_samples = 50

    # Model
    model = get_model(input_dim=1, output_dim=1, hidden_dim=100).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Data
    t_data = np.linspace(0.25, 1, n_samples) + np.random.randn(n_samples) * 0.1
    u_data = solution(t_data) +  np.random.randn(n_samples) * 0.1

    # Mesh
    t = np.linspace(0, 2, 1001)
    ones = np.ones_like(t_data)

    # target = {"ic": [0*ones, 1*ones]}
    target = {"ic": [0*ones, 1*ones], "data": [t_data, u_data]}

    # Train
    ivp = Trainer(model, optimizer, loss_fn).init()
    losses = ivp.fit(t, n_epochs, target=target)

    t = np.linspace(-1, 3, 1001)
    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(10, 4))
    for name in losses:
        ax1.semilogy(losses[name], label=name)
    ax1.legend(); ax1.set_xlabel("Epoch"); ax1.set_ylabel("Loss")

    ax2.plot(t, solution(t), 'k:', label="Exact")
    ax2.plot(t_data, solution(t_data), 'ko', label="Data")
    ax2.plot(t, predict(model, t), 'r', label="Prediction")
    ax2.legend()
    plt.show()
