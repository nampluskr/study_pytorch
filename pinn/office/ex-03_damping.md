```python
import random
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim

from pinn import PINN, Trainer, gradient, to_tensor

seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# [Youtube] Physics-Informed Neural Networks (PINNs) - An Introduction - Ben Moseley | Jousef Murad
# Set equations (functions): du/dt = cos(2*pi*t), u(0) = 1
d, w0 = 2, 20
mu, k = 2*d, w0**2

def u_sol(t):
    w = np.sqrt(w0**2 - d**2)
    phi = np.arctan(-d/w)
    A = 1/(2*np.cos(phi))
    return 2*A*np.cos(phi + w*t)*np.exp(-d*t)

def residual_loss(model, t):
    t.requires_grad = True
    u = model(t)
    u_t = gradient(u, t)        # du/dt
    u_tt = gradient(u_t, t)     # d2u/dt2
    residual = u_tt + mu*u_t + k*u
    return torch.mean(residual**2) * 1e-4

def ic_du_loss(model, t):
    t.requires_grad = True
    u = model(t)
    u_t = gradient(u, t)
    return torch.mean((u_t[0] - 0)**2) * 1e-2

loss_functions = {}     # { name: loss_fn, ... }
loss_functions["residual"] = residual_loss
# loss_functions["ic_u"] = ic_u_loss
loss_functions["ic_du"] = ic_du_loss

# Set points (numpy arrays and torch tensors)
def to_tensor(x):
    return torch.tensor(x).float().view(-1, 1).to(device)

targets = {}            # { name: (target_inputs, target_output), ... }
targets["ic_u"] = to_tensor(0), to_tensor(1)

t = np.linspace(0, 1, 101)
inputs = to_tensor(t)   # training tensor or list of tensors

# Hyperparameters
layers = [1, 50, 50, 50, 1]
learning_rate = 1e-3
n_epochs = 20000

model = PINN(layers_dim=layers, activation="tanh").to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.95)

solver = Trainer(model, optimizer, loss_functions, targets)
solver.fit(inputs, n_epochs, scheduler=scheduler, update_step=20)
solver.show_history()


n_epochs = 20000
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.95)
solver.fit(inputs, n_epochs, scheduler=scheduler, update_step=20)
solver.show_history()

t_test = np.linspace(0, 1, 1001)
u_pred = solver.predict(to_tensor(t_test))

fig, ax = plt.subplots(figsize=(5, 3))
ax.plot(t_test, u_sol(t_test), "k:", lw=2, label="Exact")
ax.plot(t_test, u_pred, "k", lw=0.5, label="PINN")
ax.legend()
fig.tight_layout()
plt.show()
```

### Inverse PINN

```python
def residual_loss(model, t):
    t.requires_grad = True
    u = model(t)
    u_t = gradient(u, t)        # du/dt
    u_tt = gradient(u_t, t)     # d2u/dt2
    residual = u_tt + mu*u_t + k*u
    return torch.mean(residual**2) * 1e-4

loss_functions = {}     # { name: loss_fn, ... }
loss_functions["residual"] = residual_loss

n_data = 20
t_data = np.random.rand(n_data)
u_data = u_sol(t_data) + 0.04*np.random.randn(n_data)

targets = {}
targets["data"] = to_tensor(t_data), to_tensor(u_data)
# targets["ic_u"] = to_tensor(0), to_tensor(1)

t = np.linspace(0, 1, 30)
inputs = to_tensor(t)

# Hyperparameters
layers = [1, 50, 50, 50, 1]
learning_rate = 1e-3
n_epochs = 10000

mu = torch.nn.Parameter(to_tensor(0).requires_grad_(True))

model = PINN(layers_dim=layers, activation="tanh").to(device)
optimizer = optim.AdamW(list(model.parameters()) + [mu], lr=learning_rate)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.95)

solver = Trainer(model, optimizer, loss_functions, targets)
solver.fit(inputs, n_epochs, scheduler=scheduler, update_step=20)
solver.show_history()

n_epochs = 20000
learning_rate = 1e-4

solver.optimizer = optim.AdamW(list(model.parameters()) + [mu], lr=learning_rate)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.95)
solver.fit(inputs, n_epochs, scheduler=scheduler, update_step=20)
solver.show_history()

t_test = np.linspace(0, 1, 1001)
u_pred = solver.predict(to_tensor(t_test))
print(mu.item())

fig, ax = plt.subplots(figsize=(5, 3))
ax.plot(t_test, u_sol(t_test), "k:", lw=2, label="Exact")
ax.plot(t_data, u_data, "ro", ms=5, label="Data")
ax.plot(t_test, u_pred, "k", lw=0.5, label="PINN")
ax.legend()
fig.tight_layout()
plt.show()


```
