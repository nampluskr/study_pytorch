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

# Set equations (functions): Burgers's Equation u(x, t)
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
    return torch.mean(residual**2) * 1e1

def ic_loss(model, inputs):
    x, t = inputs
    x.requires_grad = True
    t0 = torch.full_like(x, 0)          # t = 0
    u = model([x, t0])
    return torch.mean((u - ic(x))**2) * 1e2

# def bc_left_loss(model, inputs):
#     x, t = inputs
#     t.requires_grad = True
#     x_left = torch.full_like(t, -1.0)   # x = -1
#     u = model([x_left, t])
#     return torch.mean((u - bc_left(t))**2)

# def bc_right_loss(model, inputs):
#     x, t = inputs
#     t.requires_grad = True
#     x_right = torch.full_like(t, 1.0)   # x = 1
#     u = model([x_right, t])
#     return torch.mean((u - bc_right(t))**2)

def ic(x):
    return -torch.sin(np.pi * x)    # u(x, 0) = -sin(pi*x)

# def bc_left(t):
#     return torch.full_like(t, 0.0)  # u(-1, t) = 0

# def bc_right(t):
#     return torch.full_like(t, 0.0)  # u(1, t) = 0

loss_functions = {}     # { name: loss_fn, ... }
loss_functions["residual"] = residual_loss
loss_functions["ic"] = ic_loss
# loss_functions["bc_left"] = bc_left_loss
# loss_functions["bc_right"] = bc_right_loss

# Set points (numpy arrays and torch tensors)
def to_tensor(x):
    return torch.tensor(x).float().view(-1, 1).to(device)

x_np = np.linspace(-1, 1, 201)
t_np = np.linspace(0, 1, 101)
X_np, T_np = np.meshgrid(x_np, t_np)

x_train, t_train = to_tensor(X_np.flatten()), to_tensor(T_np.flatten())
x, t = to_tensor(x_np), to_tensor(t_np)
t0 = torch.full_like(x, 0)
x_min = torch.full_like(t, -1)
x_max = torch.full_like(t, 1)

targets = {}            # { name: (target_inputs, target_output), ... }
# targets["ic"] = [x, t0], -torch.sin(np.pi * x)
targets["bc_xmin"] = [x_min, t], torch.full_like(t, 0)
targets["bc_xmax"] = [x_max, t], torch.full_like(t, 0)

inputs = [x_train, t_train]   # training tensor or list of tensors

# Hyperparameters
layers = [2, 50, 100, 50, 1]
learning_rate = 1e-3
n_epochs = 10000

model = PINN(layers_dim=layers, activation="tanh").to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.985)

solver = Trainer(model, optimizer, loss_functions, targets)
solver.fit(inputs, n_epochs, scheduler=scheduler, update_step=20)
solver.show_history()

n_epochs = 10000
learning_rate = 1e-4

optimizer = optim.Adam(model.parameters(), lr=learning_rate)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.985)

solver = Trainer(model, optimizer, loss_functions, targets)
solver.fit(inputs, n_epochs, scheduler=scheduler, update_step=20)
solver.show_history()

import scipy
data = scipy.io.loadmat("burgers_shock.mat")
U_sol = np.real(data["usol"]).T

x_test = np.linspace(-1, 1, 256)
t_test = np.linspace(0, 1, 100)
X_test, T_test = np.meshgrid(x_test, t_test)
U_pred = solver.predict([to_tensor(X_test.flatten()), to_tensor(T_test.flatten())])
U_pred = U_pred.reshape(X_test.shape)

fig, (ax1, ax2, ax3) = plt.subplots(ncols=3, figsize=(10, 3))
ax1.plot(x_test, U_sol[0, :], 'k:', label="t=0")
ax1.plot(x_test, U_pred[0, :], 'k')
ax1.plot(x_test, U_sol[25, :], 'g:', label="t=0.25")
ax1.plot(x_test, U_pred[25, :], 'g')
ax1.plot(x_test, U_sol[50, :], 'r:', label="t=0.5")
ax1.plot(x_test, U_pred[50, :], 'r')
ax1.legend(loc="upper right"); ax2.set_xlabel("x"); ax2.set_ylabel("u(x, t)")
    
ax1.plot(x_test, U_sol[0, :], 'k:')
cp1 = ax2.contourf(X_test, T_test, U_sol, levels=100, cmap="jet")
cp2 = ax3.contourf(X_test, T_test, U_pred, levels=100, cmap="jet")
fig.colorbar(cp1)
fig.colorbar(cp2)
fig.tight_layout()
plt.show()

```


