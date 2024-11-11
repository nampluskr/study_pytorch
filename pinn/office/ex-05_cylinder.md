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

# https://github.com/ComputationalDomain/PINNs/blob/main/Cylinder-Wake/NS_PINNS.py
nu = 0.01

def residual_loss(model, inputs):
    x, y, t = inputs
    x.requires_grad = True
    y.requires_grad = True
    t.requires_grad = True
    output = model([x, y, t])
    u, v, p = output[:, 0:1], output[:, 1:2], output[:, 2:3]
    
    u_x, u_y, u_t = gradient(u, x), gradient(u, y), gradient(u, t)
    u_xx, u_yy = gradient(u_x, x), gradient(u_y, y)
    
    v_x, v_y, v_t = gradient(v, x), gradient(v, y), gradient(v, t)
    v_xx, v_yy = gradient(v_x, x), gradient(v_y, y)
    
    p_x, p_y = gradient(p, x), gradient(p, y)
    
    residual_x = (u_t + u * u_x + v * u_y + p_x - nu * (u_xx + u_yy))
    residual_y = (v_t + u * v_x + v * v_y + p_y - nu * (v_xx + v_yy))
    return torch.mean(residual_x**2) + torch.mean(residual_y**2)
    
loss_functions = {}     # { name: loss_fn, ... }
loss_functions["residual"] = residual_loss

import scipy
data = scipy.io.loadmat("cylinder_wake.mat")

N, T = 5000, 200
X_star = data["X_star"]     # (N, 2): x, y
t_star = data["t"]          # (T, 1)
U_star = data["U_star"]     # (N, 2, T): u, v
p_star = data["p_star"]     # (N, T)

def to_tensor(x):
    return torch.tensor(x).float().view(-1, 1).to(device)

X_test = np.tile(X_star[:, 0:1], (1, T))    # (N, T)
Y_test = np.tile(X_star[:, 1:2], (1, T))    # (N, T)
T_test = np.tile(t_star, (1, N)).T          # (N, T)

U_test = U_star[:, 0, :]    # (N, T)
V_test = U_star[:, 1, :]    # (N, T)
P_test = p_star             # (N, T)

num_train = 5000
idx = np.random.choice(N*T, num_train, replace=False)
x_train = to_tensor(X_test.flatten()[idx])
y_train = to_tensor(Y_test.flatten()[idx])
t_train = to_tensor(T_test.flatten()[idx])

u_train = to_tensor(U_test.flatten()[idx])
v_train = to_tensor(V_test.flatten()[idx])
p_train = to_tensor(P_test.flatten()[idx])

targets = {}
targets["data"] = [x_train, y_train, t_train], torch.hstack([u_train, v_train, p_train])

inputs = [x_train, y_train, t_train]

# Hyperparameters
layers = [3, 20, 20, 20, 20, 20, 20, 20, 20, 3]
learning_rate = 1e-3
n_epochs = 10000

model = PINN(layers_dim=layers, activation="tanh").to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.985)

solver = Trainer(model, optimizer, loss_functions, targets)
solver.fit(inputs, n_epochs, scheduler=scheduler, update_step=20)
solver.show_history()

learning_rate = 1e-4
n_epochs = 10000

optimizer = optim.Adam(model.parameters(), lr=learning_rate)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.985)
solver.fit(inputs, n_epochs, scheduler=scheduler, update_step=20)
solver.show_history()

torch.save(model.state_dict(), "model_weights.pt")

x_test = X_star[:, 0]
y_test = X_star[:, 1]
t_test = np.ones_like(x_test)

u_test = U_star[:, 0, 0]
v_test = U_star[:, 1, 0]
p_test = p_star[:, 0]; print(p_test.shape)

u_pred, v_pred, p_pred = solver.predict([to_tensor(x_test), to_tensor(y_test), to_tensor(t_test)]).T

fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(nrows=3, ncols=2, figsize=(8, 8))
ax1.contourf(u_pred.reshape(50, 100), levels=100, cmap="jet")
ax3.contourf(v_pred.reshape(50, 100), levels=100, cmap="jet")
ax5.contourf(p_pred.reshape(50, 100), levels=100, cmap="jet")

ax2.contourf(u_test.reshape(50, 100), levels=100, cmap="jet")
ax4.contourf(v_test.reshape(50, 100), levels=100, cmap="jet")
ax6.contourf(p_test.reshape(50, 100), levels=100, cmap="jet")

fig.tight_layout()
plt.show()
```
