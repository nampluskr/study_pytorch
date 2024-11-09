### u(t) = sin(2pi t) / (2pi)

```python
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim

np.random.seed(42)
torch.manual_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def loss_fn(model, t, u_data):
    t = torch.tensor(t).float().view(-1, 1).to(device)
    u_data = torch.tensor(u_data).float().view(-1, 1).to(device)
    t.requires_grad = True

    u = model(t)
    u_t = torch.autograd.grad(u, t, torch.ones_like(u), create_graph=True)[0]

    mse_loss = nn.MSELoss()
    loss_ode = mse_loss(u_t, torch.cos(2 * np.pi * t))
    loss_ic = mse_loss(model(torch.zeros_like(t)), torch.ones_like(t))
    loss_data = mse_loss(u, u_data)
    return loss_ode + loss_ic + loss_data

def u_exact(t):
    return np.sin(2*np.pi*t)/(2*np.pi) + 1

@torch.no_grad()
def predict(model, t):
    model.eval()
    t = torch.tensor(t).float().view(-1, 1).to(device)
    pred = model(t)
    return pred.detach().cpu().numpy()

# Data
n_samples = 50
# t_data = np.random.rand(n_samples) * 2
t_data = np.linspace(0.25, 1, n_samples) + np.random.randn(n_samples) * 0.1
u_data = u_exact(t_data) +  np.random.randn(n_samples) * 0.1

# Model
hidden_dim = 32
model = nn.Sequential(
    nn.Linear(1, hidden_dim), nn.Tanh(),
    nn.Linear(hidden_dim, 1)
).to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# Train
n_epochs = 20000
losses = []
model.train()
for epoch in range(1, n_epochs + 1):
    optimizer.zero_grad()
    loss = loss_fn(model, t_data, u_data)
    loss.backward()
    optimizer.step()

    losses.append(loss.item())
    if epoch % (n_epochs // 10) == 0:
        print(f"Epoch[{epoch}/{n_epochs}] loss: {losses[-1]:.4f}")

plt.semilogy(losses)
plt.show()
```

```python
t = np.linspace(0, 2, 101)

fig, ax = plt.subplots()
ax.plot(t, u_exact(t), 'k', label="Exact")
ax.plot(t_data, u_exact(t_data), 'ko', label="Data")
ax.plot(t, predict(model, t), 'r:', label="Prediction")
ax.legend()
plt.show()
```

### Cooling

```python
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim

np.random.seed(42)
torch.manual_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def to_torch(x):
    return torch.tensor(t).float().view(-1, 1).to(device)

def to_numpy(t):
    return t.detach().cpu().numpy()

def loss_fn(model, t, T_data, Tenv=25, T0=100, R=0.005):
    t = to_torch(t)
    t.requires_grad = True
    # t_data = to_torch(t_data)
    T_data = to_torch(T_data)

    T = model(t)
    T_t = torch.autograd.grad(T, t, torch.ones_like(T), create_graph=True)[0]

    mse_loss = nn.MSELoss()
    loss_ode = mse_loss(T_t, R*(Tenv - T))
    loss_ic = mse_loss(model(torch.zeros_like(t)), T0*torch.ones_like(t))
    loss_data = mse_loss(model(t), T_data)
    return loss_ode + loss_ic #  + loss_data

@torch.no_grad()
def predict(model, t):
    model.eval()
    t = torch.tensor(t).float().view(-1, 1).to(device)
    pred = model(t)
    return pred.detach().cpu().numpy()

# Data
def T_exact(t, Tenv=25, T0=100, R=0.005):
    return Tenv + (T0 - Tenv) * np.exp(-R * t)

t = np.linspace(0, 1000, 1001)
n_samples = 30
t_data = np.linspace(200, 600, n_samples)
T_data = T_exact(t_data) + np.random.randn(n_samples) * 0.1

# Model
hidden_dim = 100
model = nn.Sequential(
    nn.Linear(1, hidden_dim),          nn.Tanh(),
    nn.Linear(hidden_dim, hidden_dim), nn.Tanh(),
    nn.Linear(hidden_dim, hidden_dim), nn.Tanh(),
    nn.Linear(hidden_dim, 1)
).to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# Train
n_epochs = 10000
losses = []
model.train()
for epoch in range(1, n_epochs + 1):
    optimizer.zero_grad()
    loss = loss_fn(model, t_data, T_data)
    loss.backward()
    optimizer.step()

    losses.append(loss.item())
    if epoch % (n_epochs // 10) == 0:
        print(f"Epoch[{epoch}/{n_epochs}] loss: {losses[-1]:.4f}")

plt.semilogy(losses)
plt.show()
```

```python
t = np.linspace(0, 1000, 1001)
T_pred = predict(model, t)

plt.plot(t, T_exact(t), 'k', label="Exact")
plt.plot(t_data, T_data, 'o', label="Data")
plt.plot(t, T_pred, 'r:', label="Prediction")
plt.legend()
plt.grid()
plt.show()
```
