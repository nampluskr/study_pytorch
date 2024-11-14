## Operator Learning and Implementation in Pytorch

- https://johncsu.github.io/DeepONet_Demo/
- https://github.com/JohnCSu/DeepONet_Pytorch_Demo/blob/main/DeepONet.ipynb

```python
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

seed = 42
np.random.seed(seed)
torch.manual_seed(seed)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class MLP(nn.Module):
    def __init__(self, in_size, out_size, hidden_size, num_hidden):
        super().__init__()
        self.linear_in = nn.Linear(in_size, hidden_size)
        self.linear_out = nn.Linear(hidden_size, out_size)
        self.activation = torch.tanh
        self.layers = nn.ModuleList([self.linear_in] + [nn.Linear(hidden_size, hidden_size)] * num_hidden)

    def forward(self,x):
        for layer in self.layers:
            x = self.activation(layer(x))
        return self.linear_out(x)

class DeepONet(nn.Module):
    def __init__(self, latent_features, out_features, branch, trunk):
        super().__init__()
        self.branch = branch
        self.trunk = trunk
        self.fc = nn.Linear(latent_features, out_features, bias = False)

    def forward(self, y, u):
        return self.fc(self.branch(u) * self.trunk(y))

from numpy.polynomial.chebyshev import chebval
from scipy.integrate import cumulative_trapezoid

def get_data(n_samples, n_points, degree=30, M=10, seed=42):
    np.random.seed(seed)
    y = np.empty((n_samples, n_points))
    u = np.empty((n_samples, n_points))
    Guy = np.empty((n_samples, n_points))

    for i in range(n_samples):
        y[i] = np.linspace(0, 2, n_points)
        coeff = (np.random.rand(degree + 1) - 0.5)*2*np.abs(M)
        u[i] = chebval(np.linspace(-1, 1, n_points), coeff)
        Guy[i] = cumulative_trapezoid(u[i], y[i], initial=0)
        
    return y, u, Guy

n_points, n_samples = 201, 100
y, u, Guy = get_data(n_samples, n_points, seed=41)

i = 1
fig, ax = plt.subplots(figsize=(5, 3))
ax.plot(y[i], u[i], 'k', label='u')
ax.plot(y[i], Guy[i], 'r', label='Gu')
ax.legend()
plt.show()
```

```python
class Dataset(torch.utils.data.Dataset):
    def __init__(self, y, u, Guy):
        self.y = np.expand_dims(y, axis=-1)
        self.u = np.expand_dims(u, axis=1)
        self.Guy = np.expand_dims(Guy, axis=-1)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        y = torch.tensor(self.y[idx]).float()
        u = torch.tensor(self.u[idx]).float()
        Guy = torch.tensor(self.Guy[idx]).float()
        return y, u, Guy

n_samples, n_points = 10000, 101
y, u, Guy = get_data(n_samples, n_points, degree=20, M=5)
dataloader = torch.utils.data.DataLoader(Dataset(y, u, Guy), batch_size=1000, shuffle=True)

branch = MLP(101, 75, 75, 4)
trunk = MLP(1, 75, 75, 4)
model = DeepONet(75, 1, branch, trunk).to(device)

loss_fn = nn.MSELoss()
optimizer = optim.AdamW(model.parameters(), lr=1e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.9)

x = torch.linspace(0, 2, 101)
a = 4
dy = torch.sin(a*x)

x_in = x.unsqueeze(0).unsqueeze(-1).to(device)
dy_in = dy.unsqueeze(0).unsqueeze(1).to(device)
out_true = -1/a*(torch.cos(a*x) - 1)
out_true = out_true.to(device)
print(dy_in.shape, x_in.shape)
```

```python
n_epochs = 3000
loss_list, acc_list = [], []

with tqdm(range(1, n_epochs + 1), leave=False) as pbar:
    for epoch in pbar:
        model.train()
        batch_loss, batch_acc = 0, 0

        for i, (y, u, Guy) in enumerate(dataloader):
            y, u, Guy = y.to(device), u.to(device), Guy.to(device)
            Guy_pred = model(y, u)
            loss = loss_fn(Guy_pred, Guy)
            batch_loss += loss.item()

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            model.eval()
            with torch.no_grad():
                out = model(x_in, dy_in)
                acc = loss_fn(out.squeeze(), out_true.squeeze())
                batch_acc += acc.item()

        scheduler.step()
        loss_list.append(batch_acc / (i + 1))
        acc_list.append(batch_acc / (i+1))
        pbar.set_description(f"Epoch[{epoch}/{n_epochs}] (lr: {scheduler.get_last_lr()[0]:.2e}) loss: {loss_list[-1]:.2e}, acc: {acc_list[-1]:.2e}")
```

```python
a = 12

x = torch.linspace(0,2,101)
dy = torch.sin(a*x)
dy_in = dy.unsqueeze(0).unsqueeze(1)
x_in = x.unsqueeze(0).unsqueeze(-1)
out_true = -1/a*(torch.cos(a*x) - 1)
with torch.no_grad():
    out = model.cpu()(x_in,dy_in).squeeze()
print(out.shape)

plt.plot(x, out, label = 'Net')
plt.plot(x, out_true,label = 'Analytic')
plt.legend(loc='upper left')
plt.title(f'Indefinite Integral for sin({a}x)')
# plt.savefig(f'a_{a}.png')
plt.show()
```
