```python
import numpy as np

from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load data
x, y = load_diabetes(return_X_y=True)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

print(x_train_scaled.shape, y_train.shape)
print(x_test_scaled.shape, y_test.shape)

import torch
import torch.nn as nn
import torch.optim as optim

## regression: loss function and metric
mse_loss = nn.MSELoss()     

def r2_score(y_pred, y_true):
    mean_y = torch.mean(y_true)
    ss_tot = torch.sum((y_true - mean_y)**2)
    ss_res = torch.sum((y_true - y_pred)**2)  
    return 1 - (ss_res / ss_tot) 

# Hyperparameters
n_epochs = 50000
learning_rate = 0.1

input_dim = 10
hidden_dim = 100
output_dim = 1

# Data
x = torch.tensor(x_train_scaled).float()
y = torch.tensor(y_train).float().view(-1, 1)
print(x.shape, y.shape)
```

```python
## Method - 1
## Model
torch.manual_seed(42)
w1 = torch.randn(input_dim, hidden_dim).requires_grad_(False)
b1 = torch.zeros(hidden_dim).requires_grad_(False)
w2 = torch.randn(hidden_dim, output_dim).requires_grad_(False)
b2 = torch.zeros(output_dim).requires_grad_(False)

for epoch in range(1, n_epochs + 1):
    # Forward propagation
    z1 = torch.mm(x, w1) + b1
    a1 = torch.sigmoid(z1)
    z2 = torch.mm(a1, w2) + b2
    a2 = z2     ## Identity activation

    y_pred = a2
    loss = mse_loss(y_pred, y)
    score = r2_score(y_pred, y)

    # Backward progapation
    grad_a2 = 2 * (a2 - y) / y.shape[0]
    grad_z2 = grad_a2
    grad_w2 = torch.mm(a1.t(), grad_z2)
    grad_b2 = torch.sum(grad_z2, dim=0)
    
    grad_a1 = torch.mm(grad_z2, w2.t())
    grad_z1 = a1 * (1 - a1) * grad_a1
    grad_w1 = torch.mm(x.t(), grad_z1)
    grad_b1 = torch.sum(grad_z1, dim=0)

    # Update weights and biases
    w1 -= learning_rate * grad_w1
    b1 -= learning_rate * grad_b1
    w2 -= learning_rate * grad_w2
    b2 -= learning_rate * grad_b2

    if epoch % (n_epochs // 10) == 0:
        print(f"[{epoch}/{n_epochs}] loss: {loss.item():.4f} score: {score:.4f}")
```

```python
## Method - 2
## Model
torch.manual_seed(42)
w1 = torch.randn(input_dim, hidden_dim).requires_grad_(True)
b1 = torch.zeros(hidden_dim).requires_grad_(True)
w2 = torch.randn(hidden_dim, output_dim).requires_grad_(True)
b2 = torch.zeros(output_dim).requires_grad_(True)

for epoch in range(1, n_epochs + 1):
    # Forward propagation
    z1 = torch.mm(x, w1) + b1
    a1 = torch.sigmoid(z1)
    z2 = torch.mm(a1, w2) + b2
    a2 = z2     ## Identity activation

    y_pred = a2
    loss = mse_loss(y_pred, y)
    score = r2_score(y_pred, y)

    # Backward progapation
    grads = torch.autograd.grad(loss, [w1, b1, w2, b2], create_graph=True)

    # Update weights and biases
    with torch.no_grad():
        w1 -= learning_rate * grads[0]
        b1 -= learning_rate * grads[1]
        w2 -= learning_rate * grads[2]
        b2 -= learning_rate * grads[3]
        
    if epoch % (n_epochs // 10) == 0:
        print(f"[{epoch}/{n_epochs}] loss: {loss.item():.4f} score: {score:.4f}")
```

```python
## Model - 3
torch.manual_seed(42)
w1 = torch.randn(input_dim, hidden_dim).requires_grad_(True)
b1 = torch.zeros(hidden_dim).requires_grad_(True)
w2 = torch.randn(hidden_dim, output_dim).requires_grad_(True)
b2 = torch.zeros(output_dim).requires_grad_(True)

for epoch in range(1, n_epochs + 1):
    # Forward propagation
    z1 = torch.mm(x, w1) + b1
    a1 = torch.sigmoid(z1)
    z2 = torch.mm(a1, w2) + b2
    a2 = z2     ## Identity activation

    y_pred = a2
    loss = mse_loss(y_pred, y)
    score = r2_score(y_pred, y)

    # Backward progapation
    loss.backward()

    # Update weights and biases
    with torch.no_grad():    
        w1 -= learning_rate * w1.grad
        b1 -= learning_rate * b1.grad
        w2 -= learning_rate * w2.grad
        b2 -= learning_rate * b2.grad

        w1.grad.zero_()
        b1.grad.zero_()
        w2.grad.zero_()
        b2.grad.zero_()

    if epoch % (n_epochs // 10) == 0:
        print(f"[{epoch}/{n_epochs}] loss: {loss.item():.4f} score: {score:.4f}")
```

```python
## Method - 4
## Model
torch.manual_seed(42)
w1 = torch.randn(input_dim, hidden_dim).requires_grad_(True)
b1 = torch.zeros(hidden_dim).requires_grad_(True)
w2 = torch.randn(hidden_dim, output_dim).requires_grad_(True)
b2 = torch.zeros(output_dim).requires_grad_(True)

optimizer = optim.SGD([w1, b1, w2, b2], lr=0.1)

for epoch in range(1, n_epochs + 1):
    # Forward propagation
    lin1 = torch.mm(x, w1) + b1
    act1 = torch.sigmoid(lin1)
    out = torch.mm(act1, w2) + b2
    y_pred = out

    loss = mse_loss(y_pred, y)
    score = r2_score(y_pred, y)

    # Backward progapation
    loss.backward()

    # Update weights and biases
    optimizer.step()
    optimizer.zero_grad()

    if epoch % (n_epochs // 10) == 0:
        print(f"[{epoch}/{n_epochs}] loss: {loss.item():.4f} score: {score:.4f}")
```

```python
## Method - 5
## Model
# model = nn.Sequential(
#     nn.Linear(input_dim, hidden_dim),
#     nn.Sigmoid(),
#     nn.Linear(hidden_dim, output_dim),
# )
# optimizer = optim.Adam(model.parameters(), lr=0.001)

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, output_dim)
        
        ## initialization
        torch.nn.init.normal_(self.linear1.weight)
        torch.nn.init.normal_(self.linear2.weight)

    def forward(self, x):
        x = self.linear1(x)
        x = torch.sigmoid(x)
        x = self.linear2(x)  
        return x

model = MLP(input_dim, hidden_dim, output_dim)
optimizer = optim.SGD(model.parameters(), lr=0.1)

for epoch in range(1, n_epochs + 1):
    # Forward propagation
    y_pred = model(x)
    loss = mse_loss(y_pred, y)
    score = r2_score(y_pred, y)

    # Backward progapation
    loss.backward()

    # Update weights and biases
    optimizer.step()
    optimizer.zero_grad()

    if epoch % (n_epochs // 10) == 0:
        print(f"[{epoch}/{n_epochs}] loss: {loss.item():.4f} score: {score:.4f}")
```
