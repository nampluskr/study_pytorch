```python
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim

x_np = np.linspace(0, 2, 101)
y_np = 2 * x_np + 1 + np.random.randn(x_np.size) * 0.1

# fig, ax = plt.subplots(figsize=(5, 3))
# ax.plot(x, y, 'kx')
# ax.plot(x, 2*x + 1, 'r')
# plt.show()
```

```python
# Model: y = a * x + b
a = 0.01    # weight
b = 0.0     # bias

# Training data
x, y = x_np, y_np

# Hyperparameters
n_epochs = 1000
learning_rate = 0.01

# Training
for epoch in range(1, n_epochs + 1):
    # Forward
    out = a * x + b
    loss = np.mean((out - y)**2)
    
    # Backward
    grad_out = 2 * (out - y) / len(y)
    grad_a = np.sum(grad_out * x, axis=0)
    grad_b = np.sum(grad_out * 1, axis=0)
    
    # Update parameters
    a -= learning_rate * grad_a
    b -= learning_rate * grad_b
        
    if epoch % (n_epochs // 10) == 0:
        print(f"Epoch[{epoch}/{n_epochs}] loss: {loss:.4f} a:{a:.4f} b:{b:.4f}")
```

```python
# Model: y = a * x + b
a = torch.tensor(0.01).float().requires_grad_(True)    # weight
b = torch.tensor(0.0).float().requires_grad_(True)     # bias

# Training data
x = torch.tensor(x_np).float()
y = torch.tensor(y_np).float()

# Hyperparameters
n_epochs = 1000
learning_rate = 0.01

# Training loop
for epoch in range(1, n_epochs + 1):
    # Forward
    out = a * x + b
    loss = torch.mean((out - y)**2)
    
    # Backward
    grads = torch.autograd.grad(loss, [a, b], create_graph=True)
    
    # Update parameters
    with torch.no_grad():
        a -= learning_rate * grads[0]
        b -= learning_rate * grads[1]
        
    if epoch % (n_epochs // 10) == 0:
        print(f"Epoch[{epoch}/{n_epochs}] loss: {loss.item():.4f} a: {a.item():.4f} b: {b.item():.4f}")
```

```python
# Model: y = a * x + b
a = torch.tensor(0.01).float().requires_grad_(True)    # weight
b = torch.tensor(0.0).float().requires_grad_(True)     # bias

# Training data
x = torch.tensor(x_np).float()
y = torch.tensor(y_np).float()

# Hyperparameters
n_epochs = 1000
learning_rate = 0.01

# Training loop
for epoch in range(1, n_epochs + 1):
    # Forward
    out = a * x + b
    loss = torch.mean((out - y)**2)
    
    # Backward
    loss.backward()
    
    # Update parameters
    with torch.no_grad():
        a -= learning_rate * a.grad
        b -= learning_rate * b.grad
        
        a.grad.zero_()
        b.grad.zero_()
        
    if epoch % (n_epochs // 10) == 0:
        print(f"Epoch[{epoch}/{n_epochs}] loss: {loss.item():.4f} a: {a.item():.4f} b: {b.item():.4f}")
```

```python
# Model: y = a * x + b
a = torch.tensor(0.01).float().requires_grad_(True)    # weight
b = torch.tensor(0.0).float().requires_grad_(True)     # bias

# Training data
x = torch.tensor(x_np).float()
y = torch.tensor(y_np).float()

# Hyperparameters
n_epochs = 1000
learning_rate = 0.01

optimizer = optim.SGD([a, b], lr=learning_rate)

# Training loop
for epoch in range(1, n_epochs + 1):
    # Forward
    out = a * x + b
    loss = torch.mean((out - y)**2)
    
    # Backward
    loss.backward()
    
    # Update parameters
    optimizer.step()
    optimizer.zero_grad()
        
    if epoch % (n_epochs // 10) == 0:
        print(f"Epoch[{epoch}/{n_epochs}] loss: {loss.item():.4f} a: {a.item():.4f} b: {b.item():.4f}")
```
