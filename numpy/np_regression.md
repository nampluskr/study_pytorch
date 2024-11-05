# Regression

## Linear Regression

```python
# Data
def create_data(a, b, n_samples):
    x = np.linspace(0, 10, n_samples)
    y = a * x + b + np.random.normal(0, 2, n_samples)
    return x, y

x, y = create_data(a=2, b=1, n_samples=10001)

# x = np.array([1, 2, 3, 4, 5])
# y = np.array([3, 5, 7, 9, 11])

# Model
a = 0.01
b = 0.01

learning_rate = 0.001
n_epochs = 10000

for epoch in range(1, n_epochs + 1):
    out = a * x + b
    loss = np.mean((out - y)**2)

    grad_out = 2 * (out - y) / len(y)
    grad_a = np.sum(grad_out * x, axis=0)
    grad_b = np.sum(grad_out, axis=0)
    
    a -= learning_rate * grad_a
    b -= learning_rate * grad_b
    
    if epoch % 1000 == 0:
        print(f'Epoch[{epoch}/{n_epochs}] loss: {loss:.4f} a: {a:.4f} b: {b:.4f}')

print(f'Result: a={a:.2f}, b={b:.2f}')
```

## Load Data

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
```

## Machine Learning: scikit-learn

```python
# sklearn
model = LinearRegression()
model.fit(x_train_scaled, y_train)

# Training
y_pred = model.predict(x_train_scaled)
acc = r2_score(y_train, y_pred)
print(f'Train Accuracy: {acc:.4f}')

# Evaluation
y_pred = model.predict(x_test_scaled)
acc = r2_score(y_test, y_pred)
print(f'Test  Accuracy: {acc:.4f}')
```

## Deep Learning: numpy

```python
#from scipy.special import expit as sigmoid, logsumexp

def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -20, 20)))

# def sigmoid(x):
    # return 1 / (1 + np.exp(-x))
    # return np.log1p(np.exp(x))
#     x = np.maximum(-np.max(x), -10) 
#     # x = np.clip(-np.max(x), -10, None)
#     return 1 / (1 + np.exp(x))

def mse_loss(y_pred, y_true):
    return np.mean((y_true - y_pred)**2)

def r2_score(y_pred, y_true):
    mean_y_true = np.mean(y_true)
    ss_tot = np.sum((y_true - mean_y_true) ** 2)
    ss_res = np.sum((y_true - y_pred) ** 2)
    return 1 - (ss_res / ss_tot)

# numpy
n_epochs = 10000
learning_rate = 0.1

input_dim = 10
hidden_dim = 100
output_dim = 1

## Model
w1 = np.random.randn(input_dim, hidden_dim)
b1 = np.zeros(hidden_dim)
w2 = np.random.randn(hidden_dim, output_dim)
b2 = np.zeros(output_dim)

x, y = x_train_scaled, y_train.reshape(-1, 1)
print(x.shape, y.shape)

for epoch in range(1, n_epochs + 1):
    # Forward propagation
    lin1 = np.dot(x, w1) + b1
    act1 = sigmoid(lin1)
    out = np.dot(act1, w2) + b2
    y_pred = out

    loss = mse_loss(y_pred, y)
    score = r2_score(y_pred, y)

    # # Backward progapation
    grad_out = 2 * (out - y) / y.shape[0]
    grad_w2 = np.dot(act1.T, grad_out)
    grad_b2 = np.sum(grad_out, axis=0)

    grad_act1 = np.dot(grad_out, w2.T)
    grad_lin1 = act1 * (1 - act1) * grad_act1
    grad_w1 = np.dot(x.T, grad_lin1)
    grad_b1 = np.sum(grad_lin1, axis=0)

    # # Update weights and biases
    w2 -= learning_rate * grad_w2
    b2 -= learning_rate * grad_b2
    w1 -= learning_rate * grad_w1
    b1 -= learning_rate * grad_b1

    if epoch % 1000 == 0:
        print(f"[{epoch}/{n_epochs}] loss: {loss.item():.4f} score: {score:.4f}")
```
