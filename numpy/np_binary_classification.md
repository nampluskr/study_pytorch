# Binary Classification

## Load data

```python
import numpy as np

from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load data
x, y = load_breast_cancer(return_X_y=True)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

print(x_train_scaled.shape, y_train.shape)
print(x_test_scaled.shape, y_test.shape)
```

## Machine Learning: scikit-learn

```pydthon
# sklearn
model = LogisticRegression()
model.fit(x_train_scaled, y_train)

# Training
y_pred = model.predict(x_train_scaled)
acc = accuracy_score(y_train, y_pred)
print(f'Train Accuracy: {acc:.4f}')

# Evaluation
y_pred = model.predict(x_test_scaled)
acc = accuracy_score(y_test, y_pred)
print(f'Test  Accuracy: {acc:.4f}')
```

## Deep Learning: numpy

```python
# from scipy.special import expit as sigmoid

def sigmoid(x):
    # return 1 / (1 + np.exp(-x))
# def softplus(x):
    return np.log1p(np.exp(x)) 

def bce_loss(y_pred, y_true):
    eps = 1e-8
    return -np.mean(y_true*np.log(y_pred + eps) + (1 - y_true)*np.log(1 - y_pred + eps))

def binary_accuracy(y_pred, y_true):
    y_pred = (y_pred > 0.5).astype(int)
    return np.mean(y_pred == y_true)

# numpy
n_epochs = 100
learning_rate = 0.1

input_dim = 30
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
    lin2 = np.dot(act1, w2) + b2
    y_pred = out = sigmoid(lin2)

    loss = bce_loss(y_pred, y)
    score = binary_accuracy(y_pred, y)

    # Backward propagation
    grad_out = (out - y) / out / (1 - out) / y.shape[0]
    grad_lin2 = out * (1 - out) * grad_out
    grad_w2 = np.dot(act1.T, grad_lin2) 
    grad_b2 = np.sum(grad_lin2, axis=0)

    grad_act1 = np.dot(grad_lin2, w2.T)
    grad_lin1 = act1 * (1 - act1) * grad_act1
    grad_w1 = np.dot(x.T, grad_lin1) 
    grad_b1 = np.sum(grad_lin1, axis=0)

    # Update weights and biases
    w1 -= learning_rate * grad_w1
    b1 -= learning_rate * grad_b2
    w2 -= learning_rate * grad_w2
    b2 -= learning_rate * grad_b2

    if epoch % 10 == 0:
        print(f'Epoch [{epoch}/{n_epochs}] loss: {loss.item():.4f} score: {score.item():.4f}')        
```
