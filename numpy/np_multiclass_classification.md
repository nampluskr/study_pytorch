# Multiclass Classification

## Load Data

```python
import numpy as np

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Load data
x, y = load_iris(return_X_y=True)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.4, random_state=42)

scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

print(x_train_scaled.shape, y_train.shape)
print(x_test_scaled.shape, y_test.shape)
```

## Machine Learning: scikit-learn

```python
# sklearn
model = SVC()
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
# from scipy.special import expit as sigmoid, logsumexp

def one_hot(y, n_classes):
    return np.eye(n_classes)[y]

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def softmax(x):
    e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return e_x / np.sum(e_x, axis=1, keepdims=True)

# def softmax(x):
#     logsum = logsumexp(x, 1).reshape(-1, 1) if x.ndim == 2 else logsumexp(x)
#     return np.exp(x - logsum)

def cross_entropy(y_pred, y_true):
    batch_size = y_pred.shape[0] if y_pred.ndim == 2 else 1
    return -np.sum(y_true*np.log(y_pred + 1.0E-8))/batch_size

def accuracy(y_pred, y_true):
    return (y_pred.argmax(1) == y_true.argmax(1)).mean()

# numpy
n_epochs = 100
learning_rate = 0.1

input_dim = 4
hidden_dim = 100
output_dim = 3

## Model
w1 = np.random.randn(input_dim, hidden_dim)
b1 = np.zeros(hidden_dim)
w2 = np.random.randn(hidden_dim, output_dim)
b2 = np.zeros(output_dim)

x, y = x_train_scaled, one_hot(y_train, n_classes=3)
print(x.shape, y.shape)

for epoch in range(1, n_epochs + 1):
    # Forward propagation
    lin1 = np.dot(x, w1) + b1
    act1 = sigmoid(lin1)
    out = np.dot(act1, w2) + b2
    y_pred = softmax(out)

    loss = cross_entropy(y_pred, y)
    score = accuracy(y_pred, y)

    # Backward progapation
    grad_out = (out - y) / y.shape[0]
    grad_w2 = np.dot(act1.T, grad_out)
    grad_b2 = np.sum(grad_out, axis=0)

    grad_act1 = np.dot(grad_out, w2.T)
    grad_lin1 = act1 * (1 - act1) * grad_act1
    grad_w1 = np.dot(x.T, grad_lin1)
    grad_b1 = np.sum(grad_lin1, axis=0)

    # Update model parameters
    w2 -= learning_rate * grad_w2
    b2 -= learning_rate * grad_b2
    w1 -= learning_rate * grad_w1
    b1 -= learning_rate * grad_b1
    
    if epoch % 10 == 0:
        print(f"[{epoch}/{n_epochs}] loss: {loss.item():.4f} score: {score:.4f}")
```
