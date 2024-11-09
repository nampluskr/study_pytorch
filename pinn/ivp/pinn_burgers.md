```python
def loss_fn(model, x, t):
    x = to_torch(x).requires_grad_(True)
    t = to_torch(t).requires_grad_(True)
    
    u = model(x)
    u_t = gradient(u, t)
    u_x = gradient(u, x)
    u_xx = gradient(u_x, x)
    eqn = u_t + u * u_t - (0.01/np.pi) * u_xx 
    return torch.mean(eqn**2)

x_size, t_size = 1001, 501

x, dx = np.linspace(0, 2, x_size, retstep=True)
t, dt = np.linspace(0, 0.48, t_size, retstep=True)
X, T = np.meshgrid(x, t, indexing='ij')
inputs = np.hstack([X[:, 0][:, None], T[:, 0][:, None]])

# x, t = to_torch(x), to_torch(t)
# x.shape, t.shape

X.shape, T.shape, inputs.shape

# https://adarshgouda.github.io/html_pages/Burgers_FDM_PINN.html

# Initial condtion at t = 0
left_X = np.hstack((X[:, 0].reshape(-1, 1), T[:, 0].reshape(-1, 1)))
left_U = np.sin(np.pi*left_X[:, 0]).reshape(-1, 1)

# Boundary condition
bottom_X = np.hstack((X[0, :].reshape(-1, 1), T[0, :].reshape(-1, 1)))
bottom_U = np.zeros((bottom_X.shape[0], 1))

# Boundary condition
top_X = np.hstack((X[-1, :].reshape(-1, 1), T[-1,:].reshape(-1, 1)))
top_U = np.zeros((top_X.shape[0], 1))

print(left_X.shape, left_U.shape)
print(bottom_X.shape, bottom_U.shape)
print(top_X.shape, top_U.shape)
```
