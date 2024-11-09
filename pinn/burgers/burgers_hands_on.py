## https://medium.com/@hridaym.211me129/hands-on-physics-informed-neural-networks-2e9bca75ab59

import scipy
from scipy.interpolate import griddata
import numpy as np
import torch
import torch.nn as nn 
import matplotlib.pyplot as plt 
from tqdm.auto import tqdm


class Network(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(2, 16)
        self.fc2 = nn.Linear(16, 32)
        self.fc3 = nn.Linear(32, 1)
    
    def forward(self,x):
        x = nn.functional.relu(self.fc1(x))
        x = nn.functional.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class PINN:
    def __init__(self, X, u, lb, ub, physics):

        self.lb = torch.tensor(lb).float()
        self.ub = torch.tensor(ub).float()
        self.physics = physics

        self.x = torch.tensor(X[:, 0].view(-1, 1), requires_grad=True).float().to(device)
        self.t = torch.tensor(X[:, 1].view(-1, 1), requires_grad=True).float().to(device)
        self.u = torch.tensor(u).float().to(device)

        self.model = Network().to(device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)

    def forward(self, x, t):
        X = torch.cat([x,t], 1)
        return self.model(X)

    def residual(self, x, t):
        u = self.forward(x, t)
        u_t = torch.autograd.grad(u, t, grad_outputs=torch.ones_like(u),  create_graph=True)[0]
        u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]
        u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x), create_graph=True)[0]
        return u_t + u*u_x - (0.01 / np.pi)*u_xx

    def train(self, epochs):
        lossTracker = []
        self.model.train()
        with tqdm(range(1, epochs + 1), ascii=True) as pbar:
            for idx in pbar:
                u_pred = self.forward(self.x, self.t)
                residual_pred = self.residual(self.x, self.t)
                loss = torch.mean((self.u - u_pred)**2)

                if self.physics == True:
                    loss += torch.mean(residual_pred**2)

                lossTracker.append(loss.item())

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                if idx % 1000 == 0:
                    print(f"The loss at epoch {idx} is {loss.item():.2e}")

            return lossTracker

    def predict(self):
        self.model.eval()
        u = self.forward(self.x, self.t)
        res = self.residual(self.x, self.t)
        return u.detach().cpu().numpy(), res.detach().cpu().numpy()


if __name__ == "__main__":
  
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    data = scipy.io.loadmat('burgers_shock.mat')
    x = data['x'].flatten()[:, None]
    t = data['t'].flatten()[:, None]
    usol = np.real(data['usol']).T
    X, T = np.meshgrid(x, t)
    train = torch.concat([torch.Tensor(X.flatten()[:, None]), torch.Tensor(T.flatten()[:, None])], 1)
    X_min = train.min(0)
    X_max = train.max(0)
    
    # data = scipy.io.loadmat('burgers_shock.mat')
    # x = data['x'].flatten()[:, None]        # (256, 1)
    # t = data['t'].flatten()[:, None]        # (100, 1)
    
    # usol = np.real(data['usol'])            # (100, 256)
    # X, T = np.meshgrid(x, t, indexing="ij") # (100, 256)
    
    # X_train = np.concatenate([X.flatten()[:, None], T.flatten()[:, None]], 1)
    # X_min = X_train.min(axis=0)             # [-1, 0]
    # X_max = X_train.max(axis=0)             # [1, 0.99]
    
    idx = np.random.choice(train.shape[0], 2000, replace=False)
    X_u_train = train[idx, :]
    u_train = usol.flatten()[:, None][idx,:]
    model = PINN(X_u_train, u_train, X_min[0], X_max[0], True) # Keep False for Vanilla
    pinn = model.train(10000)
