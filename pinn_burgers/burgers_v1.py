# https://adarshgouda.github.io/html_pages/Burgers_FDM_PIU.html
import torch
import torch.autograd as autograd
from torch import Tensor
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter

import numpy as np
import time
import pandas as pd
import matplotlib.pyplot as plt
import time, sys
from tqdm.auto import tqdm
from pyDOE import lhs


def to_torch(x: np.ndarray, device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
    if x.ndim == 1:
        x = x.reshape(-1, 1)
    return torch.tensor(x).float().to(device)


def to_numpy(x: torch.Tensor):
    if x.ndim == 2 and x.shape[-1] == 1:
        x = x.reshape(-1)
    return x.detach().cpu().numpy() if type(x) == np.ndarray else x


def plot2D(x, t, u):
    X, T = np.meshgrid(to_numpy(x), to_numpy(t), indexing='ij')
    u = to_numpy(u)

    fig, ax = plt.subplots(figsize=(6, 3))
    cp = ax.contourf(T, X, u, 20, cmap=cm.rainbow) #)levels = np.linspace(-1.0,1.0,12))
    fig.colorbar(cp)
    ax.set_title('u')
    ax.set_xlabel('t')
    ax.set_ylabel('x')
    plt.show()

def plot3D(x, t, u):
    X, T = np.meshgrid(to_numpy(x), to_numpy(t), indexing='ij')
    u = to_numpy(u)

    fig, ax = plt.subplots(figsize=(4, 4), subplot_kw={"projection": "3d"})
    surf = ax.plot_surface(T, X, u, cmap=cm.rainbow, antialiased=False)
    ax.set_xlabel('t')
    ax.set_ylabel('x')
    ax.set_zlabel('u')
    #ax.set_zlim3d(-1, 1)
    plt.show()


def u_fem(x, t):
    un = torch.ones(x_size)
    res = torch.zeros([x_size, t_size])
    for j in tqdm(range(t_size)):
        un = u.clone()

        for i in range(1, x_size - 1):
            res[i,j] = u[i]
            u[i] = un[i] - un[i] * dt/dx * (un[i]-un[i-1])  + mu * (dt/dx**2) * (un[i+1]- 2*un[i] + un[i-1])
            if np.isnan(u[i]):
                print(i, j, u[i-1])
                break
    return u, res


class PINN(nn.Module):

    def __init__(self, layers_dim):
        super().__init__()

        self.depth = len(layers_dim)
        self.mse = nn.MSELoss()
        self.activation = nn.Tanh()
        self.linears = nn.ModuleList([nn.Linear(layers_dim[i], layers_dim[i + 1]) for i in range(self.depth - 1)])

        for i in range(self.depth-1):
            nn.init.xavier_normal_(self.linears[i].weight.data, gain=1.0) #xavier normalization of weights
            nn.init.zeros_(self.linears[i].bias.data) #all biases set to zero

    def Convert(self, x):
        if not torch.is_tensor(x):
            x = torch.from_numpy(x)
        return x.float().to(device)

    def forward(self, x):
        a = self.Convert(x)
        for i in range(self.depth - 2):
            z = self.linears[i](a)
            a = self.activation(z)
        a = self.linears[-1](a)
        return a

    def loss_bc(self, X_bc, U_bc):
        return self.mse(self.forward(self.Convert(X_bc)), self.Convert(U_bc))

    def loss_ic(self, X_ic, U_ic):
        return self.mse(self.forward(self.Convert(X_ic)), self.Convert(U_ic))

    def loss_pde(self, X_input):
        X = self.Convert(X_input).clone() # VERY IMPORTANT !!!
        X.requires_grad = True

        U = self.forward(X)
        U_X = torch.autograd.grad(U, X, torch.ones((len(X), 1)).to(device), retain_graph=True, create_graph=True)[0]
        U_XX = torch.autograd.grad(U_X, X, torch.ones((len(X), 2)).to(device), create_graph=True)[0]

        U_x = U_X[:, [0]]
        U_t = U_X[:, [1]]
        U_xx = U_XX[:, [0]]

        residue = U_t + U * U_x - mu * U_xx
        zeros = self.Convert(torch.zeros(len(X), 1))
        return self.mse(residue, zeros)

    def total_loss(self, X_ic, U_ic, X_bc, U_bc, X_in):
        loss_bc = self.loss_bc(X_bc, U_bc)
        loss_ic = self.loss_ic(X_ic, U_ic)
        loss_pde = self.loss_pde(X_in)
        return loss_bc + loss_pde + loss_ic


def concat(x1, x2, dim=1):
    if x1.ndim == 1:
        x1 = x1.reshape(-1, 1)
    if x2.ndim == 1:
        x2 = x2.reshape(-1, 1)
    return np.concatenate([x1, x2], axis=dim)

if __name__ == "__main__":

    torch.set_default_dtype(torch.float)
    torch.manual_seed(1234)
    np.random.seed(1234)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cgpu')

    x_min, x_max, x_size = 0, 2, 1001
    t_min, t_max, t_size = 0, 0.48, 1001
    mu = 0.01 / np.pi

    x, dx = np.linspace(x_min, x_max, x_size, retstep=True)
    t, dt = np.linspace(t_min, t_max, t_size, retstep=True)
    X, T = np.meshgrid(x, t, indexing='ij')

    # x, t = x.reshape(-1, 1), t.reshape(-1, 1)
    # u = np.sin(np.pi * x)
    # u_final, u_fem_2D = u_fem(x, t)

    # left_X = np.hstack((X[:, 0].reshape(-1, 1), T[:, 0].reshape(-1, 1)))
    left_X = concat(X[:, 0], T[:, 0])
    left_U = np.sin(np.pi * X[:, 0]).reshape(-1 , 1)

    # bottom_X = np.hstack((X[0, :].reshape(-1, 1),T[0, :].reshape(-1, 1)))
    bottom_X = concat(X[0, :], T[0, :])
    bottom_U = np.zeros((bottom_X.shape[0], 1))

    # top_X = np.hstack((X[-1, :].reshape(-1, 1),T[-1, :].reshape(-1, 1)))
    top_X = concat(X[-1, :], T[-1, :])
    top_U = np.zeros((top_X.shape[0], 1))

    # X_bc = np.vstack([bottom_X, top_X])
    # U_bc = np.vstack([bottom_U, top_U])

    X_bc = concat(bottom_X, top_X, dim=0)
    U_bc = concat(bottom_U, top_U, dim=0)

    N_ic = 1000
    N_bc = 1000
    N_pde = 30000

    idx = np.random.choice(x_size, N_ic, replace=False)
    X_ic_samples = left_X[idx, :]
    U_ic_samples = left_U[idx, :]

    idx = np.random.choice(t_size, N_bc, replace=False)
    X_bc_samples = X_bc[idx, :]
    U_bc_samples = U_bc[idx, :]

    X_test = np.hstack((X.T.flatten().reshape(-1, 1), T.T.flatten().reshape(-1, 1)))
    # u_test = u_fem_2D.transpose(1,0).flatten().reshape(-1, 1)
    lb, ub = X_test[0], X_test[-1]
    lhs_samples = lhs(2, N_pde) 
    X_train_lhs = lb + (ub - lb)*lhs_samples
    X_train_final = np.vstack((X_train_lhs, X_ic_samples, X_bc_samples))

    # n_epochs = 100000
    n_epochs = 1000
    learning_rate = 0.001
    # layers_dim = [2, 32, 128, 16, 128, 32, 1]
    layers_dim = [2, 50, 50, 1]

    model = PINN(layers_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate,amsgrad=False)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.985)
    # history = pd.DataFrame(columns=["Epochs", "Learning_Rate", "IC_Loss","BC_Loss", "PDE_Loss", "Total_Loss", "Test_Loss"])
    history = pd.DataFrame(columns=["Epochs", "Learning_Rate", "IC_Loss","BC_Loss", "PDE_Loss", "Total_Loss"])

    Epoch = []
    Learning_Rate = []
    IC_Loss = []
    BC_Loss = []
    PDE_Loss = []
    Total_Loss = []
    # Test_Loss = []

    print("Epoch \t Learning_Rate \t IC_Loss \t BC_Loss \t PDE_Loss \t Total_Loss \t Test_Loss")
    for i in tqdm(range(n_epochs), ascii=True, ncols=150):
    
        loss_ic = model.loss_ic(X_ic_samples, U_ic_samples)
        loss_bc = model.loss_bc(X_bc_samples, U_bc_samples)
        loss_pde = model.loss_pde(X_train_final)
        loss = model.total_loss(X_ic_samples, U_ic_samples, X_bc_samples, U_bc_samples, X_train_final)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % 100 == 0:
            with torch.no_grad():
                # test_loss = model.loss_bc(X_test, u_test) # Here we are using loss_bc method as a helper function to calculate L2 loss

                Epoch.append(i)
                Learning_Rate.append(scheduler.get_last_lr()[0])
                IC_Loss.append(loss_ic.item())
                BC_Loss.append(loss_bc.item())
                PDE_Loss.append(loss_pde.item())
                Total_Loss.append(loss.item())
                # Test_Loss.append(test_loss.item())

                if i%1000 ==0:
                    print(i,'\t', format(scheduler.get_last_lr()[0],".4E"),
                            '\t', format(loss_ic.item(),".4E"),
                            '\t', format(loss_bc.item(),".4E"),
                            '\t', format(loss_pde.item(),".4E"),
                            '\t',format(loss.item(),".4E"),
                            # '\t',format(test_loss.item(),".4E")
                            )

            scheduler.step()
