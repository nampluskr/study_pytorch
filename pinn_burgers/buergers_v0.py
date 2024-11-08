# https://adarshgouda.github.io/html_pages/Burgers_FDM_PINN.html
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


def plot3D(x, t, y):
    x_plot =x.squeeze(1)
    t_plot =t.squeeze(1)
    X,T= torch.meshgrid(x_plot,t_plot,indexing='ij')
    u_xt = y

    fig = plt.figure()
    ax=fig.subplots(1,1)
    cp = ax.contourf(T,X,u_xt,20,cmap=cm.rainbow) #)levels = np.linspace(-1.0,1.0,12))
    fig.colorbar(cp) # Add a colorbar to a plot
    ax.set_title('u')
    ax.set_xlabel('t')
    ax.set_ylabel('x')
    plt.show()

    ax = plt.axes(projection='3d')
    surf = ax.plot_surface(T.numpy(), X.numpy(), u_xt.numpy(),cmap=cm.rainbow, antialiased=False)
    ax.set_xlabel('t')
    ax.set_ylabel('x')
    ax.set_zlabel('u')
    #ax.set_zlim3d(-1, 1)
    plt.show()


def u_fem(x, t):
    un = torch.ones(total_points_x)
    rec = torch.zeros([total_points_x, total_points_t])
    for j in tqdm(range(total_points_t)):
        un = u.clone()

        for i in range(1,total_points_x-1):
            rec[i,j] = u[i]
            u[i] = un[i] - un[i] * dt/dx * (un[i]-un[i-1])  + viscosity * (dt/dx**2) * (un[i+1]- 2*un[i] + un[i-1])
            if np.isnan(u[i]):
              print(i, j, u[i-1])
              break
    return u, rec


class u_NN(nn.Module):

    def __init__(self, layers_list):
        super().__init__()

        self.depth = len(layers_list)
        self.loss_function = nn.MSELoss(reduction="mean")
        self.activation = nn.Tanh() #This is important, ReLU wont work
        self.linears = nn.ModuleList([nn.Linear(layers_list[i],layers_list[i+1]) for i in range(self.depth-1)])

        for i in range(self.depth-1):
          nn.init.xavier_normal_(self.linears[i].weight.data, gain=1.0) #xavier normalization of weights
          nn.init.zeros_(self.linears[i].bias.data) #all biases set to zero

    def Convert(self, x): #helper function
        if torch.is_tensor(x) !=True:
            x = torch.from_numpy(x)
        return x.float().to(device)

    def forward(self, x):
        a = self.Convert(x)
        for i in range(self.depth-2):
            z = self.linears[i](a)
            a = self.activation(z)
        a = self.linears[-1](a)
        return a

    def loss_bc(self, x_bc, u_bc):
        l_bc = self.loss_function(self.forward(self.Convert(x_bc)), self.Convert(u_bc)) #L2 loss
        return l_bc

    def loss_ic(self, x_ic, u_ic):
        l_ic = self.loss_function(self.forward(self.Convert(x_ic)), self.Convert(u_ic)) #L2 loss
        return l_ic

    def loss_pde(self, x_pde):
        x_pde = self.Convert(x_pde)
        x_pde_clone = x_pde.clone() ##VERY IMPORTANT
        x_pde_clone.requires_grad = True #enable Auto Differentiation

        NN = self.forward(x_pde_clone) #Generates predictions from u_NN
        NNx_NNt = torch.autograd.grad(NN, x_pde_clone,self.Convert(torch.ones([x_pde_clone.shape[0],1])),retain_graph=True, create_graph=True)[0] #Jacobian of dx and dt
        NNxx_NNtt = torch.autograd.grad(NNx_NNt,x_pde_clone, self.Convert(torch.ones(x_pde_clone.shape)), create_graph=True)[0] #Jacobian of dx2, dt2

        NNxx = NNxx_NNtt[:,[0]]     # Extract only dx2 terms
        NNt = NNx_NNt[:,[1]]        # Extract only dt terms
        NNx = NNx_NNt[:,[0]]        # Extract only dx terms

        # {(du/dt) = viscosity * (d2u/dx2)} is the pde and the NN residue will be 
        # {du_NN/dt - viscosity*(d2u_NN)/dx2} which is == {NNt - viscosity*NNxx}
        residue = NNt + self.forward(x_pde_clone)*(NNx) - (viscosity)*NNxx
        zeros = self.Convert(torch.zeros(residue.shape[0],1))
        l_pde = self.loss_function(residue, zeros) #L2 Loss
        return l_pde

    def total_loss(self, x_ic, u_ic, x_bc, u_bc, x_pde): #Combine both loss
        l_bc = self.loss_bc(x_bc, u_bc)
        l_ic = self.loss_ic(x_ic, u_ic)
        l_pde = self.loss_pde(x_pde)
        return l_bc + l_pde + l_ic #this HAS to be a sca


if __name__ == "__main__":

    torch.set_default_dtype(torch.float)
    torch.manual_seed(1234)
    np.random.seed(1234)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cgpu')

    x_min, x_max = 0, 2
    t_min, t_max = 0, 0.48
    viscosity = 0.01 / np.pi

    total_points_x = 1001
    total_points_t = 1000

    dx = (x_max - x_min)/(total_points_x - 1)
    dt = (t_max - t_min)/(total_points_t)

    x = torch.linspace(x_min, x_max, total_points_x).view(-1, 1)
    t = torch.linspace(t_min, t_max, total_points_t).view(-1, 1)
    u = torch.from_numpy(np.sin(np.pi * x.numpy()))

    # u_final, u_fem_2D = u_fem(x, t)

    X, T = torch.meshgrid(x.squeeze(1), t.squeeze(1), indexing='ij')

    left_X = torch.hstack((X[:,0][:, None], T[:,0][:, None]))
    left_U = torch.sin(np.pi*left_X[:,0]).unsqueeze(1) 

    bottom_X = torch.hstack((X[0,:][:, None],T[0,:][:, None]))
    bottom_U = torch.zeros(bottom_X.shape[0],1)

    top_X = torch.hstack((X[-1,:][:, None],T[-1,:][:, None]))
    top_U = torch.zeros(top_X.shape[0],1)

    X_bc = torch.vstack([bottom_X, top_X])
    U_bc = torch.vstack([bottom_U, top_U])

    N_ic = 1000
    N_bc = 1000
    N_pde = 30000

    idx = np.random.choice(X_bc.shape[0], N_bc, replace=False)
    X_bc_samples = X_bc[idx, :]
    U_bc_samples = U_bc[idx, :]

    idx = np.random.choice(left_X.shape[0], N_ic, replace=False)
    X_ic_samples = left_X[idx, :]
    U_ic_samples = left_U[idx, :]

    x_test = torch.hstack((X.transpose(1,0).flatten()[:, None], T.transpose(1,0).flatten()[:, None]))
    # u_test = u_fem_2D.transpose(1,0).flatten()[:, None]
    lb, ub = x_test[0], x_test[-1]
    lhs_samples = lhs(2, N_pde) 
    X_train_lhs = lb + (ub - lb)*lhs_samples
    X_train_final = torch.vstack((X_train_lhs, X_ic_samples, X_bc_samples))

    # EPOCHS = 100000
    EPOCHS = 10000
    initial_lr = 0.001
    # layers_list = [2, 32, 128, 16, 128, 32, 1]
    layers_list = [2, 50, 50, 1]

    PINN = u_NN(layers_list).to(device)
    optimizer = torch.optim.Adam(PINN.parameters(), lr=initial_lr,amsgrad=False)
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

    for i in tqdm(range(EPOCHS), ascii=True, ncols=150):
        if i==0:
            print("Epoch \t Learning_Rate \t IC_Loss \t BC_Loss \t PDE_Loss \t Total_Loss \t Test_Loss")

        l_ic = PINN.loss_ic(X_ic_samples,U_ic_samples)
        l_bc = PINN.loss_bc(X_bc_samples,U_bc_samples)
        l_pde = PINN.loss_pde(X_train_final)
        loss = PINN.total_loss(X_ic_samples,U_ic_samples,X_bc_samples,U_bc_samples, X_train_final)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % 100 == 0: #print losses and step the exponential learning rate.
            with torch.no_grad():
                # test_loss = PINN.loss_bc(x_test, u_test) # Here we are using loss_bc method as a helper function to calculate L2 loss

                Epoch.append(i)
                Learning_Rate.append(scheduler.get_last_lr()[0])
                IC_Loss.append(l_ic.detach().cpu().numpy())
                BC_Loss.append(l_bc.detach().cpu().numpy())
                PDE_Loss.append(l_pde.detach().cpu().numpy())
                Total_Loss.append(loss.detach().cpu().numpy())
                # Test_Loss.append(test_loss.detach().cpu().numpy())

                if i%1000 ==0:
                    print(i,'\t', format(scheduler.get_last_lr()[0],".4E"),
                            '\t', format(l_ic.detach().cpu().numpy(),".4E"),
                            '\t', format(l_bc.detach().cpu().numpy(),".4E"),
                            '\t', format(l_pde.detach().cpu().numpy(),".4E"),
                            '\t',format(loss.detach().cpu().numpy(),".4E"),
                            # '\t',format(test_loss.detach().cpu().numpy(),".4E")
                            )

            scheduler.step()
