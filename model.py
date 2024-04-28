import torch
from torch import nn
from torch.autograd import grad
from torch import optim
import numpy as np

def sine_activation(x):
    return torch.sin(x)

class PINN(nn.Module):
    def __init__(self, units, activation=sine_activation):
        super(PINN, self).__init__()
        self.hidden_layer1 = nn.Linear(3, units)
        self.hidden_layer2 = nn.Linear(units, units)
        self.hidden_layer3 = nn.Linear(units, units)
        self.hidden_layer4 = nn.Linear(units, units)
        self.hidden_layer5 = nn.Linear(units, 3)
        #self.hidden_layer6 = nn.Linear(units, units)
        #self.hidden_layer7 = nn.Linear(units, units)        
        #self.hidden_layer8 = nn.Linear(units, 3)
        self.activation = activation

    def forward(self, inputs):
        x = inputs
        h1 = self.hidden_layer1(x)
        h1 = self.activation(h1)
        h2 = self.hidden_layer2(h1)
        h2 = self.activation(h2)
        h3 = self.hidden_layer3(h2+h1)
        h3 = self.activation(h3)
        h4 = self.hidden_layer4(h3+h2+h1)
        h4 = self.activation(h4)
        h5 = self.hidden_layer5(h4+h3+h2+h1)
        #h5 = self.activation(h5)
        #h6 = self.hidden_layer6(h5)
        #h6 = self.activation(h6)
        #h7 = self.hidden_layer7(h6)
        #h7 = self.activation(h7)
        #h8 = self.hidden_layer8(h7)
        return h5

def gradients(u, x):
    return grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True,  only_inputs=True, allow_unused=True)[0]

class PINN_Loss(nn.Module):
    #初始化神经网络输入，定义输入参数
    def __init__(self, N_f, L, device, addBC):
        super(PINN_Loss, self).__init__() #继承tf.keras.Model的功能
        self.N_f = N_f
        self.L = L
        self.device = device
        if(addBC==0):
            self.addBC = False
        if(addBC==1):
            self.addBC = True

    def forward(self, data, pred, labels, model):
        device = self.device
        train_x = data[:,0].view(-1,1).requires_grad_(True)
        train_y = data[:,1].view(-1,1).requires_grad_(True)
        train_z = data[:,2].view(-1,1).requires_grad_(True)
        B = model(torch.cat((train_x, train_y, train_z), axis=1))
        B_x = B[:,0].requires_grad_(True)
        B_y = B[:,1].requires_grad_(True)
        B_z = B[:,2].requires_grad_(True)
        dx = gradients(B_x, train_x)
        dy = gradients(B_y, train_y)
        dz = gradients(B_z, train_z)
        dzy = gradients(B_z, train_y)
        dzx = gradients(B_z, train_x)
        dyz = gradients(B_y, train_z)
        dyx = gradients(B_y, train_x)
        dxy = gradients(B_x, train_y)
        dxz = gradients(B_x, train_z)        
        loss_BC_div = torch.mean(torch.square(dx+dy+dz))
        loss_BC_cul = torch.mean(torch.square(dzy - dyz) + torch.square(dxz - dzx) + torch.square(dyx - dxy))

        y_f = np.random.default_rng().uniform(low = -self.L/2, high = self.L/2, size = ((self.N_f, 1)))
        if(train_y.max()>0):
            x_f = np.random.default_rng().uniform(low = -self.L/2, high = self.L/2, size = ((self.N_f, 1)))
        else:
            x_f = np.random.default_rng().uniform(low = -self.L/10, high = self.L/10, size = ((self.N_f, 1)))
        z_f = np.random.default_rng().uniform(low = -self.L/2, high = self.L/2, size = ((self.N_f, 1)))
        self.x_f = torch.tensor(x_f, dtype = torch.float32).to(device).requires_grad_(True)
        self.y_f = torch.tensor(y_f, dtype = torch.float32).to(device).requires_grad_(True)
        self.z_f = torch.tensor(z_f, dtype = torch.float32).to(device).requires_grad_(True)
        temp_pred = model(torch.cat((self.x_f, self.y_f, self.z_f), axis=1))
        temp_ux = temp_pred[:,0].requires_grad_(True)
        temp_uy = temp_pred[:,1].requires_grad_(True)
        temp_uz = temp_pred[:,2].requires_grad_(True)
        u_x = gradients(temp_ux, self.x_f)
        u_y = gradients(temp_uy, self.y_f)
        u_z = gradients(temp_uz, self.z_f)
        u_zy = gradients(temp_uz, self.y_f) #dBz_f/dy_f
        u_zx = gradients(temp_uz, self.x_f) #dBz_f/dx_f
        u_yz = gradients(temp_uy, self.z_f) #dBy_f/dz_f
        u_yx = gradients(temp_uy, self.x_f) #dBy_f/dx_f
        u_xz = gradients(temp_ux, self.z_f) #dBx_f/dz_f
        u_xy = gradients(temp_ux, self.y_f) #dBx_f/dy_f
        #计算散度：div B = ∇·B = dBx_f/dx_f + dBy_f/dy_f + dBz_f/dz_f
        f = torch.mean(torch.square(u_x + u_y + u_z))
        #计算散度的平方作为loss_∇·B
        loss_f = torch.mean(torch.square(f))
        #计算旋度的模方：|∇×B|^2，作为loss_∇×B
        loss_cross = torch.mean(torch.square(u_zy - u_yz) + torch.square(u_xz - u_zx) + torch.square(u_yx - u_xy))
        #计算采样磁场大小和预测磁场大小的差，作为loss_B
        loss_u = torch.mean(torch.square(pred - labels))
        if(self.addBC):
            loss = loss_f + loss_u + loss_cross + loss_BC_div + loss_BC_cul
        else:
            loss  = loss_f + loss_u + loss_cross
        return loss_f, loss_u, loss_cross, loss_BC_div, loss_BC_cul, loss
