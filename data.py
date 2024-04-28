import numpy as np
import torch
import os
import scipy.special as sc

class data_generation:
    def __init__(self, radius=1, N_sample=3, N_test=1000, L=0.5):
        self.radius = radius
        self.N_sample = N_sample
        self.N_test = N_test
        self.L = L

    def circB(self,x,y,z):
        #Defining some parameters to be used in the formulas
        r_sq = x**2 + y**2 + z**2
        rho_sq = x**2 + y**2
        alpha_sq = 1. + r_sq - 2. * np.sqrt(rho_sq)
        beta_sq = 1. + r_sq + 2. * np.sqrt(rho_sq)
        k_sq = 1. - alpha_sq/beta_sq
    
        #Evaluate elliptic integrals
        e_k_sq = sc.ellipe(k_sq)
        k_k_sq = sc.ellipk(k_sq)
    
        #Magnetic field components in Cartesian coordinates
        Bx = 2. * x * z / (alpha_sq * rho_sq * np.sqrt(beta_sq)) * ((self.radius + r_sq) * e_k_sq - alpha_sq * k_k_sq)
        By = y * Bx / x
        Bz = 2. / (alpha_sq * np.sqrt(beta_sq)) * ((self.radius - r_sq) * e_k_sq + alpha_sq * k_k_sq)
    
        #return np.concatenate([Bx.reshape(-1,1),By.reshape(-1,1),Bz.reshape(-1,1)], axis=1)*-100
        return np.array([Bx,By,Bz])*-100
   
    def circB_xy(self,x,y,z):
        #Defining some parameters to be used in the formulas
        r_sq = x**2 + y**2 + z**2
        rho_sq = x**2 + y**2
        alpha_sq = 1. + r_sq - 2. * np.sqrt(rho_sq)
        beta_sq = 1. + r_sq + 2. * np.sqrt(rho_sq)
        k_sq = 1. - alpha_sq/beta_sq

        #Evaluate elliptic integrals
        e_k_sq = sc.ellipe(k_sq)
        k_k_sq = sc.ellipk(k_sq)

        #Magnetic field components in Cartesian coordinates
        Bx = 2. * x * z / (alpha_sq * rho_sq * np.sqrt(beta_sq)) * ((self.radius + r_sq) * e_k_sq - alpha_sq * k_k_sq)
        By = y * Bx / x
        Bz = 2. / (alpha_sq * np.sqrt(beta_sq)) * ((self.radius - r_sq) * e_k_sq + alpha_sq * k_k_sq)

        #return np.concatenate([Bx.reshape(-1,1),By.reshape(-1,1),Bz.reshape(-1,1)], axis=1)*-100
        return np.array([Bx,By,Bz])

    def circB_yz(self, x, y, z):
    
        # 坐标转换，将线圈旋转到yz平面
        x_prime = z  # x' = z
        y_prime = y  # y' = y
        z_prime = -x # z' = -x
    
        # 定义一些参数，用于公式中
        r_sq_prime = x_prime**2 + y_prime**2 + z_prime**2
        rho_sq_prime = x_prime**2 + y_prime**2
        alpha_sq_prime = 1. + r_sq_prime - 2. * np.sqrt(rho_sq_prime)
        beta_sq_prime = 1. + r_sq_prime + 2. * np.sqrt(rho_sq_prime)
        k_sq_prime = 1. - alpha_sq_prime / beta_sq_prime
    
        # 计算椭圆积分
        e_k_sq_prime = sc.ellipe(k_sq_prime)
        k_k_sq_prime = sc.ellipk(k_sq_prime)
    
        # 计算旋转后的磁场分量
        Bx_prime = 2. * x_prime * z_prime / (alpha_sq_prime * rho_sq_prime * np.sqrt(beta_sq_prime)) * ((self.radius + r_sq_prime) * e_k_sq_prime - alpha_sq_prime * k_k_sq_prime)
        By_prime = y_prime * Bx_prime / x_prime
        Bz_prime = 2. / (alpha_sq_prime * np.sqrt(beta_sq_prime)) * ((self.radius - r_sq_prime) * e_k_sq_prime + alpha_sq_prime * k_k_sq_prime)
    
        # 由于线圈旋转，磁场方向也需要相应地调整
        # 原始函数中Bx, By, Bz对应于xy平面内的磁场分量，现在我们需要将它们转换为yz平面内的分量
        # 旋转90度后，Bx变为Bz，By保持不变，Bz变为-Bx
        Bx_rotated = Bz_prime
        By_rotated = By_prime
        Bz_rotated = -Bx_prime
    
        # 返回旋转后的磁场分量
        return np.array([Bx_rotated, By_rotated, Bz_rotated]) 

    def circB_zx(self, x, y, z):

        # 坐标转换，将线圈旋转到yz平面
        x_prime = x  # x' = x
        y_prime = -z  # y' = -z
        z_prime = y # z' = y

        # 定义一些参数，用于公式中
        r_sq_prime = x_prime**2 + y_prime**2 + z_prime**2
        rho_sq_prime = x_prime**2 + y_prime**2
        alpha_sq_prime = 1. + r_sq_prime - 2. * np.sqrt(rho_sq_prime)
        beta_sq_prime = 1. + r_sq_prime + 2. * np.sqrt(rho_sq_prime)
        k_sq_prime = 1. - alpha_sq_prime / beta_sq_prime

        # 计算椭圆积分
        e_k_sq_prime = sc.ellipe(k_sq_prime)
        k_k_sq_prime = sc.ellipk(k_sq_prime)

        # 计算旋转后的磁场分量
        Bx_prime = 2. * x_prime * z_prime / (alpha_sq_prime * rho_sq_prime * np.sqrt(beta_sq_prime)) * ((self.radius + r_sq_prime) * e_k_sq_prime - alpha_sq_prime * k_k_sq_prime)
        By_prime = y_prime * Bx_prime / x_prime
        Bz_prime = 2. / (alpha_sq_prime * np.sqrt(beta_sq_prime)) * ((self.radius - r_sq_prime) * e_k_sq_prime + alpha_sq_prime * k_k_sq_prime)

        # 由于线圈旋转，磁场方向也需要相应地调整
        # 原始函数中Bx, By, Bz对应于xy平面内的磁场分量，现在我们需要将它们转换为yz平面内的分量
        # 旋转90度后，Bx变为Bz，By保持不变，Bz变为-Bx
        Bx_rotated = Bx_prime
        By_rotated = -Bz_prime
        Bz_rotated = By_prime

        # 返回旋转后的磁场分量
        return np.array([Bx_rotated, By_rotated, Bz_rotated]) 


    def B(self,x,y,z):
        field = 1*self.circB(x + 1.01,y + 1.0,z - 4.0) + 1*self.circB(x - 1.01,y - 1.0, z - 4.0) + 1*self.circB(x + 1.01,y - 1.0,z - 4.0) + 1*self.circB(x - 1.01,y + 1.0,z - 4.0) + 1*self.circB(x + 1.01,y + 1.0,z + 4.0) + 1*self.circB(x - 1.01,y - 1.0,z + 4.0) + 1*self.circB(x + 1.01,y - 1.0,z + 4.0) + 1*self.circB(x - 1.01,y + 1.0,z + 4.0)
        return field.tolist()

    def HelmholtzB(self, x, y, z):
        field  = 0
        field += self.circB_xy(x, y, z-self.radius/2) + self.circB_xy(x, y, z+self.radius/2)
        field += self.circB_yz(x-self.radius/2, y, z) + self.circB_yz(x+self.radius/2, y, z)
        field += self.circB_zx(x, y+self.radius/2, z) + self.circB_zx(x, y-self.radius/2, z)
        return field.tolist()

    def train_data_cube(self, Btype='normal', inner_sample=False):
        L1 = self.L
        N = self.N_sample
        x1 = np.concatenate((-L1*np.ones([N, 1]), L1*np.ones([N, 1]), #L和-L上各取Nu个点，确保采样点在格子的表面
                        np.random.default_rng().uniform(low = -L1, high = L1, size = (4*N, 1))), #在-L到L之间随机生成4*Nu个点
                        axis = 0)
        y1 = np.concatenate((np.random.default_rng().uniform(low = -L1, high = L1, size = (2*N, 1)),
                        -L1*np.ones([N, 1]), L1*np.ones([N, 1]),
                        np.random.default_rng().uniform(low = -L1, high = L1, size = (2*N, 1))),
                        axis = 0)
        z1 = np.concatenate((np.random.default_rng().uniform(low = -L1, high = L1, size = (4*N, 1)),
                        -L1*np.ones([N, 1]), L1*np.ones([N, 1])),
                        axis = 0)
        x1 = torch.tensor(x1, dtype=torch.float32)
        y1 = torch.tensor(y1, dtype=torch.float32)
        z1 = torch.tensor(z1, dtype=torch.float32)
        pos1 = torch.cat((x1, y1, z1), axis=1)
        if(inner_sample):
            L2 = self.L*0.75
            x2 = np.concatenate((-L2*np.ones([N, 1]), L2*np.ones([N, 1]), 
                            np.random.default_rng().uniform(low = -L2, high = L2, size = (4*N, 1))),
                            axis = 0)
            y2 = np.concatenate((np.random.default_rng().uniform(low = -L2, high = L2, size = (2*N, 1)),
                            -L2*np.ones([N, 1]), L2*np.ones([N, 1]),
                            np.random.default_rng().uniform(low = -L2, high = L2, size = (2*N, 1))),
                            axis = 0)
            z2 = np.concatenate((np.random.default_rng().uniform(low = -L2, high = L2, size = (4*N, 1)),
                            -L2*np.ones([N, 1]), L2*np.ones([N, 1])),
                            axis = 0)
            x2 = torch.tensor(x2, dtype=torch.float32)
            y2 = torch.tensor(y2, dtype=torch.float32)
            z2 = torch.tensor(z2, dtype=torch.float32)
            pos2 = torch.cat((x2, y2, z2), axis=1)
            x  = torch.cat((x1,x2))
            y  = torch.cat((y1,y2))
            z  = torch.cat((z1,z2))
            pos = torch.cat((pos1, pos2))
        else:
            x = x1
            y = y1
            z = z1
            pos = pos1          
        if(Btype=='Helmholtz'):
            labels = torch.tensor([self.HelmholtzB(x[i], y[i], z[i]) for i in range(len(x))], requires_grad=True)
        else:
            labels = torch.tensor([self.B(x[i], y[i], z[i]) for i in range(len(x))], requires_grad=True)
        #if(os.path.exists('./data/xyz.pt')==False):
        #    torch.save(pos, f'./data/xyz.pt')
        #if(os.path.exists('./data/B.pt')==False):
        #    torch.save(labels, f'./data/B.pt')
        return  pos, labels
    
    def train_data_slice(self, Btype='normal'):
        L = self.L
        N = self.N_sample
        x = np.array([0,0,0,0,0,0]).reshape(-1,1)
        y = np.array([-L,0,L,-L,0,L]).reshape(-1,1)        
        z = np.array([L,L,L,-L,-L,-L]).reshape(-1,1)
        x = torch.tensor(x, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.float32)
        z = torch.tensor(z, dtype=torch.float32)
        pos = torch.cat((x, y, z), axis=1)
        if(Btype=='Helmholtz'):
            labels = torch.tensor([self.HelmholtzB(x[0], y[i], z[i]) for i in range(len(x))], requires_grad=True)
        else:
            labels = torch.tensor([self.B(0, y[i], z[i]) for i in range(len(y))], requires_grad=True)
        #if(os.path.exists('./data/xyz_slice.pt')==False):
        #    torch.save(pos, f'./data/xyz_slice.pt')
        #if(os.path.exists('./data/B_slice.pt')==False):
        #    torch.save(labels, f'./data/B_slice.pt')
        return  pos, labels

    def test_data_cube(self, Btype='normal'):
        L = self.L
        N = self.N_test
        x = np.random.default_rng().uniform(low = -L, high = L, size = ((N, 1))) #shape： 1000*1
        y = np.random.default_rng().uniform(low = -L, high = L, size = ((N, 1)))
        z = np.random.default_rng().uniform(low = -L, high = L, size = ((N, 1)))
        x = torch.tensor(x, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.float32)
        z = torch.tensor(z, dtype=torch.float32)        
        pos = torch.cat((x, y, z), axis = 1)
        if(Btype=='Helmholtz'):
            labels = torch.tensor([self.HelmholtzB(x[i], y[i], z[i]) for i in range(len(x))], requires_grad=True)
        else:
            labels = torch.tensor([self.B(x[i], y[i], z[i]) for i in range(len(x))], requires_grad=True)
        #if(os.path.exists('./data/test_xyz.pt')==False):
        #    torch.save(pos, f'./data/test_xyz.pt')
        #if(os.path.exists('./data/test_B.pt')==False):
        #    torch.save(labels, f'./data/test_B.pt')
        return pos, labels

    def test_data_slice(self, Btype='normal'):
        L = self.L
        N = int(self.N_test/10)
        x = np.random.default_rng().uniform(low = -L/10, high = L/10, size = ((N, 1)))
        y = np.random.default_rng().uniform(low = -L/2, high = L/2, size = ((N, 1))) #shape： 1000*1
        z = np.random.default_rng().uniform(low = -L/2, high = L/2, size = ((N, 1)))
        x = torch.tensor(x, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.float32)
        z = torch.tensor(z, dtype=torch.float32)
        pos = torch.cat((x, y, z), axis = 1)
        if(Btype=='Helmholtz'):
            print(Btype)
            labels = torch.tensor([self.HelmholtzB(x[0], y[i], z[i]) for i in range(len(x))], requires_grad=True)
        else:
            labels = torch.tensor([self.B(x[0], y[i], z[i]) for i in range(len(x))], requires_grad=True)
        #if(os.path.exists('./data/test_xyz_slice.pt')==False):
        #    torch.save(pos, f'./data/test_xyz_slice.pt')
        #if(os.path.exists('./data/test_B_slice.pt')==False):
        #    torch.save(labels, f'./data/test_B_slice.pt')
        return pos, labels

