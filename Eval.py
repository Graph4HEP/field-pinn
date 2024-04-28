import torch
from data import *
import matplotlib.pyplot as plt

def getdB(Bpred, Breal):
    Bx = Bpred[:,0].detach().numpy()
    By = Bpred[:,1].detach().numpy()
    Bz = Bpred[:,2].detach().numpy()
    Bx_r = Breal[:,0]
    By_r = Breal[:,1]
    Bz_r = Breal[:,2]
    dBx = Bx - Bx_r
    dBy = By - By_r
    dBz = Bz - Bz_r
    dB = np.sqrt(Bx**2+By**2+Bz**2) - np.sqrt(Bx_r**2+By_r**2+Bz_r**2)
    return dBx, dBx, dBz, dB

def Eval(model, config, field):

    path  = config['path']
    mode  = config['geo']
    Btype = config['Btype']
    mean  = config['mean']
    std   = config['std']
    L = config['length']/2
    N_val =100

    model.to('cpu')
    y_test_np_grid = np.linspace(-L, L, N_val)
    if(mode=='cube'):
        x_test_np_grid = np.linspace(-L, L, N_val)
    if(mode=='slice'):
        x_test_np_grid = np.linspace(-0, 0, N_val)
    z_test_np_grid = np.linspace(-L, L, N_val)
    xx, yy, zz = np.meshgrid(x_test_np_grid, y_test_np_grid, z_test_np_grid, sparse=False)
    x_test_np = xx.reshape((N_val**3, 1))
    y_test_np = yy.reshape((N_val**3, 1))
    z_test_np = zz.reshape((N_val**3, 1))
    x_test = torch.tensor(x_test_np, dtype=torch.float32)
    y_test = torch.tensor(y_test_np, dtype=torch.float32)
    z_test = torch.tensor(z_test_np, dtype=torch.float32)
    
    inputs = torch.cat([x_test, y_test, z_test], axis = 1)
    if(Btype=='Helmholtz'):
        temp_final = np.array([field.HelmholtzB(x_test_np[i], y_test_np[i], z_test_np[i]) for i in range(N_val**3)])
    else:
        temp_final = np.array([field.B(x_test_np[i], y_test_np[i], z_test_np[i]) for i in range(N_val**3)])
    temp_final = temp_final.reshape((N_val, N_val, N_val, 3))
   
    model_output = model(inputs).view(N_val, N_val, N_val, 3)*std+mean

    model_output = model_output.reshape(N_val**3, 3)
    temp_final = temp_final.reshape(N_val**3, 3)
    
    dBx, dBy, dBz, dB = getdB(model_output, temp_final)
    fig_stat = plt.figure(figsize=([16,16]))
    fig_stat.add_subplot(2,2,1)
    plt.hist(dBx, bins=20, label=f"Bx_pred - Bx_real: mean {dBx.mean():.5f} std {dBx.std():.5f}")
    plt.legend()
    plt.yscale('log')
    fig_stat.add_subplot(2,2,2)
    plt.hist(dBy, bins=20, label=f"By_pred - By_real: mean {dBy.mean():.5f} std {dBy.std():.5f}")
    plt.legend()
    plt.yscale('log')
    fig_stat.add_subplot(2,2,3)
    plt.hist(dBz, bins=20, label=f"Bz_pred - Bz_real: mean {dBz.mean():.5f} std {dBz.std():.5f}")
    plt.legend()
    plt.yscale('log')
    fig_stat.add_subplot(2,2,4)
    plt.hist(dB, bins=20, label=f"B_pred - B_real: mean {dB.mean():.5f} std {dB.std():.5f}")
    plt.legend()
    plt.yscale('log')
    plt.savefig(f'{path}/hist_result.png')
    plt.show()
    plt.close()
    
    pred = model_output.reshape(N_val, N_val**2, 3)
    real = temp_final.reshape(N_val, N_val**2, 3)
    figure = plt.figure(figsize=(20,4))
    plt.rc('font', size=16)
    figure.add_subplot(1,3,1)
    R = xx.reshape(N_val, N_val**2)[10,:]
    Z = zz.reshape(N_val, N_val**2)[10,:]
    BZ = pred[10,:,2].detach().numpy()
    print(BZ.shape)
    R = R.reshape(N_val, N_val)
    Z = Z.reshape(N_val, N_val)
    BZ = BZ.reshape(N_val, N_val)
    CS = plt.contourf(R, Z, BZ, N_val, cmap='jet')
    plt.colorbar(CS)
    plt.xlabel('x')
    plt.ylabel('z')
    
    figure.add_subplot(1,3,2)
    plt.rc('font', size=16)
    R = xx.reshape(N_val, N_val**2)[10,:]
    Z = zz.reshape(N_val, N_val**2)[10,:]
    BZ = real[10,:,2]
    print(BZ.shape)
    R = R.reshape(N_val, N_val)
    Z = Z.reshape(N_val, N_val)
    BZ = BZ.reshape(N_val, N_val)
    CS = plt.contourf(R, Z, BZ, N_val, cmap='jet')
    plt.colorbar(CS)
    plt.xlabel('x')
    
    figure.add_subplot(1,3,3)
    plt.rc('font', size=16)
    R = xx.reshape(N_val, N_val**2)[10,:]
    Z = zz.reshape(N_val, N_val**2)[10,:]
    BZ = real[10,:,2]
    R = R.reshape(N_val, N_val)
    Z = Z.reshape(N_val, N_val)
    BZ = (pred[10,:,2].detach().numpy().reshape(N_val, N_val)-BZ.reshape(N_val, N_val)) / BZ.reshape(N_val, N_val)
    CS = plt.contourf(R,Z,BZ,N_val, cmap='jet')
    plt.colorbar(CS)
    plt.xlabel('x')
    plt.savefig(f'{path}/slice_result.png')
    plt.show()
    plt.close()
    
