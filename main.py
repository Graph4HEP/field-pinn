from data import *
from model import *
from train import *
from Eval import *
from utils import *
import argparse
import json

parser = argparse.ArgumentParser(description='PINN field prediction')
parser.add_argument('--logdir', type=str, default='./log/', metavar='--',
                    help='log dir')
parser.add_argument('--experiment', type=str, default='training', metavar='--',
                    help='name of the experiment you want to do, like scan different learning rate, scan different sample points')
parser.add_argument('--device', type=str, default='cpu', metavar='--', choices=['cpu', 'cuda:0'],
                    help='device type, cpu or cuda:0')
parser.add_argument('--lr', type=float, default=0.001, metavar='--',
                    help='learning rate')
parser.add_argument('--adjust_lr', type=int, default=0, metavar='--', choices=[0, 1],
                    help='whether adjust the lr during training, 0 means no, 1 means yes')
parser.add_argument('--Nsamples', type=int, default=16, metavar='--',
                    help='number of sample points per surface')
parser.add_argument('--Ntest', type=int, default=1000, metavar='--', 
                    help='number of test points')
parser.add_argument('--radius', type=float, default=1, metavar='--',
                    help='radius of the coils')
parser.add_argument('--length', type=float, default=1, metavar='--',
                    help='side length of the area that you want to predict')
parser.add_argument('--units', type=int, default=32, metavar='--', 
                    help='number of neurals in a network layer')
parser.add_argument('--Nep', type=int, default=100001, metavar='--', 
                    help='number of epochs')
parser.add_argument('--Npde', type=int, default=256, metavar='--',
                    help='number of points to join the PDE calculation')
parser.add_argument('--addBC', type=int, default=0, metavar='--', choices=[0, 1],
                    help='add BC constrains or not, 0 means no, 1 means yes')
parser.add_argument('--standard', type=int, default=0, metavar='--', choices=[0, 1],
                    help='perform standardization or not, 0 means no, 1 mean yes')
parser.add_argument('--geo', type=str, default='cube', metavar='--', choices=['cube', 'slice'],
                    help='geo of the coils')
parser.add_argument('--Btype', type=str, default='Helmholtz', metavar='--', choices=['Helmholtz', 'normal'],
                    help='which type field you want to generation, Helmholtz or normal field')

args = parser.parse_args()
config = {}
config['logdir']    = args.logdir + '/' + args.experiment
config['lr']        = args.lr
config['adjust_lr'] = args.adjust_lr
config['device']    = args.device
config['Nsamples']  = args.Nsamples
config['Ntest']     = args.Ntest
config['radius']    = args.radius
config['length']    = args.length
config['units']      = args.units
config['Nep']       = args.Nep
config['Npde']      = args.Npde
config['addBC']     = args.addBC
config['standard']  = args.standard
config['geo']       = args.geo
config['Btype']     = args.Btype

path = mkdir(config['logdir'])
config['path'] = path

field = data_generation(radius=config['radius'],
                        N_sample=config['Nsamples'], 
                        N_test=config['Ntest'], 
                        L=config['length']/2
                       )

if(config['geo']=='cube'):
    train_data, train_labels = field.train_data_cube(config['Btype'])
    test_data, test_labels = field.test_data_cube(config['Btype'])
if(config['geo']=='slice'):
    train_data, train_labels = field.train_data_slice(config['Btype'])
    test_data, test_labels = field.test_data_slice(config['Btype'])

mean = 0
std  = 1 
if(config['standard']==1):
    mean = train_labels.mean()
    std  = train_labels.std()
    train_labels = ((train_labels - mean)/std).detach().numpy()
    test_labels  = ((test_labels  - mean)/std).detach().numpy()
    train_labels = torch.tensor(train_labels)
    test_labels  = torch.tensor(test_labels)
config['mean'] = mean
config['std']  = std
    
with open(f"{path}/config.json", 'w') as config_file:
    config_file.write( json.dumps(config, indent=4) )

model = train( train_data, train_labels, test_data, test_labels, config ) 

Eval(model, config, field)
