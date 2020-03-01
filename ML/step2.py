import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from skorch import NeuralNetRegressor
import matplotlib.pyplot as plt
from torch.autograd import Variable
import time
from scipy.io import loadmat
from sklearn.model_selection import GridSearchCV

# load step3 data and other HF data
maindata = loadmat('main4-5.mat')
combineddata = maindata['combineddata']
t_d_oup  = combineddata[:,[0]].astype(np.float32)
t_d_inp  = combineddata[:,7:].astype(np.float32)
test_oup = maindata['data4'][:,[0]].astype(np.float32)
test_inp = maindata['data4'][:,7:].astype(np.float32)
yaxis = maindata['data4'][:,6].astype(np.float32)

# random split training and development set
nop = t_d_inp.shape[0]
indices = np.random.RandomState(seed=42).permutation(nop)
bp = np.int(nop*0.9)
train_idx, dev_idx = indices[:bp], indices[bp:]
train_inp, dev_inp = t_d_inp[train_idx,:], t_d_inp[dev_idx,:]
train_oup, dev_oup = t_d_oup[train_idx,:], t_d_oup[dev_idx,:]

# GPR
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
#kernel = ConstantKernel(1.0) * RBF(length_scale=1.0)
kernel = 1.0 * Matern(length_scale=1.0, length_scale_bounds=(1e-1, 10.0),
                        nu=1.5)
gpr = GaussianProcessRegressor(kernel=kernel).fit(train_inp, train_oup)
print (gpr.score(train_inp, train_oup))
predicted_gp1 = gpr.predict(train_inp) 
predicted_gp2 = gpr.predict(dev_inp) 
predicted_gp3 = gpr.predict(test_inp) 


class Net(nn.Module):
    def __init__(self, input_size=2, hidden_size=500, output_size=1, dropout=0.5):
        super(Net, self).__init__()
        #self.fc1 = nn.Linear(input_size, hidden_size) 
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.l1 = nn.ReLU()
        self.l2 = nn.Sigmoid()
        self.l3 = nn.Tanh()
        self.l4 = nn.ELU()
        self.l5 = nn.Hardshrink()
        self.ln = nn.Linear(hidden_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.dp = nn.Dropout(dropout)

    def forward(self, x):
        out = self.fc1(x)
        out = self.l3(out)
        out = self.ln(out)
        #out = self.dp(out)
        out = self.l1(out)
        out = self.fc2(out)
        return out

net_regr = NeuralNetRegressor(
    Net(hidden_size=500),
    max_epochs=200,
    lr=0.003,
    device='cuda', 
    optimizer=torch.optim.Adam,
    train_split=None,
    verbose=1,
)
net_regr.initialize()
net_regr.load_params('step1result')
res = net_regr.fit(t_d_inp, t_d_oup)
# save
net_regr.save_params(f_params='step2result')

pred = net_regr.predict(test_inp)
mse = ((test_oup-pred)**2).mean()
print('test error(NN) = '+str(mse))
mse2 = ((test_oup-predicted_gp3)**2).mean()
print('test error(GP) = '+str(mse2))
# plot 1 loss
loss = net_regr.history[:, 'train_loss']
plt.figure()
plt.plot(loss)
plt.ylabel('loss')
plt.ylim([0,loss[-1]*4])
# plot 2
plt.figure(figsize=(6,5))
plt.plot(yaxis,pred,'.-',label="ML_NeuralNetwork")
plt.plot(yaxis,test_oup,'.-',label="DNS")
plt.plot(yaxis,predicted_gp3,'.-',label="ML_Gaussianprocess")
plt.legend()
plt.ylabel('uu')
plt.xlabel('y+')
lossmessgae = 'MSE of NN = %10.3e\nMSE of GP = %10.3e' % (mse, mse2)
plt.annotate(lossmessgae, xy=(0.05, 0.05), xycoords='axes fraction')
#lgnd = plt.legend(loc='center left', bbox_to_anchor=(1.1, 0.65))
plt.savefig("step2_compare", bbox_inches='tight')

#net_regr.get_params
#params = {
#    'lr': [0.01, 0.001, 0.0001],
#    'max_epochs': [50, 500],
#    'module__hidden_size': [50, 100],
#}
#gs = GridSearchCV(net_regr, params, refit=False)
#gs.fit(t_d_inp, t_d_oup)
#print(gs.best_score_, gs.best_params_)