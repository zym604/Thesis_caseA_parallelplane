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
    max_epochs=5000,
    lr=0.01,
    device='cuda', 
    optimizer=torch.optim.Adam,
    train_split=None,
    verbose=1,
)

res = net_regr.fit(t_d_inp, t_d_oup)
# save
net_regr.save_params(f_params='step1result')

pred = net_regr.predict(test_inp)
mse = ((test_oup-pred)**2).mean()
print('test error = '+str(mse))
# plot 1 loss
loss = net_regr.history[:, 'train_loss']
plt.figure()
plt.plot(loss)
plt.ylabel('loss')
plt.ylim([0,loss[-1]*4])
# plot 2
plt.figure()
s=3
plt.scatter(yaxis,pred,s=s,label="Prediction")
plt.scatter(yaxis,test_oup,s=s,label="DNS")
plt.legend()

#net_regr.get_params
#params = {
#    'lr': [0.01, 0.001, 0.0001],
#    'max_epochs': [50, 500],
#    'module__hidden_size': [50, 100],
#}
#gs = GridSearchCV(net_regr, params, refit=False)
#gs.fit(t_d_inp, t_d_oup)
#print(gs.best_score_, gs.best_params_)