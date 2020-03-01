import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from sklearn.datasets import make_regression
from skorch import NeuralNetRegressor
X_regr, y_regr = make_regression(1000, 20, n_informative=10, random_state=0)
X_regr = X_regr.astype(np.float32)
y_regr = y_regr.astype(np.float32) / 100
y_regr = y_regr.reshape(-1, 1)
X_regr.shape, y_regr.shape, y_regr.min(), y_regr.max()

class RegressorModule(nn.Module):
    def __init__(
            self,
            num_units=10,
            nonlin=F.relu,
    ):
        super(RegressorModule, self).__init__()
        self.num_units = num_units
        self.nonlin = nonlin

        self.dense0 = nn.Linear(20, num_units)
        self.nonlin = nonlin
        self.dense1 = nn.Linear(num_units, 10)
        self.output = nn.Linear(10, 1)

    def forward(self, X, **kwargs):
        X = self.nonlin(self.dense0(X))
        X = F.relu(self.dense1(X))
        X = self.output(X)
        return X

net_regr = NeuralNetRegressor(
    RegressorModule,
    max_epochs=20,
    lr=0.1,
    device='cuda', 
)

net_regr.fit(X_regr, y_regr)

y_pred = net_regr.predict(X_regr[:5])
y_pred


a,b=net_regr.train_split(X_regr)
