__author__ = "Max Dippel, Michael Burkart and Matthias Urban"
__version__ = "0.0.1"
__license__ = "BSD"

import os, sys
sys.path.append(os.path.abspath(os.path.join(__file__, "..", "..", "..")))
from autoPyTorch import AutoNetRegression
from autoPyTorch.data_management.data_manager import DataManager

# Note: You can write your own datamanager! Call fit train, valid data (numpy matrices) 
dm = DataManager()
dm.generate_regression(num_features=21, num_samples=1500)
X_train=dm.X
Y_train=dm.Y
X_valid=dm.X_train
Y_valid=dm.Y_train

# Note: every parameter has a default value, you do not have to specify anything. The given parameter allow a fast test.
autonet = AutoNetRegression(budget_type='epochs', min_budget=1, max_budget=9, num_iterations=1, log_level='info')

res = autonet.fit(X_train=X_train, Y_train=Y_train, X_valid=X_valid, Y_valid=Y_valid)

print(res)

res_autonet = autonet.predict(X_train)

##retrain
import numpy as np
import torch
from torch.autograd import Variable
model = autonet.get_pytorch_model()
model.eval()
#autonet.print_help()
X_train = X_train.astype(np.float32)
res_model = model(Variable(torch.from_numpy(X_train).cuda())).data.cpu().numpy()

#plot
import matplotlib.pyplot as plt
plt.figure(figsize=(10,7))
plt.subplot(221)
plt.plot(Y_train,res_model,'.')
plt.xlabel('True result')
plt.ylabel('Model result')
plt.subplot(222)
plt.plot(Y_train,res_autonet,'.')
plt.xlabel('True result')
plt.ylabel('Model result')
plt.subplot(223)
plt.plot(res_autonet,res_model,'.')
plt.xlabel('AutoNet result')
plt.ylabel('Model result')