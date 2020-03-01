import numpy as np
import matplotlib.pyplot as plt
import time
from scipy.io import loadmat

# print start time
time_s = time.time()
print("Start time = "+time.ctime())

# load step3 data and other HF data
maindata = loadmat('main4-5.mat')
combineddata = maindata['combineddata']
#t_d_oup  = combineddata[:,:4]
t_d_oup  = combineddata[:,[0]]
t_d_inp  = combineddata[:,7:]
#test_oup = maindata['data4'][:,:4]
test_oup = maindata['data4'][:,[0]]
test_inp = maindata['data4'][:,7:]
yaxis = maindata['data4'][:,6]

# random split training and development set
nop = t_d_inp.shape[0]
indices = np.random.RandomState(seed=42).permutation(nop)
bp = np.int(nop*0.9)
train_idx, dev_idx = indices[:bp], indices[bp:]
train_inp, dev_inp = t_d_inp[train_idx,:], t_d_inp[dev_idx,:]
train_oup, dev_oup = t_d_oup[train_idx,:], t_d_oup[dev_idx,:]
print ("finish load and seperating data!")

# 3D plot
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
fig = plt.figure()
i=0
ax = fig.add_subplot(111, projection='3d')
ax.scatter(train_inp[:,0], train_inp[:,1], train_oup[:,i],label="training set")
ax.scatter(dev_inp[:,0], dev_inp[:,1], dev_oup[:,i],label="training set")
ax.scatter(test_inp[:,0], test_inp[:,1], test_oup[:,i],label="test set")
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')
ax.view_init(30, 210)
plt.legend()
plt.savefig('step2_fig1')

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
#plot 1
plt.figure(figsize=(10,7))
s = 3
for i in range(t_d_oup.shape[1]):
    plt.subplot(2,3,i+1)
    plt.scatter(train_oup[:,i],predicted_gp1[:,i],s=s,label="training set")
    plt.scatter(  dev_oup[:,i],predicted_gp2[:,i],s=s,label="development set")
    plt.scatter( test_oup[:,i],predicted_gp3[:,i],s=s,label="test set")
    train_error = ((predicted_gp1[:,i] - train_oup[:,i])**2).mean()
    dev_error = ((predicted_gp2[:,i] - dev_oup[:,i])**2).mean()
    test_error = ((predicted_gp3[:,i] - test_oup[:,i])**2).mean()
plt.legend()
lgnd = plt.legend(loc='center left', bbox_to_anchor=(1.1, 0.65))
for handle in lgnd.legendHandles:
    handle.set_sizes([30.0])
lossmessgae = "MSEs are:\n"+"Training set: "+str(train_error)+"\nDevelop set: "+str(dev_error)+"\nTest set      : "+str(test_error)
plt.annotate(lossmessgae, xy=(1.15, 0.1), xycoords='axes fraction')
plt.savefig("step2_fig2", bbox_inches='tight')
#plot 2
plt.figure(figsize=(10,7))
s = 3
for i in range(t_d_oup.shape[1]):
    plt.subplot(2,3,i+1)
    plt.scatter(yaxis,predicted_gp3[:,i],s=s,label="Prediction")
    plt.scatter(yaxis,test_oup[:,i],s=s,label="DNS")
plt.legend()
lgnd = plt.legend(loc='center left', bbox_to_anchor=(1.1, 0.65))
for handle in lgnd.legendHandles:
    handle.set_sizes([30.0])
plt.savefig("step2_fig3", bbox_inches='tight')

# read data
x_train = train_inp
y_train = train_oup
x_dev = dev_inp
y_dev = dev_oup
x_test = test_inp
y_test = test_oup
#inp = inp*[4,100,1,4,0.04,1]
#oup = oup*500
t_d_inp = t_d_inp.astype(np.float32)
t_d_oup = t_d_oup.astype(np.float32)
x_train = x_train.astype(np.float32)
y_train = y_train.astype(np.float32)
x_dev = x_dev.astype(np.float32)
y_dev = y_dev.astype(np.float32)
x_test = x_test.astype(np.float32)
y_test = y_test.astype(np.float32)

# Auto-pytorch 
# Load json
import json
with open("step1_results_fit.json") as file:
    results_fit=json.load(file)

from autoPyTorch import AutoNetRegression
autonet_config = {
    "result_logger_dir" : "refit/",
    "budget_type" : "epochs",
    "log_level" : "info", 
    "use_tensorboard_logger" : True,
    }
autonet = AutoNetRegression(**autonet_config)
results_refit = autonet.refit(X_train=t_d_inp,
                          Y_train=t_d_oup,
                          hyperparameter_config=results_fit['optimized_hyperparameter_config'],
                          autonet_config=autonet.get_current_autonet_config(),
                          budget=5000)
    
# print start time
time_s = time.time()
print("End time = "+time.ctime())

pred = autonet.predict(x_test)
score = autonet.score(x_train,y_train)
print(score)
#plot 1
import matplotlib.pyplot as plt
plt.figure()
plt.plot(t_d_oup,autonet.predict(t_d_inp),'.')
plt.xlabel('DNS')
plt.ylabel('prediction')
plt.savefig("step2_fig4", bbox_inches='tight')
#plot 2
plt.figure(figsize=(10,7))
s = 3
for i in range(t_d_oup.shape[1]):
    plt.subplot(2,3,i+1)
    plt.scatter(yaxis,pred,s=s,label="Prediction")
    plt.scatter(yaxis,y_test,s=s,label="DNS")
plt.legend()
lgnd = plt.legend(loc='center left', bbox_to_anchor=(1.1, 0.65))
for handle in lgnd.legendHandles:
    handle.set_sizes([30.0])
plt.savefig("step2_fig5", bbox_inches='tight')

# Save json
import json
with open("step2_results_refit.json", "w") as file:
    json.dump(results_refit, file)
