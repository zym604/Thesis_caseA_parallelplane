import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch.autograd import Variable
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
plt.show()

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
plt.savefig("step4_gp", bbox_inches='tight')
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

# read data
x_train = train_inp
y_train = train_oup
x_dev = dev_inp
y_dev = dev_oup
x_test = test_inp
y_test = test_oup
#inp = inp*[4,100,1,4,0.04,1]
#oup = oup*500
x_train = x_train.astype(np.float32)
y_train = y_train.astype(np.float32)
x_dev = x_dev.astype(np.float32)
y_dev = y_dev.astype(np.float32)
x_test = x_test.astype(np.float32)
y_test = y_test.astype(np.float32)



# Auto-pytorch
import os, sys
sys.path.append(os.path.abspath(os.path.join(__file__, "..", "..", "..")))
from autoPyTorch import AutoNetRegression
# Note: every parameter has a default value, you do not have to specify anything. The given parameter allow a fast test.
autonet = AutoNetRegression(budget_type='epochs', max_runtime = 1500, min_budget=10, max_budget=90, num_iterations=1, log_level='info')

res = autonet.fit(X_train=x_train, Y_train=y_train, X_valid=x_dev, Y_valid=y_dev)

print(res)
# Save fit results as json
import json
with open("logs/results_fit.json", "w") as file:
    json.dump(res, file)

pred = autonet.predict(x_train)
score = autonet.score(x_train,y_train)
print(score)
#plot
import matplotlib.pyplot as plt
plt.figure()
plt.plot(y_train,pred,'.')
plt.savefig('regression')

##retrain
model = autonet.get_pytorch_model()
model.eval()
#autonet.print_help()
predicted1 = model(Variable(torch.from_numpy(x_train).cuda())).data.cpu().numpy()
#plot
plt.figure()
plt.plot(y_train,predicted1,'.')
plt.savefig('regression2')
#plot
plt.figure()
plt.plot(pred,predicted1,'.')
plt.savefig('regression2')

#save the model
torch.save(model,'savedmodel')
json.dump( res, open( "result_optimizer", 'w' ) )

## Hyper Parameters
#input_size = x_train.shape[1]
#hidden_size = 500
#output_size = y_train.shape[1]
#num_epochs = 50000
#learning_rate = 0.001
#error = 0.01
#relativerange = 500
#relativeerror = 0.00001
#dropout_rate = 0.5
#weight_decay = 0.0002
#loadtrainedmodel = False
#onlyplot = False
#
## Linear Regression Model
#class Net(nn.Module):
#    def __init__(self, input_size, hidden_size, output_size, dropout=0.5):
#        super(Net, self).__init__()
#        #self.fc1 = nn.Linear(input_size, hidden_size) 
#        self.fc1 = nn.Linear(input_size, hidden_size)
#        self.l1 = nn.ReLU()
#        self.l2 = nn.Sigmoid()
#        self.l3 = nn.Tanh()
#        self.l4 = nn.ELU()
#        self.l5 = nn.Hardshrink()
#        self.ln = nn.Linear(hidden_size, hidden_size)
#        self.fc2 = nn.Linear(hidden_size, output_size)
#        self.dp = nn.Dropout(dropout)
#
#    def forward(self, x):
#        out = self.fc1(x)
#        out = self.l3(out)
#        out = self.ln(out)
#        #out = self.dp(out)
#        out = self.l1(out)
#        out = self.fc2(out)
#        return out
#
#model = Net(input_size, hidden_size, output_size, dropout_rate)
## print model summary
#nodata = np.prod(x_train.shape)
#noparas = sum([param.nelement() for param in model.parameters()])
#print("Total number of data elements:"+str(nodata))
#print("Total number of parameters   :"+str(noparas))
#for name, param in model.named_parameters():
#    print (name, "\t", param.nelement(), "\t\t", param.data.shape)
#if noparas>nodata:
#    print ("Use too much neurons!!!")
#else:
#    print ("Network is OK!")
#
## load trained model parameters
#if loadtrainedmodel:
#    model.load_state_dict(torch.load('step4_model_trained.pkl'))
#
## Loss and Optimizer
#criterion = nn.MSELoss()
#optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate ,weight_decay = weight_decay)
#
####### GPU
#if torch.cuda.is_available():
#    print ("We are using GPU now!!!")
#    model = model.cuda()
#else:
#    print ("We are NOT using GPU now!!!")
#    model = model.cuda()
#
#loss_values = []
## Train the Model 
#for epoch in range(num_epochs):
#    if onlyplot:
#        break
#    # Convert numpy array to torch Variable
#    if torch.cuda.is_available():
#        inputs  = Variable(torch.from_numpy(x_train).cuda())
#        targets = Variable(torch.from_numpy(y_train).cuda())
#    else:
#        inputs  = Variable(torch.from_numpy(x_train))
#        targets = Variable(torch.from_numpy(y_train))
#    # Forward + Backward + Optimize
#    optimizer.zero_grad()  
#    outputs = model(inputs)
#    loss = criterion(outputs, targets)
#    loss.backward()
#    optimizer.step()
#    # use another stop criterion
#    loss_values.append(loss.item())
#    avgloss = np.mean(loss_values[-relativerange:])
#    lastloss = loss_values[-1]
#    relat_error = abs(lastloss-avgloss)/avgloss
#    if (epoch+1) % 100 == 0:
#        print ('Epoch [%d/%d],\t Loss: %.4f,\t relative error: %.4f' 
#               %(epoch+1, num_epochs, loss.item(), relat_error))
#    if relat_error < relativeerror and epoch>relativerange and lastloss< error:
#        print ('Epoch [%d/%d],\t Loss: %.4f,\t relative error: %.4f' 
#               %(epoch+1, num_epochs, loss.item(), relat_error))
#        break
#
## print end time
#time_e = time.time()
#print ("End time = "+time.ctime())
#totaltime = time_e - time_s
#print ("total used time (s) = "+str(totaltime))
#print ("total used time (min) = "+str(totaltime/60.0))
#
## Save the Model
#torch.save(model.state_dict(), 'step4_model.pkl')
#
## Plot learning curve
#plt.figure()
##plt.plot(np.array(loss_values), markersize=1, marker=".", linewidth=1)
#plt.plot(np.array(loss_values))
#plt.ylim(0,lastloss*4)
#plt.ylabel("training loss")
#plt.xlabel("iteration")
#plt.savefig("step4_loss", bbox_inches='tight')
#print ("***learning curve plotted!!!")
#
## plot the graph - training, development and test set
#if torch.cuda.is_available():
#    predicted1 = model(Variable(torch.from_numpy(x_train).cuda())).data.cpu().numpy()
#    predicted2 = model(Variable(torch.from_numpy(x_dev).cuda())).data.cpu().numpy()
#    predicted3 = model(Variable(torch.from_numpy(x_test).cuda())).data.cpu().numpy()
#else:
#    predicted1 = model(Variable(torch.from_numpy(x_train))).data.numpy()
#    predicted2 = model(Variable(torch.from_numpy(x_dev))).data.numpy()
#    predicted3 = model(Variable(torch.from_numpy(x_test))).data.numpy()
#train_error = ((predicted1 - y_train)**2).mean()
#dev_error = ((predicted2 - y_dev)**2).mean()
#test_error = ((predicted3 - y_test)**2).mean()
#
#plt.figure(figsize=(10,7))
#titles=["u'u'","v'v'","w'w'","u'v'","u'w'","v'w'"]
#for i in range(t_d_oup.shape[1]):
#    plt.subplot(2,3,i+1)
#    ms = 5
#    plt.scatter(y_train[:,i],predicted1[:,i],s=ms,label="training set")
#    plt.scatter(y_dev[:,i]  ,predicted2[:,i],s=ms,label="development set")
#    plt.scatter(y_test[:,i] ,predicted3[:,i],s=ms,label="test set")
#    plt.title(titles[i])
#lgnd = plt.legend(loc='center left', bbox_to_anchor=(1.1, 0.65))
#for handle in lgnd.legendHandles:
#    handle.set_sizes([30.0])
#lossmessgae = "MSEs are:\n"+"Training set: "+str(train_error)+"\nDevelop set: "+str(dev_error)+"\nTest set      : "+str(test_error)
#plt.annotate(lossmessgae, xy=(1.15, 0.1), xycoords='axes fraction')
#plt.savefig("step4_compare", bbox_inches='tight')
##plot 2
#plt.figure(figsize=(10,7))
#s = 3
#for i in range(t_d_oup.shape[1]):
#    plt.subplot(2,3,i+1)
#    plt.scatter(yaxis,predicted3[:,i],s=s,label="Prediction")
#    plt.scatter(yaxis,test_oup[:,i],s=s,label="DNS")
#plt.legend()
#lgnd = plt.legend(loc='center left', bbox_to_anchor=(1.1, 0.65))
#for handle in lgnd.legendHandles:
#    handle.set_sizes([30.0])
#    
## calculate variance
#variance = dev_error/train_error
#print ("variance = " + str(variance))
