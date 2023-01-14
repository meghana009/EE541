#Q1a

import h5py
import numpy as np
import matplotlib.pyplot as plt

data1 = h5py.File('mnist_traindata.hdf5', 'r')
list(data1.keys())

xtrain = np.asarray(data1['xdata'])
ytrain = np.asarray(data1['ydata'])

y1 = ytrain.tolist()

yt = []
for i in range(60000):
  if(y1[i]==[0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]):
    yt.append(1)
  else:
    yt.append(0)
yt = np.array(yt).reshape(-1,1)

data2 = h5py.File('mnist_testdata.hdf5')
xtest = np.asarray(data2['xdata'])
ytest = np.asarray(data2['ydata'])

y2 = []
y2 = ytest.tolist()

ytesto = []
for i in range(10000):
  if(y2[i]==[0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]):
    ytesto.append(1)
  else:
    ytesto.append(0)
ytesto = np.array(ytesto).reshape(-1,1)

#Intialising the Weights and Bias for the hidden Layer
np.random.seed(seed = 1)
w = np.random.normal(0,10,size = (1,784))
#w = np.zeros((1,784))
#b = np.zeros(1)
b = np.random.randn(1) 

#Calculating the predicted values, binary log loss, updating the wieghts and biases
alpha = 0.09
loss = []
epsilon = 1e-40
i = 0
lam = 0.1
loss.append(np.inf)

trainacc = []
testaccu = []

#Appending one iteration of the loss
y = 1/(1 + np.exp(-(np.matmul(xtrain,w.T) + b)))
l = np.sum((np.sum((-yt)*np.log(y + epsilon)-(1-yt)*np.log(1-y + epsilon))))/60000 + lam*np.sum(w**2)
loss.append(l)

losst = []

ytest2 = 1/(1 + np.exp(-(np.matmul(xtest,w.T) + b)))
ltest = np.sum((np.sum((-ytesto)*np.log(ytest2+epsilon)-(1-ytesto)*np.log(1-ytest2+epsilon))))/10000 + lam*np.sum(w**2)
losst.append(np.inf)
losst.append(ltest)

#for i in range(2500):
while(np.abs(loss[i]-loss[i+1])> 10**-5):
  
  print( np.sum((y > 0.5) == yt) )
  trainacc.append( np.sum((y > 0.5) == yt) )
  
  print( np.sum((ytest2 > 0.5) == ytesto) )
  testaccu.append( np.sum((ytest2 > 0.5) == ytesto) )
  
  dw = (1/60000)*(np.matmul(xtrain.T,(y-yt))) + 2*lam*w.T
  db = (1/60000)*(np.sum(y-yt))
  #print("Shape of db:",db.shape)
  #print(dw)
  #print(db)
  
  w = w - alpha*dw.T
  
  #print("Shape of w:",w.shape)

  b = b - alpha*db
  y = 1/(1 + np.exp(-(np.matmul(xtrain,w.T) + b))) 
  #print("Shape of y:",y.shape)
  l = np.sum((np.sum((-yt)*np.log(y+epsilon)-(1-yt)*np.log(1-y+epsilon))))/60000 + lam*np.sum(w**2)
  
  #loss for test data
  ytest2 = 1/(1 + np.exp(-(np.matmul(xtest,w.T) + b)))
  ltest = np.sum((np.sum((-ytesto)*np.log(ytest2+epsilon)-(1-ytesto)*np.log(1-ytest2+epsilon))))/10000 + lam*np.sum(w**2)
  

  #Appending loss to a list
  loss.append(l)
  losst.append(ltest)
  print("Iteration and loss: ", i, l)
  
  
  i = i+1

plt.plot(loss)
plt.plot(losst)
plt.xlabel('Number of iterations')
plt.ylabel('Log Loss')
plt.title('Log Loss for the model using L2 regularisation, lambda = 0.1 and learning rate = 0.09')
plt.legend(['Training Set Loss', 'Test Set Loss'])

#Calculating the Accuracy

traina = []
for j in range(len(trainacc)):
  k = trainacc[j]/60000
  traina.append(k)
testa = []
for k in range(len(testaccu)):
  l = testaccu[k]/10000
  testa.append(l)

#Plotting the Graph
plt.plot(traina)
plt.plot(testa)
plt.xlabel('Number of iterations')
plt.ylabel('Accuracy')
plt.title('Number of Iterations vs Accuracy')
plt.legend(['Training Set Accuracy', 'Test Set Accuracy'])
