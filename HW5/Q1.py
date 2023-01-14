#TANH with 0.00001

import h5py
import numpy as np
import matplotlib.pyplot as plt

data1 = h5py.File('mnist_traindata.hdf5', 'r')
list(data1.keys())

xdata = np.asarray(data1['xdata'])
ydata = np.asarray(data1['ydata'])

#Training Dataset

train_ind = int((5/6)*xdata.shape[0])
train_x = xdata[:train_ind]
train_y = ydata[:train_ind]

#Validation Dataset

valid_x = xdata[train_ind:]
valid_y = ydata[train_ind:]

#importing test dataset
test = h5py.File('mnist_testdata.hdf5', 'r')

xtest = np.asarray(test['xdata'])
ytest = np.asarray(test['ydata'])

#Activation Functions

def softmax(x):
  a = np.exp(x - np.max(x, axis = 1).reshape(-1,1))
  return a/np.sum(a).reshape(-1,1)

def linear(x,w,b):
  return np.matmul(x, w) + b.reshape(1,-1)

def relu(x):
  return np.maximum(x,0)

def tanh(x):
  a = (np.exp(x) - np.exp(-x))/(np.exp(x)+np.exp(-x))
  return a

def d(x):
  return (x > 0).astype(float)

def tanh_d(x):
  a = 1- x**2
  return a

def acc(y_pred, y_true):
  z = y_pred >= np.max(y_pred, axis = 1).reshape(-1,1)
  a = np.equal(y_true,z)
  acc = np.sum(np.sum(a, axis = 1)==y_pred.shape[1])
  return acc/y_true.shape[0]

learning_rate = 0.00001
mini_batch = 1
n_epochs = 50
n_iters = int(train_x.shape[0]/mini_batch)
loss_train_list = []
loss_test_list = []
loss_valid_list = []

acc_train_list = []
acc_test_list = []
acc_valid_list = []


#Initialising the weights
np.random.seed(seed = 243)
w1 = np.random.normal(0,1, size =(784,100))
b1 = np.random.normal(0,1, size =(100,1))

w2 = np.random.normal(0,1, size = (100,50))
b2 = np.random.normal(0,1, (50,1))

w3 = np.random.normal(0,1, size = (100,10))
b3 = np.random.normal(0,1, size = (10,1))


for i in range(n_epochs):
  iters = 1
  gradient_w3 = np.zeros(w3.shape)
  # gradient_w2 = np.zeros(w2.shape)
  gradient_w1 = np.zeros(w1.shape)

  gradient_b3 = np.zeros(b3.shape)
  # gradient_b2 = np.zeros(b2.shape)
  gradient_b1 = np.zeros(b1.shape)
  # In each epoch store the loss and y_pred
  for j in range(n_iters):
    iters += 1

    #Learning Rate Decay
    if ((i+1)%20==0):
       learning_rate = learning_rate/2
    
    #Defining the dataset according to the minibatch
    x_train_batch = train_x[j*mini_batch: (j+1)*mini_batch]
    y_train_batch = train_y[j*mini_batch: (j+1)*mini_batch]

    #Layer 1
    a1 = tanh(linear(x_train_batch, w1, b1))
    
    #Layer 2
    # a2 = relu(linear(a1, w2, b2))

    #Layer 3
    y_train_pred = softmax(linear(a1, w3, b3))

    #Backprop

    #Calculating deltas
    d3 = y_train_pred - y_train_batch
    # d2 = d(a2)*(np.matmul(d3, w3.T))
    d1 = (tanh_d(a1)*(np.matmul(d3,w3.T)))

    #Updating weights and biases
    gradient_w3 = gradient_w3 + np.matmul(a1.T, d3)
    gradient_b3 = gradient_b3 + np.sum(d3, axis = 0).reshape(-1,1)

    # gradient_w2 = gradient_w2 + np.matmul(a1.T, d2)
    # gradient_b2 = gradient_b2 + np.sum(d2, axis = 0).reshape(-1,1)

    gradient_w1 = gradient_w1 + np.matmul(x_train_batch.T, d1)
    gradient_b1 = gradient_b1 + np.sum(d1, axis = 0).reshape(-1,1)

    if iters % 100 == 0:
      w3 = w3 - learning_rate * (1/mini_batch) * gradient_w3
      # w2 = w2 - learning_rate * (1/mini_batch) * gradient_w2
      w1 = w1 - learning_rate * (1/mini_batch) * gradient_w1

      b3 = b3 - learning_rate * (1/mini_batch) * gradient_b3
      # b2 = b2 - learning_rate * (1/mini_batch) * gradient_b2
      b1 = b1 - learning_rate * (1/mini_batch) * gradient_b1
      # print('b3',b3.T)

      gradient_w3 = np.zeros(w3.shape)
      # gradient_w2 = np.zeros(w2.shape)
      gradient_w1 = np.zeros(w1.shape)

      gradient_b3 = np.zeros(b3.shape)
      # gradient_b2 = np.zeros(b2.shape)
      gradient_b1 = np.zeros(b1.shape)

  #for validation
  y1 = tanh(linear(valid_x, w1, b1))
  # y2 = relu(linear(y1, w2, b2))
  valid_pred = softmax(linear(y1, w3, b3))
  
  loss_valid = - (1/10000)*np.sum(valid_y*np.log(valid_pred + 1e-39))
  loss_valid_list.append(loss_valid)
  acc_valid_list.append(acc(valid_pred, valid_y))
  
  #for training 
  yt1 = tanh(linear(train_x, w1, b1))
  # yt2 = relu(linear(yt1, w2, b2))
  train_pred = softmax(linear(yt1, w3, b3))
  
  loss_train = - (1/60000)*np.sum(train_y*np.log(train_pred  + 1e-39))
  loss_train_list.append(loss_train)
  acc_train_list.append(acc(train_pred, train_y))

  print("Epoch : ", i)
  print(loss_train, loss_valid)

markers_on = [20,40]
plt.plot(acc_train_list,'-gD',markevery=markers_on)
plt.plot(acc_valid_list)
plt.xlabel('Number of epochs')
plt.ylabel('Accuracy')
plt.title('Number of epochs vs Accuracy')
plt.legend(['Training Set Accuracy', 'Validation Set Accuracy'])

#TANH with 0.0001

import h5py
import numpy as np
import matplotlib.pyplot as plt

data1 = h5py.File('mnist_traindata.hdf5', 'r')
list(data1.keys())

xdata = np.asarray(data1['xdata'])
ydata = np.asarray(data1['ydata'])

#Training Dataset

train_ind = int((5/6)*xdata.shape[0])
train_x = xdata[:train_ind]
train_y = ydata[:train_ind]

#Validation Dataset

valid_x = xdata[train_ind:]
valid_y = ydata[train_ind:]

#importing test dataset
test = h5py.File('mnist_testdata.hdf5', 'r')

xtest = np.asarray(test['xdata'])
ytest = np.asarray(test['ydata'])

#Activation Functions

def softmax(x):
  a = np.exp(x - np.max(x, axis = 1).reshape(-1,1))
  return a/np.sum(a).reshape(-1,1)

def linear(x,w,b):
  return np.matmul(x, w) + b.reshape(1,-1)

def relu(x):
  return np.maximum(x,0)

def tanh(x):
  a = (np.exp(x) - np.exp(-x))/(np.exp(x)+np.exp(-x))
  return a

def d(x):
  return (x > 0).astype(float)

def tanh_d(x):
  a = 1- x**2
  return a

def acc(y_pred, y_true):
  z = y_pred >= np.max(y_pred, axis = 1).reshape(-1,1)
  a = np.equal(y_true,z)
  acc = np.sum(np.sum(a, axis = 1)==y_pred.shape[1])
  return acc/y_true.shape[0]

learning_rate = 0.0001
mini_batch = 1
n_epochs = 50
n_iters = int(train_x.shape[0]/mini_batch)
loss_train_list = []
loss_test_list = []
loss_valid_list = []

acc_train_list = []
acc_test_list = []
acc_valid_list = []


#Initialising the weights
np.random.seed(seed = 243)
w1 = np.random.normal(0,1, size =(784,100))
b1 = np.random.normal(0,1, size =(100,1))

w2 = np.random.normal(0,1, size = (100,50))
b2 = np.random.normal(0,1, (50,1))

w3 = np.random.normal(0,1, size = (100,10))
b3 = np.random.normal(0,1, size = (10,1))


for i in range(n_epochs):
  iters = 1
  gradient_w3 = np.zeros(w3.shape)
  # gradient_w2 = np.zeros(w2.shape)
  gradient_w1 = np.zeros(w1.shape)

  gradient_b3 = np.zeros(b3.shape)
  # gradient_b2 = np.zeros(b2.shape)
  gradient_b1 = np.zeros(b1.shape)
  # In each epoch store the loss and y_pred
  for j in range(n_iters):
    iters += 1

    #Learning Rate Decay
    if ((i+1)%20==0):
       learning_rate = learning_rate/2
    
    #Defining the dataset according to the minibatch
    x_train_batch = train_x[j*mini_batch: (j+1)*mini_batch]
    y_train_batch = train_y[j*mini_batch: (j+1)*mini_batch]

    #Layer 1
    a1 = tanh(linear(x_train_batch, w1, b1))
    
    #Layer 2
    # a2 = relu(linear(a1, w2, b2))

    #Layer 3
    y_train_pred = softmax(linear(a1, w3, b3))

    #Backprop

    #Calculating deltas
    d3 = y_train_pred - y_train_batch
    # d2 = d(a2)*(np.matmul(d3, w3.T))
    d1 = (tanh_d(a1)*(np.matmul(d3,w3.T)))

    #Updating weights and biases
    gradient_w3 = gradient_w3 + np.matmul(a1.T, d3)
    gradient_b3 = gradient_b3 + np.sum(d3, axis = 0).reshape(-1,1)

    # gradient_w2 = gradient_w2 + np.matmul(a1.T, d2)
    # gradient_b2 = gradient_b2 + np.sum(d2, axis = 0).reshape(-1,1)

    gradient_w1 = gradient_w1 + np.matmul(x_train_batch.T, d1)
    gradient_b1 = gradient_b1 + np.sum(d1, axis = 0).reshape(-1,1)

    if iters % 100 == 0:
      w3 = w3 - learning_rate * (1/mini_batch) * gradient_w3
      # w2 = w2 - learning_rate * (1/mini_batch) * gradient_w2
      w1 = w1 - learning_rate * (1/mini_batch) * gradient_w1

      b3 = b3 - learning_rate * (1/mini_batch) * gradient_b3
      # b2 = b2 - learning_rate * (1/mini_batch) * gradient_b2
      b1 = b1 - learning_rate * (1/mini_batch) * gradient_b1
      # print('b3',b3.T)

      gradient_w3 = np.zeros(w3.shape)
      # gradient_w2 = np.zeros(w2.shape)
      gradient_w1 = np.zeros(w1.shape)

      gradient_b3 = np.zeros(b3.shape)
      # gradient_b2 = np.zeros(b2.shape)
      gradient_b1 = np.zeros(b1.shape)

  #for validation
  y1 = tanh(linear(valid_x, w1, b1))
  # y2 = relu(linear(y1, w2, b2))
  valid_pred = softmax(linear(y1, w3, b3))
  
  loss_valid = - (1/10000)*np.sum(valid_y*np.log(valid_pred + 1e-39))
  loss_valid_list.append(loss_valid)
  acc_valid_list.append(acc(valid_pred, valid_y))
  
  #for training 
  yt1 = tanh(linear(train_x, w1, b1))
  # yt2 = relu(linear(yt1, w2, b2))
  train_pred = softmax(linear(yt1, w3, b3))
  
  loss_train = - (1/60000)*np.sum(train_y*np.log(train_pred  + 1e-39))
  loss_train_list.append(loss_train)
  acc_train_list.append(acc(train_pred, train_y))

  print("Epoch : ", i)
  print(loss_train, loss_valid)

markers_on = [20,40]
plt.plot(acc_train_list,'-gD',markevery=markers_on)
plt.plot(acc_valid_list)
plt.xlabel('Number of epochs')
plt.ylabel('Accuracy')
plt.title('Number of epochs vs Accuracy')
plt.legend(['Training Set Accuracy', 'Validation Set Accuracy'])

#TANH with 0.001

import h5py
import numpy as np
import matplotlib.pyplot as plt

data1 = h5py.File('mnist_traindata.hdf5', 'r')
list(data1.keys())

xdata = np.asarray(data1['xdata'])
ydata = np.asarray(data1['ydata'])

#Training Dataset

train_ind = int((5/6)*xdata.shape[0])
train_x = xdata[:train_ind]
train_y = ydata[:train_ind]

#Validation Dataset

valid_x = xdata[train_ind:]
valid_y = ydata[train_ind:]

#importing test dataset
test = h5py.File('mnist_testdata.hdf5', 'r')

xtest = np.asarray(test['xdata'])
ytest = np.asarray(test['ydata'])

#Activation Functions

def softmax(x):
  a = np.exp(x - np.max(x, axis = 1).reshape(-1,1))
  return a/np.sum(a).reshape(-1,1)

def linear(x,w,b):
  return np.matmul(x, w) + b.reshape(1,-1)

def relu(x):
  return np.maximum(x,0)

def tanh(x):
  a = (np.exp(x) - np.exp(-x))/(np.exp(x)+np.exp(-x))
  return a

def d(x):
  return (x > 0).astype(float)

def tanh_d(x):
  a = 1- x**2
  return a

def acc(y_pred, y_true):
  z = y_pred >= np.max(y_pred, axis = 1).reshape(-1,1)
  a = np.equal(y_true,z)
  acc = np.sum(np.sum(a, axis = 1)==y_pred.shape[1])
  return acc/y_true.shape[0]

learning_rate = 0.001
mini_batch = 1
n_epochs = 50
n_iters = int(train_x.shape[0]/mini_batch)
loss_train_list = []
loss_test_list = []
loss_valid_list = []

acc_train_list = []
acc_test_list = []
acc_valid_list = []


#Initialising the weights
np.random.seed(seed = 243)
w1 = np.random.normal(0,1, size =(784,100))
b1 = np.random.normal(0,1, size =(100,1))

w2 = np.random.normal(0,1, size = (100,50))
b2 = np.random.normal(0,1, (50,1))

w3 = np.random.normal(0,1, size = (100,10))
b3 = np.random.normal(0,1, size = (10,1))


for i in range(n_epochs):
  iters = 1
  gradient_w3 = np.zeros(w3.shape)
  # gradient_w2 = np.zeros(w2.shape)
  gradient_w1 = np.zeros(w1.shape)

  gradient_b3 = np.zeros(b3.shape)
  # gradient_b2 = np.zeros(b2.shape)
  gradient_b1 = np.zeros(b1.shape)
  # In each epoch store the loss and y_pred
  for j in range(n_iters):
    iters += 1

    #Learning Rate Decay
    if ((i+1)%20==0):
       learning_rate = learning_rate/2
    
    #Defining the dataset according to the minibatch
    x_train_batch = train_x[j*mini_batch: (j+1)*mini_batch]
    y_train_batch = train_y[j*mini_batch: (j+1)*mini_batch]

    #Layer 1
    a1 = tanh(linear(x_train_batch, w1, b1))
    
    #Layer 2
    # a2 = relu(linear(a1, w2, b2))

    #Layer 3
    y_train_pred = softmax(linear(a1, w3, b3))

    #Backprop

    #Calculating deltas
    d3 = y_train_pred - y_train_batch
    # d2 = d(a2)*(np.matmul(d3, w3.T))
    d1 = (tanh_d(a1)*(np.matmul(d3,w3.T)))

    #Updating weights and biases
    gradient_w3 = gradient_w3 + np.matmul(a1.T, d3)
    gradient_b3 = gradient_b3 + np.sum(d3, axis = 0).reshape(-1,1)

    # gradient_w2 = gradient_w2 + np.matmul(a1.T, d2)
    # gradient_b2 = gradient_b2 + np.sum(d2, axis = 0).reshape(-1,1)

    gradient_w1 = gradient_w1 + np.matmul(x_train_batch.T, d1)
    gradient_b1 = gradient_b1 + np.sum(d1, axis = 0).reshape(-1,1)

    if iters % 100 == 0:
      w3 = w3 - learning_rate * (1/mini_batch) * gradient_w3
      # w2 = w2 - learning_rate * (1/mini_batch) * gradient_w2
      w1 = w1 - learning_rate * (1/mini_batch) * gradient_w1

      b3 = b3 - learning_rate * (1/mini_batch) * gradient_b3
      # b2 = b2 - learning_rate * (1/mini_batch) * gradient_b2
      b1 = b1 - learning_rate * (1/mini_batch) * gradient_b1
      # print('b3',b3.T)

      gradient_w3 = np.zeros(w3.shape)
      # gradient_w2 = np.zeros(w2.shape)
      gradient_w1 = np.zeros(w1.shape)

      gradient_b3 = np.zeros(b3.shape)
      # gradient_b2 = np.zeros(b2.shape)
      gradient_b1 = np.zeros(b1.shape)

  #for validation
  y1 = tanh(linear(valid_x, w1, b1))
  # y2 = relu(linear(y1, w2, b2))
  valid_pred = softmax(linear(y1, w3, b3))
  
  loss_valid = - (1/10000)*np.sum(valid_y*np.log(valid_pred + 1e-39))
  loss_valid_list.append(loss_valid)
  acc_valid_list.append(acc(valid_pred, valid_y))
  
  #for training 
  yt1 = tanh(linear(train_x, w1, b1))
  # yt2 = relu(linear(yt1, w2, b2))
  train_pred = softmax(linear(yt1, w3, b3))
  
  loss_train = - (1/60000)*np.sum(train_y*np.log(train_pred  + 1e-39))
  loss_train_list.append(loss_train)
  acc_train_list.append(acc(train_pred, train_y))

  print("Epoch : ", i)
  print(loss_train, loss_valid)

markers_on = [20,40]
plt.plot(acc_train_list,'-gD',markevery=markers_on)
plt.plot(acc_valid_list)
plt.xlabel('Number of epochs')
plt.ylabel('Accuracy')
plt.title('Number of epochs vs Accuracy')
plt.legend(['Training Set Accuracy', 'Validation Set Accuracy'])

#ReLU with 0.00001

import h5py
import numpy as np
import matplotlib.pyplot as plt

data1 = h5py.File('mnist_traindata.hdf5', 'r')
list(data1.keys())

xdata = np.asarray(data1['xdata'])
ydata = np.asarray(data1['ydata'])

#Training Dataset

train_ind = int((5/6)*xdata.shape[0])
train_x = xdata[:train_ind]
train_y = ydata[:train_ind]

#Validation Dataset

valid_x = xdata[train_ind:]
valid_y = ydata[train_ind:]

#importing test dataset
test = h5py.File('mnist_testdata.hdf5', 'r')

xtest = np.asarray(test['xdata'])
ytest = np.asarray(test['ydata'])

#Activation Functions

def softmax(x):
  a = np.exp(x - np.max(x, axis = 1).reshape(-1,1))
  return a/np.sum(a).reshape(-1,1)

def linear(x,w,b):
  return np.matmul(x, w) + b.reshape(1,-1)

def relu(x):
  return np.maximum(x,0)

def tanh(x):
  a = (np.exp(x) - np.exp(-x))/(np.exp(x)+np.exp(-x))
  return a

def d(x):
  return (x > 0).astype(float)

def tanh_d(x):
  a = 1- x**2
  return a

def acc(y_pred, y_true):
  z = y_pred >= np.max(y_pred, axis = 1).reshape(-1,1)
  a = np.equal(y_true,z)
  acc = np.sum(np.sum(a, axis = 1)==y_pred.shape[1])
  return acc/y_true.shape[0]

learning_rate = 0.00001
mini_batch = 1
n_epochs = 50
n_iters = int(train_x.shape[0]/mini_batch)
loss_train_list = []
loss_test_list = []
loss_valid_list = []

acc_train_list = []
acc_test_list = []
acc_valid_list = []


#Initialising the weights
np.random.seed(seed = 243)
w1 = np.random.normal(0,1, size =(784,100))
b1 = np.random.normal(0,1, size =(100,1))

w2 = np.random.normal(0,1, size = (100,50))
b2 = np.random.normal(0,1, (50,1))

w3 = np.random.normal(0,1, size = (100,10))
b3 = np.random.normal(0,1, size = (10,1))


for i in range(n_epochs):
  iters = 1
  gradient_w3 = np.zeros(w3.shape)
  # gradient_w2 = np.zeros(w2.shape)
  gradient_w1 = np.zeros(w1.shape)

  gradient_b3 = np.zeros(b3.shape)
  # gradient_b2 = np.zeros(b2.shape)
  gradient_b1 = np.zeros(b1.shape)
  # In each epoch store the loss and y_pred
  for j in range(n_iters):
    iters += 1

    #Learning Rate Decay
    if ((i+1)%20==0):
       learning_rate = learning_rate/2
    
    #Defining the dataset according to the minibatch
    x_train_batch = train_x[j*mini_batch: (j+1)*mini_batch]
    y_train_batch = train_y[j*mini_batch: (j+1)*mini_batch]

    #Layer 1
    a1 = relu(linear(x_train_batch, w1, b1))
    
    #Layer 2
    # a2 = relu(linear(a1, w2, b2))

    #Layer 3
    y_train_pred = softmax(linear(a1, w3, b3))

    #Backprop

    #Calculating deltas
    d3 = y_train_pred - y_train_batch
    # d2 = d(a2)*(np.matmul(d3, w3.T))
    d1 = (d(a1)*(np.matmul(d3,w3.T)))

    #Updating weights and biases
    gradient_w3 = gradient_w3 + np.matmul(a1.T, d3)
    gradient_b3 = gradient_b3 + np.sum(d3, axis = 0).reshape(-1,1)

    # gradient_w2 = gradient_w2 + np.matmul(a1.T, d2)
    # gradient_b2 = gradient_b2 + np.sum(d2, axis = 0).reshape(-1,1)

    gradient_w1 = gradient_w1 + np.matmul(x_train_batch.T, d1)
    gradient_b1 = gradient_b1 + np.sum(d1, axis = 0).reshape(-1,1)

    if iters % 100 == 0:
      w3 = w3 - learning_rate * (1/mini_batch) * gradient_w3
      # w2 = w2 - learning_rate * (1/mini_batch) * gradient_w2
      w1 = w1 - learning_rate * (1/mini_batch) * gradient_w1

      b3 = b3 - learning_rate * (1/mini_batch) * gradient_b3
      # b2 = b2 - learning_rate * (1/mini_batch) * gradient_b2
      b1 = b1 - learning_rate * (1/mini_batch) * gradient_b1
      # print('b3',b3.T)

      gradient_w3 = np.zeros(w3.shape)
      # gradient_w2 = np.zeros(w2.shape)
      gradient_w1 = np.zeros(w1.shape)

      gradient_b3 = np.zeros(b3.shape)
      # gradient_b2 = np.zeros(b2.shape)
      gradient_b1 = np.zeros(b1.shape)

  #for validation
  y1 = relu(linear(valid_x, w1, b1))
  # y2 = relu(linear(y1, w2, b2))
  valid_pred = softmax(linear(y1, w3, b3))
  
  loss_valid = - (1/10000)*np.sum(valid_y*np.log(valid_pred + 1e-39))
  loss_valid_list.append(loss_valid)
  acc_valid_list.append(acc(valid_pred, valid_y))
  
  #for training 
  yt1 = relu(linear(train_x, w1, b1))
  # yt2 = relu(linear(yt1, w2, b2))
  train_pred = softmax(linear(yt1, w3, b3))
  
  loss_train = - (1/60000)*np.sum(train_y*np.log(train_pred  + 1e-39))
  loss_train_list.append(loss_train)
  acc_train_list.append(acc(train_pred, train_y))

  print("Epoch : ", i)
  print(loss_train, loss_valid)

markers_on = [20,40]
plt.plot(acc_train_list,'-gD',markevery=markers_on)
plt.plot(acc_valid_list)
plt.xlabel('Number of epochs')
plt.ylabel('Accuracy')
plt.title('Number of epochs vs Accuracy')
plt.legend(['Training Set Accuracy', 'Validation Set Accuracy'])

#ReLU with 0.0001

import h5py
import numpy as np
import matplotlib.pyplot as plt

data1 = h5py.File('mnist_traindata.hdf5', 'r')
list(data1.keys())

xdata = np.asarray(data1['xdata'])
ydata = np.asarray(data1['ydata'])

#Training Dataset

train_ind = int((5/6)*xdata.shape[0])
train_x = xdata[:train_ind]
train_y = ydata[:train_ind]

#Validation Dataset

valid_x = xdata[train_ind:]
valid_y = ydata[train_ind:]

#importing test dataset
test = h5py.File('mnist_testdata.hdf5', 'r')

xtest = np.asarray(test['xdata'])
ytest = np.asarray(test['ydata'])

#Activation Functions

def softmax(x):
  a = np.exp(x - np.max(x, axis = 1).reshape(-1,1))
  return a/np.sum(a).reshape(-1,1)

def linear(x,w,b):
  return np.matmul(x, w) + b.reshape(1,-1)

def relu(x):
  return np.maximum(x,0)

def tanh(x):
  a = (np.exp(x) - np.exp(-x))/(np.exp(x)+np.exp(-x))
  return a

def d(x):
  return (x > 0).astype(float)

def tanh_d(x):
  a = 1- x**2
  return a

def acc(y_pred, y_true):
  z = y_pred >= np.max(y_pred, axis = 1).reshape(-1,1)
  a = np.equal(y_true,z)
  acc = np.sum(np.sum(a, axis = 1)==y_pred.shape[1])
  return acc/y_true.shape[0]

learning_rate = 0.0001
mini_batch = 1
n_epochs = 50
n_iters = int(train_x.shape[0]/mini_batch)
loss_train_list = []
loss_test_list = []
loss_valid_list = []

acc_train_list = []
acc_test_list = []
acc_valid_list = []


#Initialising the weights
np.random.seed(seed = 243)
w1 = np.random.normal(0,1, size =(784,100))
b1 = np.random.normal(0,1, size =(100,1))

w2 = np.random.normal(0,1, size = (100,50))
b2 = np.random.normal(0,1, (50,1))

w3 = np.random.normal(0,1, size = (100,10))
b3 = np.random.normal(0,1, size = (10,1))


for i in range(n_epochs):
  iters = 1
  gradient_w3 = np.zeros(w3.shape)
  # gradient_w2 = np.zeros(w2.shape)
  gradient_w1 = np.zeros(w1.shape)

  gradient_b3 = np.zeros(b3.shape)
  # gradient_b2 = np.zeros(b2.shape)
  gradient_b1 = np.zeros(b1.shape)
  # In each epoch store the loss and y_pred
  for j in range(n_iters):
    iters += 1

    #Learning Rate Decay
    if ((i+1)%20==0):
       learning_rate = learning_rate/2
    
    #Defining the dataset according to the minibatch
    x_train_batch = train_x[j*mini_batch: (j+1)*mini_batch]
    y_train_batch = train_y[j*mini_batch: (j+1)*mini_batch]

    #Layer 1
    a1 = relu(linear(x_train_batch, w1, b1))
    
    #Layer 2
    # a2 = relu(linear(a1, w2, b2))

    #Layer 3
    y_train_pred = softmax(linear(a1, w3, b3))

    #Backprop

    #Calculating deltas
    d3 = y_train_pred - y_train_batch
    # d2 = d(a2)*(np.matmul(d3, w3.T))
    d1 = (d(a1)*(np.matmul(d3,w3.T)))

    #Updating weights and biases
    gradient_w3 = gradient_w3 + np.matmul(a1.T, d3)
    gradient_b3 = gradient_b3 + np.sum(d3, axis = 0).reshape(-1,1)

    # gradient_w2 = gradient_w2 + np.matmul(a1.T, d2)
    # gradient_b2 = gradient_b2 + np.sum(d2, axis = 0).reshape(-1,1)

    gradient_w1 = gradient_w1 + np.matmul(x_train_batch.T, d1)
    gradient_b1 = gradient_b1 + np.sum(d1, axis = 0).reshape(-1,1)

    if iters % 100 == 0:
      w3 = w3 - learning_rate * (1/mini_batch) * gradient_w3
      # w2 = w2 - learning_rate * (1/mini_batch) * gradient_w2
      w1 = w1 - learning_rate * (1/mini_batch) * gradient_w1

      b3 = b3 - learning_rate * (1/mini_batch) * gradient_b3
      # b2 = b2 - learning_rate * (1/mini_batch) * gradient_b2
      b1 = b1 - learning_rate * (1/mini_batch) * gradient_b1
      # print('b3',b3.T)

      gradient_w3 = np.zeros(w3.shape)
      # gradient_w2 = np.zeros(w2.shape)
      gradient_w1 = np.zeros(w1.shape)

      gradient_b3 = np.zeros(b3.shape)
      # gradient_b2 = np.zeros(b2.shape)
      gradient_b1 = np.zeros(b1.shape)

  #for validation
  y1 = relu(linear(valid_x, w1, b1))
  # y2 = relu(linear(y1, w2, b2))
  valid_pred = softmax(linear(y1, w3, b3))
  
  loss_valid = - (1/10000)*np.sum(valid_y*np.log(valid_pred + 1e-39))
  loss_valid_list.append(loss_valid)
  acc_valid_list.append(acc(valid_pred, valid_y))
  
  #for training 
  yt1 = relu(linear(train_x, w1, b1))
  # yt2 = relu(linear(yt1, w2, b2))
  train_pred = softmax(linear(yt1, w3, b3))
  
  loss_train = - (1/60000)*np.sum(train_y*np.log(train_pred  + 1e-39))
  loss_train_list.append(loss_train)
  acc_train_list.append(acc(train_pred, train_y))

  print("Epoch : ", i)
  print(loss_train, loss_valid)

markers_on = [20,40]
plt.plot(acc_train_list,'-gD',markevery=markers_on)
plt.plot(acc_valid_list)
plt.xlabel('Number of epochs')
plt.ylabel('Accuracy')
plt.title('Number of epochs vs Accuracy')
plt.legend(['Training Set Accuracy', 'Validation Set Accuracy'])

#ReLU with 0.001

import h5py
import numpy as np
import matplotlib.pyplot as plt

data1 = h5py.File('mnist_traindata.hdf5', 'r')
list(data1.keys())

xdata = np.asarray(data1['xdata'])
ydata = np.asarray(data1['ydata'])

#Training Dataset

train_ind = int((5/6)*xdata.shape[0])
train_x = xdata[:train_ind]
train_y = ydata[:train_ind]

#Validation Dataset

valid_x = xdata[train_ind:]
valid_y = ydata[train_ind:]

#importing test dataset
test = h5py.File('mnist_testdata.hdf5', 'r')

xtest = np.asarray(test['xdata'])
ytest = np.asarray(test['ydata'])

#Activation Functions

def softmax(x):
  a = np.exp(x - np.max(x, axis = 1).reshape(-1,1))
  return a/np.sum(a).reshape(-1,1)

def linear(x,w,b):
  return np.matmul(x, w) + b.reshape(1,-1)

def relu(x):
  return np.maximum(x,0)

def tanh(x):
  a = (np.exp(x) - np.exp(-x))/(np.exp(x)+np.exp(-x))
  return a

def d(x):
  return (x > 0).astype(float)

def tanh_d(x):
  a = 1- x**2
  return a

def acc(y_pred, y_true):
  z = y_pred >= np.max(y_pred, axis = 1).reshape(-1,1)
  a = np.equal(y_true,z)
  acc = np.sum(np.sum(a, axis = 1)==y_pred.shape[1])
  return acc/y_true.shape[0]

learning_rate = 0.001
mini_batch = 1
n_epochs = 50
n_iters = int(train_x.shape[0]/mini_batch)
loss_train_list = []
loss_test_list = []
loss_valid_list = []

acc_train_list = []
acc_test_list = []
acc_valid_list = []


#Initialising the weights
np.random.seed(seed = 243)
w1 = np.random.normal(0,1, size =(784,100))
b1 = np.random.normal(0,1, size =(100,1))

w2 = np.random.normal(0,1, size = (100,50))
b2 = np.random.normal(0,1, (50,1))

w3 = np.random.normal(0,1, size = (100,10))
b3 = np.random.normal(0,1, size = (10,1))


for i in range(n_epochs):
  iters = 1
  gradient_w3 = np.zeros(w3.shape)
  # gradient_w2 = np.zeros(w2.shape)
  gradient_w1 = np.zeros(w1.shape)

  gradient_b3 = np.zeros(b3.shape)
  # gradient_b2 = np.zeros(b2.shape)
  gradient_b1 = np.zeros(b1.shape)
  # In each epoch store the loss and y_pred
  for j in range(n_iters):
    iters += 1

    #Learning Rate Decay
    if ((i+1)%20==0):
       learning_rate = learning_rate/2
    
    #Defining the dataset according to the minibatch
    x_train_batch = train_x[j*mini_batch: (j+1)*mini_batch]
    y_train_batch = train_y[j*mini_batch: (j+1)*mini_batch]

    #Layer 1
    a1 = relu(linear(x_train_batch, w1, b1))
    
    #Layer 2
    # a2 = relu(linear(a1, w2, b2))

    #Layer 3
    y_train_pred = softmax(linear(a1, w3, b3))

    #Backprop

    #Calculating deltas
    d3 = y_train_pred - y_train_batch
    # d2 = d(a2)*(np.matmul(d3, w3.T))
    d1 = (d(a1)*(np.matmul(d3,w3.T)))

    #Updating weights and biases
    gradient_w3 = gradient_w3 + np.matmul(a1.T, d3)
    gradient_b3 = gradient_b3 + np.sum(d3, axis = 0).reshape(-1,1)

    # gradient_w2 = gradient_w2 + np.matmul(a1.T, d2)
    # gradient_b2 = gradient_b2 + np.sum(d2, axis = 0).reshape(-1,1)

    gradient_w1 = gradient_w1 + np.matmul(x_train_batch.T, d1)
    gradient_b1 = gradient_b1 + np.sum(d1, axis = 0).reshape(-1,1)

    if iters % 100 == 0:
      w3 = w3 - learning_rate * (1/mini_batch) * gradient_w3
      # w2 = w2 - learning_rate * (1/mini_batch) * gradient_w2
      w1 = w1 - learning_rate * (1/mini_batch) * gradient_w1

      b3 = b3 - learning_rate * (1/mini_batch) * gradient_b3
      # b2 = b2 - learning_rate * (1/mini_batch) * gradient_b2
      b1 = b1 - learning_rate * (1/mini_batch) * gradient_b1
      # print('b3',b3.T)

      gradient_w3 = np.zeros(w3.shape)
      # gradient_w2 = np.zeros(w2.shape)
      gradient_w1 = np.zeros(w1.shape)

      gradient_b3 = np.zeros(b3.shape)
      # gradient_b2 = np.zeros(b2.shape)
      gradient_b1 = np.zeros(b1.shape)

  #for validation
  y1 = relu(linear(valid_x, w1, b1))
  # y2 = relu(linear(y1, w2, b2))
  valid_pred = softmax(linear(y1, w3, b3))
  
  loss_valid = - (1/10000)*np.sum(valid_y*np.log(valid_pred + 1e-39))
  loss_valid_list.append(loss_valid)
  acc_valid_list.append(acc(valid_pred, valid_y))
  
  #for training 
  yt1 = relu(linear(train_x, w1, b1))
  # yt2 = relu(linear(yt1, w2, b2))
  train_pred = softmax(linear(yt1, w3, b3))
  
  loss_train = - (1/60000)*np.sum(train_y*np.log(train_pred  + 1e-39))
  loss_train_list.append(loss_train)
  acc_train_list.append(acc(train_pred, train_y))

  print("Epoch : ", i)
  print(loss_train, loss_valid)

markers_on = [20,40]
plt.plot(acc_train_list,'-gD',markevery=markers_on)
plt.plot(acc_valid_list)
plt.xlabel('Number of epochs')
plt.ylabel('Accuracy')
plt.title('Number of epochs vs Accuracy')
plt.legend(['Training Set Accuracy', 'Validation Set Accuracy'])


