#Q1b
import h5py
import numpy as np
import matplotlib.pyplot as plt

data1 = h5py.File('mnist_traindata.hdf5', 'r')
list(data1.keys())

xtrain = np.asarray(data1['xdata'])
ytrain = np.asarray(data1['ydata'])
print(xtrain.shape, ytrain.shape)

data2 = h5py.File('mnist_testdata.hdf5')
xtest = np.asarray(data2['xdata'])
ytest = np.asarray(data2['ydata'])

#Initialising the weights
np.random.seed(seed = 0)
w = np.random.normal(0,10, size =(784,10))
b = np.random.randn(10,1)

def softmax(x):
  a = np.exp(x - np.max(x, axis = 1).reshape(-1,1))
  return a/np.sum(a).reshape(-1,1)

def linear(x,w,b):
  return np.matmul(x, w) + b.reshape(1,-1)

def acc(y_pred, y_true):
  z = y_pred >= np.max(y_pred, axis = 1).reshape(-1,1)
  a = np.equal(y_true,z)
  acc = np.sum(np.sum(a, axis = 1)==y_pred.shape[1])
  # count = 0
  # for i in range(y_true.shape[0]):
  #   print(i, y_true.shape[0])
  #   if(np.argmax(y_pred)==np.argmax(y_true)):
  #     count += 1
  return acc/y_true.shape[0]

learning_rate = 0.01
mini_batch = 60000
n_epochs = 1000
n_iters = int(xtrain.shape[0]/mini_batch)
loss_train_list = []
loss_test_list = []

acc_train_list = []
acc_test_list = []

n_updates = 0
for i in range(n_epochs):
  # In each epoch store the loss and y_pred
  for j in range(n_iters):
    if n_updates % 5000 == 0:
      y_train_pred = softmax(linear(xtrain, w, b))
      y_test_pred = softmax(linear(xtest, w, b))
      loss_train = - (1/60000)*np.sum(np.log( np.sum(y_train_pred * ytrain, axis = 1) + 1e-39))
      loss_test = - (1/10000)*np.sum(np.log( np.sum(y_test_pred * ytest, axis = 1) + 1e-39))
      acc_train_list.append(acc(y_train_pred, ytrain))
      acc_test_list.append(acc(y_test_pred, ytest))
      loss_train_list.append(loss_train)
      loss_test_list.append(loss_test)

    n_updates += mini_batch
    x_train_batch = xtrain[j*mini_batch: (j+1)*mini_batch]
    y_train_batch = ytrain[j*mini_batch: (j+1)*mini_batch]
    # print("X train batch shape : {} Y train batch shape : {}".format(x_train_batch.shape, y_train_batch.shape))

    # Perform weight update each time
    y_train_pred_batch = softmax(linear(x_train_batch, w, b))
    dw = (1/mini_batch) * (np.matmul(x_train_batch.T, y_train_batch - y_train_pred_batch))
    # print("DW : ", dw[78])
    db = (1/mini_batch) * (np.sum(y_train_batch - y_train_pred_batch, axis = 0).reshape(-1,1))
    # print("dw shape : {} db shape : {}".format(dw.shape, db.shape))

    w = w + learning_rate * dw
    b = b + learning_rate * db

  # y_train_pred = softmax(linear(xtrain, w, b))
  # y_test_pred = softmax(linear(xtest, w, b))
  # loss_train = - (1/60000)*np.sum(np.log( np.sum(y_train_pred * ytrain, axis = 1) + 1e-39))
  # loss_test = - (1/10000)*np.sum(np.log( np.sum(y_test_pred * ytest, axis = 1) + 1e-39))
  # loss_train_list.append(loss_train)
  # loss_test_list.append(loss_test)
  print("Epoch : ", i)
  print(loss_train)
  print(loss_test)

  import matplotlib.pyplot as plt

  plt.plot(loss_train_list)
plt.plot(loss_test_list)
plt.xlabel('Number of batches')
plt.ylabel('Log Loss')
plt.title('Number of batches vs Log Loss')
plt.legend(['Training Set Loss', 'Test Set Loss'])

plt.plot(acc_train_list)
plt.plot(acc_test_list)
plt.xlabel('Number of batches')
plt.ylabel('Accuracy')
plt.title('Number of batches vs Accuracy')
plt.legend(['Training Set Accuracy', 'Test Set Accuracy'])
