#Q2

#importing the libraries

import h5py
import numpy as np
import matplotlib.pyplot as plt

#Extracting the weights and biases from .hdf5 file

data1 = h5py.File('mnist_network_params.hdf5', 'r')

#Assigning the weights and biases into separate arrays

w1 = np.asarray(data1['W1'])
w2 = np.asarray(data1['W2'])
w3 = np.asarray(data1['W3'])
b1 = np.asarray(data1['b1'])
b2 = np.asarray(data1['b2'])
b3 = np.asarray(data1['b3'])

#Displaying the shapes of the weights and bias arrays
print(w1.shape, w2.shape, w3.shape, b1.shape, b2.shape, b3.shape)

#Extracting the xdata and ydata from the hdf5 file
inputdata = h5py.File('mnist_testdata.hdf5', 'r')

#Assigning the xdata and ydata into separate arrays and printing the shape
x = np.asarray(inputdata['xdata'])
y = np.asarray(inputdata['ydata'])
print(x.shape, y.shape)

#Defining the ReLU activation function
def relu(x):
  return(np.maximum(x,0))

#Calculating the inputs for hidden layer 1 using ReLU
L1 = np.zeros((10000,200))
L1 = np.matmul(x,w1.T) + b1
relu1 = relu(L1)

#Calculating the inputs for hideen layer 2 using ReLU
L2 = np.zeros((10000,100))
L2 = np.matmul(relu1, w2.T) + b2
relu2 = relu(L2)

#Calculating the Output
L3 = np.zeros((10000,100))
L3 = np.matmul(relu2, w3.T) + b3

# Defining the Softmax activation function
e = np.exp(L3)
e2 = np.sum(e, axis = 1).reshape(-1,1)
softmax = e/e2

#Writing the outputs of the final layer to json file
l = []
for i in range(10000):
  data = {
        "activations": softmax[i].tolist(),
        "index" : i,
         "classification" : int(softmax[i].argmax()),         
  }
  l.append(data)

import json
with open("result.json", "w") as f:
  f.write(json.dumps(l))

ind = np.argmax(softmax, axis = 1)
ind1 = softmax.argmax()
#Comparing the number of correct predictions made by model with ydata
y1 = []
y2 = []
y1 = y.tolist()

#Converting One Hot encoded vectors to digits
for i in range(10000):
  if(y1[i] == [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]):
    y2.append(0)
  elif(y1[i] == [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]):
    y2.append(1)
  elif(y1[i] == [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]):
    y2.append(2)
  elif(y1[i] == [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]):
    y2.append(3)
  elif(y1[i] == [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]):
    y2.append(4)
  elif(y1[i] == [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0]):
    y2.append(5)
  elif(y1[i] == [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]):
    y2.append(6)
  elif(y1[i] == [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0]):
    y2.append(7)
  elif(y1[i] == [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0]):
    y2.append(8)
  else:
    y2.append(9)

y3 = ind.tolist()

count = 0
diff = []
for a,b in zip(y2,y3):
  if a==b:
    count = count + 1

#Calculating the indexes where the predictions were made incorrectly
for i in range(10000):
  if(y2[i]!=y3[i]):
    diff.append(i)

#Storing the index values of incorrectly predicted digits

d = [8, 900, 7216, 1224, 9944]

#Visualization of incorrectly predicted digits
plt.figure(figsize=[20,20])
i =0

for j in d:
  # for i in range(len(d)):
  plt.subplot(1,len(d), i+1)
  plt.imshow(x[j].reshape(28,28))
  plt.xlabel('Predicted Value {}\n Actual Value {}'.format(y3[j], y2[j]))
  i+=1
plt.suptitle('Incorrectly Predicted Digits', y = 0.6,va = 'center', fontsize = 24)
plt.show()

#Visualization of Correctly Predicted Digits

plt.figure(figsize=[20,20])

t = [1, 45, 2000, 4500, 8567]
i =0
for j in t:
  # for i in range(len(d)):
  plt.subplot(1,len(d), i+1)
  plt.imshow(x[j].reshape(28,28))
  plt.xlabel('Predicted Value {}\n Actual Value {}'.format(y3[j], y2[j]))
  i+=1
plt.suptitle('Correctly Predicted Digits', y = 0.6,va = 'center', fontsize = 24)
plt.show()
