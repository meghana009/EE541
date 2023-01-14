#Q1

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

from pathlib import Path

import h5py
import torch
from torch.utils import data

class HDF5Dataset(data.Dataset):
  """Abstract HDF5 dataset
  
  Usage:
    from hdf5_dataset import HDF5Dataset 
    train_set = hdf5_dataset.HDF5Dataset(
      file_path = f"{PATH_TO_HDF5}", data_name = 'xdata', label_name = 'ydata')
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=100, shuffle=True)
    
    ** NOTE: labels is 1-hot encoding, not target label number (as in PyTorch MNIST dataset)
    ** keep in mind when comparing model output to target labels
  
  Input params:
    file_path: Path to the folder containing the dataset
    data_name: name of hd5_file "x" dataset
    data_name: name of hd5_file "y"
  """
  def __init__(self, file_path, data_name, label_name):
    super().__init__()
    self.data = {}
    
    self.data_name = data_name
    self.label_name = label_name

    h5dataset_fp = Path(file_path)
    assert(h5dataset_fp.is_file())
    
    with h5py.File(file_path) as h5_file:
      # iterate datasets
      for dname, ds in h5_file.items():
        self.data[dname] = ds[()]
      
        
  def __getitem__(self, index):
    # get data
    x = self.data[self.data_name][index]
    x = torch.from_numpy(x)

    # get label
    y = self.data[self.label_name][index]
    y = torch.from_numpy(y)
    return (x, y)


  def __len__(self):
    return len(self.data[self.data_name])

#Defining the train_loader and test_loader

train_set = HDF5Dataset(file_path = "/content/mnist_traindata.hdf5", data_name = 'xdata', label_name = 'ydata')
train_loader = torch.utils.data.DataLoader(train_set, batch_size=100, shuffle=True)

test_set = HDF5Dataset(file_path = "/content/mnist_testdata.hdf5", data_name = 'xdata', label_name = 'ydata')
test_loader = torch.utils.data.DataLoader(test_set, batch_size = test_set.__len__())

#Defining the Model

num_pixels = 28*28

model = torch.nn.Sequential(nn.Linear(in_features=num_pixels, out_features=10))
loss_fn = nn.CrossEntropyLoss()
optimiser = torch.optim.SGD(model.parameters(), lr= 0.001, weight_decay = 1e-05)

num_epochs = 400

#Training the model

count = 0
accuracy_train_list = []
accuracy_test_list = []
loss_train_list = []
loss_test_list = []
y_train = []
y_true = []
for epoch in range(num_epochs):
  correct = 0
  for x_batch,y_batch in train_loader:
    count += 1
    x = x_batch

    y_hat = model(x)
    loss_train = loss_fn(y_hat,y_batch)

    optimiser.zero_grad()
    loss_train.backward()

    optimiser.step()

    predictions = torch.max(y_hat, 1)[1]
    predictions_batch = torch.max(y_batch, 1)[1]
    correct += (predictions_batch == predictions).sum().numpy()
    y_train.append(y_hat)
    y_true.append(y_batch)
  with torch.no_grad():
    total_test = 0
    correct_test = 0
    
    for x_test,y_test in test_loader:
      model.eval()
      output = model(x_test)
      prediction = torch.max(output, 1)[1]
      prediction_test = torch.max(y_test, 1)[1]
      loss_test = loss_fn(output,y_test)
      correct_test += (prediction == prediction_test).sum().numpy()
      total_test += len(y_test)
    
    accuracy_test = correct_test*100/total_test

  
  loss_train_list.append(loss_train.data)
  loss_test_list.append(loss_test.data)
  accuracy_train_list.append(100*correct/len(train_loader.dataset))
  accuracy_test_list.append(accuracy_test)


  print(f'Epoch: {epoch+1:02d}, Iteration: {count: 5d}, Loss: {loss_train.data:.4f}, ' + f'Accuracy: {100*correct/len(train_loader.dataset):2.3f}%')

print('Completed')


#Plotting the curves for training and test set for logloss and accuracy

plt.subplot(1,2,1)
plt.plot(loss_train_list)
plt.plot(loss_test_list)

plt.xlabel('Number of epochs')
plt.ylabel('Cross Entropy Loss')
#plt.title('Number of epochs vs Cross Entropy Loss')
plt.legend(['Training Loss','Test Loss'])

plt.subplot(1,2,2)
plt.plot(accuracy_train_list)
plt.plot(accuracy_test_list)

plt.xlabel('Number of epochs')
plt.ylabel('Accuracy')
#plt.title('Number of epochs vs Accuracy')
plt.legend(['Training Accuracy','Test Accuracy'])

plt.tight_layout(1)

plt.suptitle('Loss and Accuracy for lr = 0.001 with lambda=1e-05', y = 1.01, va = 'center')
plt.show()

#Calculating the confusion matrix and printing the heat map

for x_test,y_test in test_loader:
      model.eval()
      output2 = model(x_test)
      prediction2 = torch.max(output2, 1)[1]
      prediction_test2 = torch.max(y_test, 1)[1]

from sklearn.metrics import confusion_matrix
import seaborn as sn

cf_matrix = confusion_matrix(prediction2, prediction_test2)
print(cf_matrix)

sn.heatmap(cf_matrix, fmt='0',annot=True)
