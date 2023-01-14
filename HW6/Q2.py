#Q2.

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

#Importing Dataset

train_set = torchvision.datasets.CIFAR10(root = "./data", train = True, download = True, transform = transforms.ToTensor())
test_set = torchvision.datasets.CIFAR10(root = "./data", train = False, download = True, transform = transforms.ToTensor())

train_loader = torch.utils.data.DataLoader(train_set, batch_size = 100, shuffle = True)
test_loader = torch.utils.data.DataLoader(test_set, batch_size = 100, shuffle = False)

#Defining the model
model = torch.nn.Sequential(nn.Linear(in_features = 3*32*32 , out_features = 256), 
                            nn.ReLU(),
                            nn.Dropout(0.3),
                            nn.Linear(in_features = 256, out_features = 128),
                            nn.ReLU(),
                            nn.Dropout(0.3)
                            )

#Defining loss function and optimiser
lossfn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr = 0.01, weight_decay = 0.0001)

num_epochs = 100

#Training the model
count = 0
accuracy_train_list = []
accuracy_test_list = []
loss_train_list = []
loss_test_list = []
l1_train = []
l2_train = []
y_train = []
y_true = []
for epoch in range(num_epochs):
  correct = 0
  for x_batch,y_batch in train_loader:
    count += 1
    #print(x_batch.shape)
    x = x_batch.reshape(100,3072)
    #print(x.shape)
    y_hat = model(x)
    # l1_train.append(model[0].weight)
    # l2_train.append(model[2].weight)
    loss_train = lossfn(y_hat,y_batch)

    optimizer.zero_grad()
    loss_train.backward()

    optimizer.step()
    #print(y_batch.shape)
    predictions = torch.max(y_hat, 1)[1]
    #predictions_batch = torch.max(y_batch, 1)[1]
    correct += (y_batch == predictions).sum().numpy()
    # y_train.append(y_hat)
    # y_true.append(y_batch)
  with torch.no_grad():
    total_test = 0
    correct_test = 0
    
    for x_test,y_test in test_loader:
      model.eval()
      output = model(x_test.reshape(100,3072))
      prediction = torch.max(output, 1)[1]
      #prediction_test = torch.max(y_test, 1)[1]
      loss_test = lossfn(output,y_test)
      correct_test += (prediction == y_test).sum().numpy()
      total_test += len(y_test)
    
    accuracy_test = correct_test*100/total_test

  
  loss_train_list.append(loss_train.data)
  loss_test_list.append(loss_test.data)
  accuracy_train_list.append(100*correct/len(train_loader.dataset))
  accuracy_test_list.append(accuracy_test)


  print(f'Epoch: {epoch+1:02d}, Iteration: {count: 5d}, Loss: {loss_train.data:.4f}, ' + f'Accuracy: {100*correct/len(train_loader.dataset):2.3f}%')

print('Completed')

#Computing the confusion matrix
for x_test,y_test in test_loader:
      model.eval()
      output2 = model(x_test.reshape(-1,3072))
      prediction2 = torch.max(output2, 1)[1]
      
from sklearn.metrics import confusion_matrix
import seaborn as sn

cf_matrix = confusion_matrix(prediction2, y_test)
print(cf_matrix)

#Generating heat map of confusion matrix
sn.heatmap(cf_matrix, fmt='0',annot=True)
