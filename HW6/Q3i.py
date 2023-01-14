#Q3i

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

#importing the dataset

train_set = torchvision.datasets.FashionMNIST(root = "./data", train = True, download = True, transform = transforms.ToTensor())
test_set = torchvision.datasets.FashionMNIST(root = "./data", train = False, download = True, transform = transforms.ToTensor())

train_loader = torch.utils.data.DataLoader(train_set, batch_size = 100, shuffle = True)
test_loader = torch.utils.data.DataLoader(test_set, batch_size = 100, shuffle = False)

#defining the model
model = torch.nn.Sequential(nn.Linear(in_features = 28*28 , out_features = 128), 
                            nn.ReLU(),
                            nn.Linear(in_features = 128, out_features = 10)
                            )
#Defining the loss function and the optimiser
lossfn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr = 0.01)

num_epochs = 40

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
    x = x_batch.reshape(100,784)
    #print(x.shape)
    y_hat = model(x)
    l1_train.append(model[0].weight)
    l2_train.append(model[2].weight)
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
      output = model(x_test.reshape(100,784))
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

#Plotting the Weights for Input Layer

plt.hist(model[0].weight.detach().numpy().reshape(-1,1), bins = 100, color = 'skyblue', ec = 'black', lw = 0.5)
plt.grid()
plt.xlim([-0.15,0.15])
plt.xlabel('Magnitude of weights')
plt.ylabel('Frequency of weights')
plt.title('Histogram of Weights for input layer')

#Plotting the Weights for Hidden Layer

plt.hist(model[2].weight.detach().numpy().reshape(-1,1), bins = 100, color = 'skyblue', ec = 'black', lw = 0.5)
#plt.xlim([-0.15,0.15])
plt.grid()
plt.xlabel('Magnitude of weights')
plt.ylabel('Frequency of weights')
plt.title('Histogram of Weights for hidden layer')
