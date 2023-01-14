#Q2b Gaussian Mixture Models

import numpy as np
from numpy import loadtxt
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt
from scipy.cluster.vq import kmeans2

values = np.loadtxt('cluster.txt', comments='#' ,delimiter=' ', usecols=[0,1])
labels = np.loadtxt('cluster.txt', comments = '#' ,delimiter=' ', usecols=[2], dtype = 'str')
print(values.shape)

labels = np.array(labels)

centroids, labels2 = kmeans2(values, 3, minit = 'points')
print(centroids)

l1 = []
l2 = []
l3 = []
l = []
for i in labels2:
  if(i==0):
    l.append('Head')
  elif(i==2):
    l.append('Ear_left')
  else:
    l.append('Ear_right')

l = np.array(l)

for i in l:
  if (i=='Head'):
    l1.append([1,0,0])
  elif (i=='Ear_left'):
    l2.append([0,1,0])
  else:
    l3.append([0,0,1])
l1 = np.array(l1)
l2 = np.array(l2)
l3 = np.array(l3)
#print(l1.shape, l2.shape,l3.shape)
r = np.concatenate((l1,l2,l3), axis = 0)
print(r.shape)

gamma = r.copy()

mean = []
cov = []
wk = []


for i in range(3):
  mean.append((np.matmul(r[:,i].reshape(1,-1),values))/(np.sum(r[:,i])))
mean_np = np.concatenate(mean, axis = 0)

for i in range(3):
  sum1 = np.zeros((2,2))
  for j in range(490):
    sum1 = sum1 + r[j,i]*np.matmul((values[j] - mean_np[i]).reshape(-1,1),(values[j] - mean_np[i]).reshape(1,-1))
  cov.append(sum1.reshape(1,2,2)/np.sum(r[:,i]))
cov_np = np.concatenate(cov, axis=0)

for i in range(3):
  wk.append(np.sum(r[:,i])/490)
wk_np = np.array(wk)

print(cov_np, mean_np)

def pdf(x, mu, cov):
  # print(x, mu, cov)
  dist =  multivariate_normal(mu, cov)
  return dist.pdf(x)

# EM Algorithm
mean_iter = []
cov_iter = []
wk_iter = []
iters = 10

for i in range(iters):
  print("Iteration : ", i)
  #E Step
  r_temp = np.zeros((490,3))
  for i in range(3):
    for j in range(490):
      # print(pdf(values[j], mean_np[i], cov_np[i]))
      r_temp[j, i] = wk_np[i]*pdf(values[j], mean_np[i], cov_np[i])
  r_temp = r_temp/(np.sum(r_temp, axis = 1)).reshape(-1,1)
  r = r_temp.copy()
  
  #M Step
  mean = []
  cov = []
  wk = []


  for i in range(3):
    mean.append((np.matmul(r[:,i].reshape(1,-1),values))/(np.sum(r[:,i])))
  mean_np = np.concatenate(mean, axis = 0)
  mean_iter.append(mean_np)
  #print("Mean : ", mean_np)

  for i in range(3):
    sum1 = np.zeros((2,2))
    for j in range(490):
      sum1 = sum1 + r[j,i]*np.matmul((values[j] - mean_np[i]).reshape(-1,1),(values[j] - mean_np[i]).reshape(1,-1))
    cov.append(sum1.reshape(1,2,2)/np.sum(r[:,i]))
  cov_np = np.concatenate(cov, axis=0)
  cov_iter.append(cov_np)
  #print("Covariance : ", cov_np)

  for i in range(3):
    wk.append(np.sum(r[:,i])/490)
  wk_np = np.array(wk)  
  wk_iter.append(wk_np)
  #print("Weight : ", wk_np)

def inf(X,mu,cov,w):
  scores = []
  for i in range(3):
    scores.append(w[i]*pdf(X,mu[i],cov[i]))
  return scores.index(max(scores))

ref = []
for i in range(490):
  ref.append(inf(values[i],mean_iter[0],cov_iter[0],wk_iter[0]))
#print(ref)

cluster_1 = values[np.array(ref) == 0]
cluster_2 = values[np.array(ref) == 1]
cluster_3 = values[np.array(ref) == 2]

l = []
for i in ref:
  if(i==0):
    l.append('Head')
  elif(i==1):
    l.append('Ear_left')
  else:
    l.append('Ear_right')

l = np.array(l)

from sklearn.metrics import confusion_matrix
#print(labels)
cm = confusion_matrix(labels, l)
print(cm)
#print(cluster_1)
plt.scatter(cluster_1[:,0], cluster_1[:,1], label ='Head', color = 'Blue')
plt.scatter(cluster_2[:,0], cluster_2[:,1], label ='Ear_right', color = 'red')
plt.scatter(cluster_3[:,0], cluster_3[:,1], label ='Ear_left', color = 'green')
#plt.scatter(centroids[:,0], centroids[:,1], label = 'Centroids', color = 'black')
plt.show()

ref1 = []
for i in range(490):
  ref1.append(inf(values[i],mean_iter[1],cov_iter[1],wk_iter[1]))
#print(ref)

cluster_1 = values[np.array(ref1) == 0]
cluster_2 = values[np.array(ref1) == 1]
cluster_3 = values[np.array(ref1) == 2]

l = []
for i in ref1:
  if(i==0):
    l.append('Head')
  elif(i==1):
    l.append('Ear_left')
  else:
    l.append('Ear_right')

l = np.array(l)

from sklearn.metrics import confusion_matrix
#print(labels)
cm = confusion_matrix(labels, l)
print(cm)

plt.scatter(cluster_1[:,0], cluster_1[:,1], label ='Head', color = 'Blue')
plt.scatter(cluster_2[:,0], cluster_2[:,1], label ='Ear_right', color = 'red')
plt.scatter(cluster_3[:,0], cluster_3[:,1], label ='Ear_left', color = 'green')
#plt.scatter(centroids[:,0], centroids[:,1], label = 'Centroids', color = 'black')
plt.show()

ref2 = []
for i in range(490):
  ref2.append(inf(values[i],mean_iter[2],cov_iter[2],wk_iter[2]))
#print(ref2)

cluster_1 = values[np.array(ref2) == 0]
cluster_2 = values[np.array(ref2) == 1]
cluster_3 = values[np.array(ref2) == 2]

l = []
for i in ref2:
  if(i==0):
    l.append('Head')
  elif(i==1):
    l.append('Ear_left')
  else:
    l.append('Ear_right')

l = np.array(l)

from sklearn.metrics import confusion_matrix
#print(labels)
cm = confusion_matrix(labels, l)
print(cm)

plt.scatter(cluster_1[:,0], cluster_1[:,1], label ='Head', color = 'Blue')
plt.scatter(cluster_2[:,0], cluster_2[:,1], label ='Ear_right', color = 'red')
plt.scatter(cluster_3[:,0], cluster_3[:,1], label ='Ear_left', color = 'green')
#plt.scatter(centroids[:,0], centroids[:,1], label = 'Centroids', color = 'black')
plt.show()

ref3 = []
for i in range(490):
  ref3.append(inf(values[i],mean_iter[3],cov_iter[3],wk_iter[3]))
#print(ref2)

cluster_1 = values[np.array(ref3) == 0]
cluster_2 = values[np.array(ref3) == 1]
cluster_3 = values[np.array(ref3) == 2]

l = []
for i in ref3:
  if(i==0):
    l.append('Head')
  elif(i==1):
    l.append('Ear_left')
  else:
    l.append('Ear_right')

l = np.array(l)

from sklearn.metrics import confusion_matrix
#print(labels)
cm = confusion_matrix(labels, l)
print(cm)

plt.scatter(cluster_1[:,0], cluster_1[:,1], label ='Head', color = 'Blue')
plt.scatter(cluster_2[:,0], cluster_2[:,1], label ='Ear_right', color = 'red')
plt.scatter(cluster_3[:,0], cluster_3[:,1], label ='Ear_left', color = 'green')
plt.show()

ref10 = []
for i in range(490):
  ref10.append(inf(values[i],mean_iter[9],cov_iter[9],wk_iter[9]))
#print(ref2)

cluster_1 = values[np.array(ref10) == 0]
cluster_2 = values[np.array(ref10) == 1]
cluster_3 = values[np.array(ref10) == 2]

l = []
for i in ref10:
  if(i==0):
    l.append('Head')
  elif(i==1):
    l.append('Ear_left')
  else:
    l.append('Ear_right')

l = np.array(l)

from sklearn.metrics import confusion_matrix
#print(labels)
cm = confusion_matrix(labels, l)
print(cm)

plt.scatter(cluster_1[:,0], cluster_1[:,1], label ='Head', color = 'Blue')
plt.scatter(cluster_2[:,0], cluster_2[:,1], label ='Ear_right', color = 'red')
plt.scatter(cluster_3[:,0], cluster_3[:,1], label ='Ear_left', color = 'green')
#plt.scatter(centroids[:,0], centroids[:,1], label = 'Centroids', color = 'black')
plt.show()
