#Q2a K-means using Scipy

import numpy as np
from numpy import loadtxt
import matplotlib.pyplot as plt
from scipy.cluster.vq import kmeans2

values = np.loadtxt('cluster.txt', comments='#' ,delimiter=' ', usecols=[0,1])
labels = np.loadtxt('cluster.txt', comments = '#' ,delimiter=' ', usecols=[2], dtype = 'str')

cluster_1 = values[labels == 'Head']
cluster_2 = values[labels == 'Ear_right']
cluster_3 = values[labels == 'Ear_left']
plt.scatter(cluster_1[:,0], cluster_1[:,1], label ='Head', color = 'blue')
plt.scatter(cluster_2[:,0], cluster_2[:,1], label ='Ear_Right', color = 'green')
plt.scatter(cluster_3[:,0], cluster_3[:,1], label ='Ear_Left', color = 'red')
plt.show()

centroids, labels2 = kmeans2(values, 3, minit = 'points')

cluster_1 = values[labels2 == 0]
cluster_2 = values[labels2 == 1]
cluster_3 = values[labels2 == 2]

l = []
for i in labels2:
  if(i==2):
    l.append('Head')
  elif(i==1):
    l.append('Ear_left')
  else:
    l.append('Ear_right')

l = np.array(l)

plt.scatter(cluster_1[:,0], cluster_1[:,1], label ='Head', color = 'Blue')
plt.scatter(cluster_2[:,0], cluster_2[:,1], label ='Ear_right', color = 'red')
plt.scatter(cluster_3[:,0], cluster_3[:,1], label ='Ear_left', color = 'green')
plt.scatter(centroids[:,0], centroids[:,1], label = 'Centroids', color = 'black')
plt.show()

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(labels, l)
print(cm)
