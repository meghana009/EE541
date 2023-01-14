#Question 3a

import h5py
import numpy as np
import matplotlib.pyplot as plt

data = h5py.File('lms_fun_v3.hdf5', 'r')

#For SNR 10dB and n = 0.05
x_10 = np.asarray(data['matched_10_x'])
v_10 = np.asarray(data['matched_10_v'])
y_10 = np.asarray(data['matched_10_y'])
z_10 = np.asarray(data['matched_10_z'])


wn = np.array(wn)
w1 = []
y = np.zeros((600,501))
sum = np.zeros((3,1))
sum1 = 0
mse = np.zeros((600,501))

for i in range(600):
  wn = np.array([0, 0, 0])
  w2 = []
  for j in range(501):
    y[i][j] = np.matmul(wn.T,v_10[i][j])
    wn = wn + 0.05*(z_10[i][j]-y[i][j])*v_10[i][j]
    w2.append(wn)
  w1.append(w2)
    
print(y.shape)
w1 = np.array(w1)
mse = np.sum((y_10 - y)**2, axis = 0)/600

w_avg0 = np.sum( w1[:,:,0], axis = 0)/600
w_avg1 = np.sum( w1[:,:,1], axis = 0)/600
w_avg2 = np.sum( w1[:,:,2], axis = 0)/600

plt.rcParams['figure.figsize'] = [20,4]

fig, (ax1, ax2, ax3) = plt.subplots(1,3)
ax1.plot(w1[1,:,0])
ax1.plot(w1[1,:,1])
ax1.plot(w1[1,:,2])
ax1.set_title('Coefficients')
ax1.set_xlabel('updates')
ax1.set_ylabel('weights')

ax2.plot(w_avg0)
ax2.plot(w_avg1)
ax2.plot(w_avg2)
ax2.set_title('Coefficients(averaged)')
ax2.set_xlabel('updates')
ax2.set_ylabel('averaged weights')

ax3.plot(mse)
ax3.set_title('Learning Curve for 10dB, n = 0.05')
ax3.set_xlabel('Updates')
ax3.set_ylabel('MSE')

#For SNR 10dB and n = 0.15

wn = np.array(wn)
y = np.zeros((600,501))
sum = np.zeros((3,1))
sum1 = 0
mse = np.zeros((600,501))
w1 = []

for i in range(600):
  wn = np.array([0, 0, 0])
  w2 = []
  for j in range(501):
    y[i][j] = np.matmul(wn.T,v_10[i][j])
    wn = wn + 0.22*(z_10[i][j]-y[i][j])*v_10[i][j]
    
    w2.append(wn)
  w1.append(w2)
 
print(y.shape)

w1 = np.array(w1)
mse = np.sum((y_10 - y)**2, axis = 0)/600

w_avg0 = np.sum( w1[:,:,0], axis = 0)/600
w_avg1 = np.sum( w1[:,:,1], axis = 0)/600
w_avg2 = np.sum( w1[:,:,2], axis = 0)/600

plt.rcParams['figure.figsize'] = [20,4]
fig, (ax1, ax2, ax3) = plt.subplots(1,3)
ax1.plot(w1[1,:,0])
ax1.plot(w1[1,:,1])
ax1.plot(w1[1,:,2])
ax1.set_title('Coefficients')
ax1.set_xlabel('updates')
ax1.set_ylabel('weights')

ax2.plot(w_avg0)
ax2.plot(w_avg1)
ax2.plot(w_avg2)
ax2.set_title('Coefficients(averaged)')
ax2.set_xlabel('updates')
ax2.set_ylabel('averaged weights')

ax3.plot(mse)
ax3.set_title('Learning Curve for 10dB, n = 0.22')
ax3.set_xlabel('Updates')
ax3.set_ylabel('MSE')

x_3 = np.asarray(data['matched_3_x'])
v_3 = np.asarray(data['matched_3_v'])
y_3 = np.asarray(data['matched_3_y'])
z_3 = np.asarray(data['matched_3_z'])

#For SNR 3dB and n = 0.05

wn = np.array(wn)
y = np.zeros((600,501))
sum = np.zeros((3,1))
sum1 = 0
mse = np.zeros((600,501))
w1 = []

for i in range(600):
  wn = np.array([0, 0, 0])
  w2 = []
  for j in range(501):
    
    y[i][j] = np.matmul(wn.T,v_3[i][j])
    wn = wn + 0.05*(z_3[i][j]-y[i][j])*v_3[i][j]
    w2.append(wn)
  w1.append(w2)
    
print(y.shape)
w1 = np.array(w1)
mse = np.sum((y_3 - y)**2, axis = 0)/600

w_avg0 = np.sum( w1[:,:,0], axis = 0)/600
w_avg1 = np.sum( w1[:,:,1], axis = 0)/600
w_avg2 = np.sum( w1[:,:,2], axis = 0)/600

plt.rcParams['figure.figsize'] = [20,4]

fig, (ax1, ax2, ax3) = plt.subplots(1,3)
ax1.plot(w1[1,:,0])
ax1.plot(w1[1,:,1])
ax1.plot(w1[1,:,2])
ax1.set_title('Coefficients')
ax1.set_xlabel('updates')
ax1.set_ylabel('weights')

ax2.plot(w_avg0)
ax2.plot(w_avg1)
ax2.plot(w_avg2)
ax2.set_title('Coefficients(averaged)')
ax2.set_xlabel('updates')
ax2.set_ylabel('averaged weights')

ax3.plot(mse)
ax3.set_title('Learning Curve for 3dB, n = 0.05')
ax3.set_xlabel('Updates')
ax3.set_ylabel('MSE')

#For SNR 3dB and n = 0.15
wn = np.array(wn)
y = np.zeros((600,501))
sum = np.zeros((3,1))
sum1 = 0
mse = np.zeros((600,501))

w1 = []
for i in range(600):
  wn = np.array([0, 0, 0])
  
  w2 = []
  for j in range(501):
    
    y[i][j] = np.matmul(wn.T,v_10[i][j])
    wn = wn + 0.15*(z_10[i][j]-y[i][j])*v_10[i][j]
    w2.append(wn)
  w1.append(w2)
    
print(y.shape)
w1 = np.array(w1)
mse = np.sum((y_10 - y)**2, axis = 0)/600

w_avg0 = np.sum( w1[:,:,0], axis = 0)/600
w_avg1 = np.sum( w1[:,:,1], axis = 0)/600
w_avg2 = np.sum( w1[:,:,2], axis = 0)/600

plt.rcParams['figure.figsize'] = [20,4]

fig, (ax1, ax2, ax3) = plt.subplots(1,3)
ax1.plot(w1[1,:,0])
ax1.plot(w1[1,:,1])
ax1.plot(w1[1,:,2])
ax1.set_title('Coefficients')
ax1.set_xlabel('updates')
ax1.set_ylabel('weights')

ax2.plot(w_avg0)
ax2.plot(w_avg1)
ax2.plot(w_avg2)
ax2.set_title('Coefficients(averaged)')
ax2.set_xlabel('updates')
ax2.set_ylabel('averaged weights')

ax3.plot(mse)
ax3.set_title('Learning Curve for 3dB, n = 0.15')
ax3.set_xlabel('Updates')
ax3.set_ylabel('MSE')
