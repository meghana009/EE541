#Question 3c

x_mm = np.asarray(data['mismatched_x'])
v_mm = np.asarray(data['mismatched_v'])
y_mm = np.asarray(data['mismatched_y'])

#For n = 0.01
wn = np.array(wn)
y = np.zeros((600,501))
sum = np.zeros((3,1))
sum1 = 0
mse = np.zeros((600,501))

vx = v_mm.T
y2 = np.zeros((600,501))

for i in range(600):
  wn = np.array([0, 0, 0])

  for j in range(501):
    y[i][j] = np.matmul(wn.T,v_mm[i][j])
    wn = wn + 0.01*(y_mm[i][j]-y[i][j])*v_mm[i][j]
  Rv = (1/600)*(np.matmul(v_mm[i].T,v_mm[i]))
  rvy = (1/600)*(np.matmul(v_mm[i].T,y[i]))
  w = np.matmul(rvy, inv(Rv))
  y2[i] = np.matmul(v_mm[i],w)
    
mse = np.sum((y_mm - y)**2, axis = 0)/600

plt.plot(mse)
plt.xlabel('Updates')
plt.ylabel('MSE')
plt.title('Average Learning Rate for n = 0.01')
wllse = np.sum((y_mm - y2)**2, axis = 0)/600
plt.plot(wllse)
plt.legend(['MSE', 'LLSE'])
print(wllse)
print(Rv)
print(rvy)

#For n = 0.04

mse = np.zeros((600,501))
vx = v_mm.T
y2 = np.zeros((600,501))

for i in range(600):
  wn = np.array([0, 0, 0])
  for j in range(501):
    y[i][j] = np.matmul(wn.T,v_mm[i][j])
    wn = wn + 0.04*(y_mm[i][j]-y[i][j])*v_mm[i][j]
  Rv = (1/600)*(np.matmul(v_mm[i].T,v_mm[i]))
  rvy = (1/600)*(np.matmul(v_mm[i].T,y[i]))
  w = np.matmul(rvy, inv(Rv))
  y2[i] = np.matmul(v_mm[i],w)
    
print(Rv)
print(rvy)

mse = np.sum((y_mm - y)**2, axis = 0)/600
wllse = np.sum((y_mm - y2)**2, axis = 0)/600
print(wllse)

plt.plot(mse)
plt.xlabel('Updates')
plt.ylabel('MSE')
plt.title('Average Learning Rate for n = 0.04')
plt.plot(wllse)
plt.legend(['MSE', 'LLSE'])

#For n = 0.05

mse = np.zeros((600,501))
vx = v_mm.T
y2 = np.zeros((600,501))

for i in range(600):
  wn = np.array([0, 0, 0])
  for j in range(501):
    y[i][j] = np.matmul(wn.T,v_mm[i][j])
    wn = wn + 0.05*(y_mm[i][j]-y[i][j])*v_mm[i][j]
  Rv = (1/600)*(np.matmul(v_mm[i].T,v_mm[i]))
  rvy = (1/600)*(np.matmul(v_mm[i].T,y[i]))
  w = np.matmul(rvy, inv(Rv))
  y2[i] = np.matmul(v_mm[i],w)
    
print(Rv)
print(rvy)

mse = np.sum((y_mm - y)**2, axis = 0)/600
wllse = np.sum((y_mm - y2)**2, axis = 0)/600
print(wllse)

plt.plot(mse)
plt.xlabel('Updates')
plt.ylabel('MSE')
plt.title('Average Learning Rate for n = 0.05')
plt.plot(wllse)
plt.legend(['MSE', 'LLSE'])

#For n = 0.07

mse = np.zeros((600,501))
vx = v_mm.T
y2 = np.zeros((600,501))

for i in range(600):
  wn = np.array([0, 0, 0])
  for j in range(501):
    y[i][j] = np.matmul(wn.T,v_mm[i][j])
    wn = wn + 0.07*(y_mm[i][j]-y[i][j])*v_mm[i][j]
  Rv = (1/600)*(np.matmul(v_mm[i].T,v_mm[i]))
  rvy = (1/600)*(np.matmul(v_mm[i].T,y[i]))
  w = np.matmul(rvy, inv(Rv))
  y2[i] = np.matmul(v_mm[i],w)
    
print(Rv)
print(rvy)

mse = np.sum((y_mm - y)**2, axis = 0)/600
wllse = np.sum((y_mm - y2)**2, axis = 0)/600
print(wllse)

plt.plot(mse)
plt.xlabel('Updates')
plt.ylabel('MSE')
plt.title('Average Learning Rate for n = 0.07')
plt.plot(wllse)
plt.legend(['MSE', 'LLSE'])

#For n = 0.15

mse = np.zeros((600,501))
vx = v_mm.T
y2 = np.zeros((600,501))

for i in range(600):
  wn = np.array([0, 0, 0])
  for j in range(501):
    y[i][j] = np.matmul(wn.T,v_mm[i][j])
    wn = wn + 0.15*(y_mm[i][j]-y[i][j])*v_mm[i][j]
  Rv = (1/600)*(np.matmul(v_mm[i].T,v_mm[i]))
  rvy = (1/600)*(np.matmul(v_mm[i].T,y[i]))
  w = np.matmul(rvy, inv(Rv))
  y2[i] = np.matmul(v_mm[i],w)
    
print(Rv)
print(rvy)

mse = np.sum((y_mm - y)**2, axis = 0)/600
wllse = np.sum((y_mm - y2)**2, axis = 0)/600

print(wllse)

plt.plot(mse)
plt.xlabel('Updates')
plt.ylabel('MSE')
plt.title('Average Learning Rate for n = 0.15')
plt.plot(wllse)
plt.legend(['MSE', 'LLSE'])
