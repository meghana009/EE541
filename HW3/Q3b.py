#Question 3b
tv_c = np.asarray(data['timevarying_coefficents'])
tv_x = np.asarray(data['timevarying_x'])
tv_y = np.asarray(data['timevarying_y'])
tv_z = np.asarray(data['timevarying_z'])
tv_v = np.asarray(data['timevarying_v'])

wn = tv_c[0]
y = np.zeros((501))
mse = np.zeros((600,501))

tv = [] 
  
for j in range(501):
  y[j] = np.matmul(wn.T,tv_v[j])
  wn = wn + 0.08*(tv_z[j]-y[j])*tv_v[j]
  tv.append(wn)
tv = np.array(tv)

plt.rcParams['figure.figsize'] = [5,40]

fig, (ax1, ax2, ax3, ax4, ax5, ax6) = plt.subplots(6,1)
ax1.plot(n,tv_c[:,0])
ax2.plot(n,tv_c[:,1])
ax3.plot(n,tv_c[:,2])

ax1.set_title('True Time Varying Coefficient-1')
ax1.set_xlabel('Time n')
ax1.set_ylabel('True Time Varying Coefficient-1')

ax2.set_title('True Time Varying Coefficient-2')
ax2.set_xlabel('Time n')
ax2.set_ylabel('True Time Varying Coefficient-2')

ax3.set_title('True Time Varying Coefficient-3')
ax3.set_xlabel('Time n')
ax3.set_ylabel('True Time Varying Coefficient-3')


ax4.plot(n,tv[:,0])
ax5.plot(n,tv[:,1])
ax6.plot(n,tv[:,2])

ax4.set_title('Estimated Time Varying Coefficient-1')
ax4.set_xlabel('Time n')
ax4.set_ylabel('Estimated Time Varying Coefficient-1')

ax5.set_title('Estimated Time Varying Coefficient-2')
ax5.set_xlabel('Time n')
ax5.set_ylabel('Estimated Time Varying Coefficient-2')

ax6.set_title('Estimated Time Varying Coefficient-3')
ax6.set_xlabel('Time n')
ax6.set_ylabel('Estimated Time Varying Coefficient-3')

plt.rcParams['figure.figsize'] = [5,18]

fig, (ax1, ax2, ax3) = plt.subplots(3,1)

ax1.plot(tv_c[:,0], Label = 'True Coefficients')
ax1.plot(tv[:,0], Label = 'Estimated Coefficients')
ax1.set_title('True Time Varying Coefficient-1 vs Estimated Coefficient-1')
ax1.set_xlabel('Updates')
ax1.set_ylabel('Coefficients')

ax2.plot(tv_c[:,1], Label = 'True Coefficients')
ax2.plot(tv[:,1], Label = 'Estimated Coefficients')
ax2.set_title('True Time Varying Coefficient-2 vs Estimated Coefficient-2')
ax2.set_xlabel('Updates')
ax2.set_ylabel('Coefficients')


ax3.plot(tv_c[:,2], Label = 'True Coefficients')
ax3.plot(tv[:,2], Label = 'Estimated Coefficients')
ax3.set_title('True Time Varying Coefficient-3 vs Estimated Coefficient-3')
ax3.set_xlabel('Updates')
ax3.set_ylabel('Coefficients')
