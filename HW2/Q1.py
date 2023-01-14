#Q1.
import numpy as np
from scipy import interpolate
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import sys
from scipy.interpolate import UnivariateSpline

#Loading the dataset raman.txt and converting it to a numpy array
dt = np.loadtxt("raman.txt", dtype=float)
x=[]
y=[]
for i in range(len(dt)):
  x.append(dt[i][0])
  y.append(dt[i][1])
x = np.array(x)
y = np.array(y)

#Raman Spectroscopy Peak Calculation for raw data
peaks1 = find_peaks(y, height = 580, threshold = 6.4, distance = 360)
heights = peaks1[1]['peak_heights']
pos = x[peaks1[0]]

#Sorting the peaks with the respective wave number
heights1, pos2 = (list(x) for x in zip(*sorted(zip(heights, pos), reverse = True)))
#Printing the wavenumbers corresponding to the 8 peaks
for i in range(8):
  sys.stdout.write(str(pos2[i]) + " ")

#Plotting the Peaks using find_peaks
fig = plt.figure()
ax = fig.subplots()
ax.plot(x,y)
ax.scatter(pos, heights, color = 'r', s = 15,marker = 'X', Label = 'Peaks')
plt.xlabel("Wave Number")
plt.ylabel("Intensity")
ax.legend()
ax.grid()
plt.show()

#Using arange function to get evenly spaced wavenumbers on the x-axis
x_fit = np.arange(x[620],x[650])

#Using splrep to define the splines
splines = interpolate.splrep(x,y, s = 0.25, k = 3)

#Using the splev to calculate the new intensity using the tuple returned by splrep and the x_fit
y2 = interpolate.splev(x_fit, splines)

x1 = []
y1 = []
for i in range(620,650):
  x1.append(x[i])
  y1.append(y[i])
x1 = np.array(x1)
y1 = np.array(y1)

peaks = find_peaks(y2, height = 100, threshold = 6.4, distance = 100)
heights1 = peaks[1]['peak_heights'][0]
pos1 = x_fit[peaks[0][0]]


fig = plt.figure()
ax = fig.subplots()
ax.plot(x_fit,y2,Label = 'Interpolated Line')
ax.plot(pos1,heights1, Label = 'Maximum Intensity', Marker = 'X')

ax.scatter(x1, y1, color = 'blue', s = 15,marker = 'X', Label = 'Original Points')
plt.xlabel("Wave Number")
plt.ylabel("Intensity")


ax.legend()
ax.grid()

plt.show()

#Using arange function to get evenly spaced wavenumbers on the x-axis
x_fit = np.arange(x[1890],x[1920])

#Using splrep to define the splines
splines = interpolate.splrep(x,y, s = 0.25, k = 3)

#Using the splev to calculate the new intensity using the tuple returned by splrep and the x_fit
y2 = interpolate.splev(x_fit, splines)

x1 = []
y1 = []
for i in range(1890,1920):
  x1.append(x[i])
  y1.append(y[i])
x1 = np.array(x1)
y1 = np.array(y1)

peaks = find_peaks(y2, height = 100, threshold = 6.4, distance = 100)
heights1 = peaks[1]['peak_heights'][0]
pos1 = x_fit[peaks[0][0]]


fig = plt.figure()
ax = fig.subplots()
ax.plot(x_fit,y2,Label = 'Interpolated Line')
ax.plot(pos1,heights1, Label = 'Maximum Intensity', Marker = 'X')

ax.scatter(x1, y1, color = 'blue', s = 15,marker = 'X', Label = 'Original Points')

plt.xlabel("Wave Number")
plt.ylabel("Intensity")
ax.legend()
ax.grid()

plt.show()

#Using arange function to get evenly spaced wavenumbers on the x-axis
x_fit = np.arange(x[1340],x[1365])

#Using splrep to define the splines
splines = interpolate.splrep(x,y, s = 0.25, k = 3)

#Using the splev to calculate the new intensity using the tuple returned by splrep and the x_fit
y2 = interpolate.splev(x_fit, splines)

x1 = []
y1 = []
for i in range(1340,1365):
  x1.append(x[i])
  y1.append(y[i])
x1 = np.array(x1)
y1 = np.array(y1)

peaks = find_peaks(y2, height = 558, threshold = 6.4, distance = 100)
heights1 = peaks[1]['peak_heights'][0]
pos1 = x_fit[peaks[0][0]]


fig = plt.figure()
ax = fig.subplots()
ax.plot(x_fit,y2,Label = 'Interpolated Line')
ax.plot(pos1,heights1, Label = 'Maximum Intensity', Marker = 'X')

ax.scatter(x1, y1, color = 'blue', s = 15,marker = 'X', Label = 'Original Points')

plt.xlabel("Wave Number")
plt.ylabel("Intensity")
ax.legend()
ax.grid()

plt.show()

#Using arange function to get evenly spaced wavenumbers on the x-axis
x_fit = np.arange(x[2340],x[2370])

#Using splrep to define the splines
splines = interpolate.splrep(x,y, s = 0.25, k = 3)

#Using the splev to calculate the new intensity using the tuple returned by splrep and the x_fit
y2 = interpolate.splev(x_fit, splines)

x1 = []
y1 = []
for i in range(2340,2370):
  x1.append(x[i])
  y1.append(y[i])
x1 = np.array(x1)
y1 = np.array(y1)

peaks = find_peaks(y2, height = 558, threshold = 6.4, distance = 100)
heights1 = peaks[1]['peak_heights'][0]
pos1 = x_fit[peaks[0][0]]


fig = plt.figure()
ax = fig.subplots()
ax.plot(x_fit,y2,Label = 'Interpolated Line')
ax.plot(pos1,heights1, Label = 'Maximum Intensity', Marker = 'X')

ax.scatter(x1, y1, color = 'blue', s = 15,marker = 'X', Label = 'Original Points')

plt.xlabel("Wave Number")
plt.ylabel("Intensity")
ax.legend()
ax.grid()

plt.show()
