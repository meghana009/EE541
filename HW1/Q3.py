# -*- coding: utf-8 -*-
"""homework3.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1DHI2ahPy-_MACiDR6RMRRAJ-3va_1HeQ
"""

import sys
from func import f

#Taking input from command line
x0 = (sys.argv[1])
x1 = (sys.argv[2])

#defining the secant function
def secant(x0, x1):
  #Initialising the number of iterations required
  count = 0
  
  #Checking the Bolzano's criterion
  while (abs(x1 - x0) > 10e-8):
    count = count + 1
    denominator = f(x1) - f(x0)
    xn = x1 - ((x1 - x0)/denominator)*f(x1)
    x0 = x1
    x1 = xn
  
  N = count

  #printing the output to each line
  sys.stdout.write(str(N) + '\n')
  sys.stdout.write(str(x0) + '\n')
  sys.stdout.write(str(x1) + '\n')
  sys.stdout.write(str(xn))

#To check if the input received is a number or not
try:
    x0 = float(x0)
    x1 = float(x1)
    if(x0 == float(x0) and x1== float(x1)):
        if ((x0 < x1)and((f(x0)*f(x1))<0)):
            
            #calling the function
            secant(x0,x1)
        else:
            sys.stderr.write('Range Error')

except ValueError:
    sys.stderr.write('Range Error')

    






