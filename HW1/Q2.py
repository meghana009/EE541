import sys
import random 
from statistics import mean
import matplotlib.pyplot as plt

#For 100 realizations of N
sum = 0
count = 0
l = []
N = []
E = []

while (len(N)<=100):
  samp = random.uniform(0,1)
  sum = sum + samp
  count = count + 1
  if (sum >= 4):
    l.append(count)
    N.append(min(l))
    sum = 0
    count = 0
E = mean(N)

print('The expected value of N is', E)

plt.hist(N)
plt.xlabel('Value of N')
plt.ylabel('Number of Realizations of N')
