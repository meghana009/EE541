#Question 1c

import sys
import random 
import matplotlib.pyplot as plt
import math

N = 500
p = 0.7
number_of_heads = 0
number_of_cont_heads = 0
freq = []

for i in range(1, N+1):
  toss = random.random()
  if toss < p:
    number_of_heads += 1
  freq.append(number_of_heads)
    

plt.hist(freq)
plt.ylabel('Number of Heads')
plt.xlabel('Number of Trials');
