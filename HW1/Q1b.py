import sys
import random 
import matplotlib.pyplot as plt
import math

#Question 1b for n =20
N = 50
p = 0.7
number_of_heads = 0
number_of_cont_heads = 0
n = 20
freq = []

for j in range(n):
  for i in range(1, N+1):
    toss = random.random()
    if toss < p:
      number_of_heads += 1
  freq.append(number_of_heads)
  number_of_heads = 0

plt.hist(freq)

plt.ylabel('Number of Experiment')
plt.xlabel('Number of Heads');



