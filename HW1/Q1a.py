import sys
import random 
import matplotlib.pyplot as plt
import math

#Question 1a
N = 50
p = 0.7
number_of_heads = 0
number_of_cont_heads = 0
n = 100
freq = []
c = 0

for i in range(1, N+1):
  toss = random.random()
  if toss < p:
    number_of_heads += 1
    c += 1
  else:
    number_of_heads = 0
  number_of_cont_heads = max(number_of_cont_heads, number_of_heads)
  freq.append(number_of_cont_heads)
  num_of_heads = 0
  num_cont_of_heads = 0
print('The longest run of obtaining heads is:', max(freq))
print('The total number of times head was obtained when the biased coin was tossed is:',c)
