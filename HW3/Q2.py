#Question 2

## copyright, Keith Chugg
##  EE599, 2020

#################################################
## this is a template to illustrate hd5 files
##
## also can be used as template for HW1 problem
##################################################

import h5py
import numpy as np
import matplotlib.pyplot as plt 

DEBUG = False
DATA_FNAME = 'meghana_hw3_2.hd5'

if DEBUG:
    num_sequences = 3
    sequence_length = 4
else:
    num_sequences = 25
    sequence_length = 20

### Enter your data here...
### Be sure to generate the data by hand.  DO NOT:
###     copy-n-paste
###     use a random number generator 
###
x_list = [[0,0,0,0,0,1,1,1,1,1,1,1,1,1,0,0,0,1,1,0],
          [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
          [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
          [1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0],
          [0,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,1],
          [1,0,0,1,0,0,1,0,0,0,1,0,0,0,1,0,0,0,1,0],
          [1,1,0,0,1,1,0,0,1,1,0,0,1,1,0,0,1,1,0,0],
          [0,0,1,1,0,0,1,1,0,0,1,1,0,0,1,1,0,0,1,1],
          [1,1,0,0,1,1,1,1,1,1,0,0,0,1,1,1,0,0,1,0],
          [0,0,0,1,1,0,0,0,1,1,1,1,1,0,0,0,0,0,0,0],
          [1,0,0,0,1,1,0,0,1,1,0,0,1,1,0,0,1,1,0,0],
          [1,1,1,1,1,1,0,0,0,0,0,0,1,1,1,1,1,1,1,1],
          [0,0,0,0,1,1,1,1,0,0,0,1,1,1,1,1,0,0,0,1],
          [1,1,1,0,0,1,1,1,0,0,1,1,1,0,0,1,1,1,0,0],
          [1,1,1,1,1,0,0,0,0,0,1,1,1,1,1,0,0,0,0,0],
          [1,1,1,0,1,1,1,0,1,1,1,0,1,1,1,0,1,1,1,0],
          [0,0,1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,1,0,0],
          [1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0],
          [1,1,0,0,1,1,0,0,0,0,0,0,1,1,1,1,0,0,1,1],
          [0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1],
          [0,0,1,1,0,0,1,1,1,1,0,0,0,1,1,1,0,0,0,1],
          [0,1,1,1,1,0,0,0,0,0,1,1,1,0,0,0,1,1,0,0],
          [0,0,0,1,1,1,0,0,0,1,1,1,0,0,0,1,1,1,0,0],
          [1,1,1,1,1,1,0,1,1,1,1,1,1,0,1,1,1,1,1,1],
          [0,1,1,1,0,1,1,1,1,1,1,1,1,0,0,1,1,1,0,0]
          ]
# x_list = [
#     [ 0, 1, 1, 0],
#     [ 1, 1, 0, 0],
#     [ 0, 0, 0, 1]
# ]

# convert list to a numpy array...
human_binary = np.asarray(x_list)

### do some error trapping:

assert human_binary.shape[0] == num_sequences, 'Error: the number of sequences was entered incorrectly'
assert human_binary.shape[1] == sequence_length, 'Error: the length of the seqeunces is incorrect'

# the with statement opens the file, does the business, and close it up for us...
with h5py.File(DATA_FNAME, 'w') as hf:
    hf.create_dataset('human_binary', data = human_binary)
    ## note you can write several data arrays into one hd5 file, just give each a different name.

###################
# Let's read it back from the file and then check to make sure it is as we wrote...
with h5py.File(DATA_FNAME, 'r') as hf:
    hb_copy = hf['human_binary'][:]

### this will throw and error if they are not the same...
np.testing.assert_array_equal(human_binary, hb_copy)
