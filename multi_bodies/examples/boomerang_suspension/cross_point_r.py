'''
Use trajectory file to write file with columns sqrt(x**2 + y**2) and z.

How to use:
python blob_r_z.py file_name num_blobs
'''

import numpy as np
import sys


if __name__ == '__main__':
  # Read input
  name = sys.argv[1]
  num_bodies = int(sys.argv[2])

  x = np.zeros(2)
  y = np.zeros(2)
  z = np.zeros(2)

  with open(name, 'r') as f:
    count = num_bodies
    for line in f:
      if count == num_bodies:
        count = 0
      else:
        data = line.split()
        x[count] = float(data[0])
        y[count] = float(data[1])
        z[count] = float(data[2])
        count += 1
      if count == (num_bodies):
        print np.sqrt((x[0]-x[1])**2 + (y[0]-y[1])**2 + (z[0]-z[1])**2)
        
