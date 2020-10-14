'''
Extract one body trajectory from configurational files.

How to use:
python get_body.py file_name num_bodies body_number dt
'''

import numpy as np
import sys


if __name__ == '__main__':
  # Read input
  name = sys.argv[1]
  num_bodies = int(sys.argv[2])
  body = int(sys.argv[3])
  dt = float(sys.argv[4])

  with open(name, 'r') as f:
    count = num_bodies
    step = -1
    for line in f:
      if count == num_bodies:
        step += 1
        time = step * dt
        count = 0
      else:
        if count == body:
          data = line.split()
          print(time, line.strip())
        count += 1
