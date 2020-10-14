'''
Small module to read the slip file of a rigid bodies.
'''

import numpy as np

def read_slip_file(name_file):
  '''
  It reads a slip file of a rigid bodies and return
  the slip as a numpy array with shape (Nblobs, 3).

  This is the slip in the reference configuration of the
  body, quaternion=(1,0,0,0). The code will rotate the
  slip with the orientation of the body in each step.
  '''
  comment_symbols = ['#']   
  slip = []
  with open(name_file, 'r') as f:
    i = 0
    for line in f:
      # Strip comments
      if comment_symbols[0] in line:
        line, comment = line.split(comment_symbols[0], 1)

      # Ignore blank lines
      line = line.strip()
      if line != '':
        if i == 0:
          Nblobs = int(line.split()[0])
        else:
          data = line.split()
          slip_blob = [float(data[0]), float(data[1]), float(data[2])]
          slip.append(slip_blob)
        i += 1

  slip = np.array(slip)
  return slip

