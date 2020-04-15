'''
Small module to read the slip file of a rigid bodies.
'''
import numpy as np

def read_slip_file(name_file):
  '''
  It reads a prescribed velocity for a rigid body 
  and returns the velocity as a numpy array with shape (Nbodies, 6).
  '''
  comment_symbols = ['#']   
  prescribed_velocity = []
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
          Nbodies = int(line.split()[0])
        else:
          data = line.split()
          velocity = [float(data[0]), float(data[1]), float(data[2]), float(data[3]), float(data[4]), float(data[5])]
          prescribed_velocity.append(velocity)
        i += 1

  prescribed_velocity = np.array(prescribed_velocity)
  return prescribed_velocity

