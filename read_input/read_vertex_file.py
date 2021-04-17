'''
Small module to read a vertex file of the rigid bodies.
'''

import numpy as np

def read_vertex_file(name_file):
  '''
  It reads a vertex file of the rigid bodies and return
  the coordinates as a numpy array with shape (Nblobs, 3).
  '''
  comment_symbols = ['#']   
  coor = []
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
          location = np.fromstring(line, sep=' ')
          coor.append(location)
        i += 1

  coor = np.array(coor)
  return coor

