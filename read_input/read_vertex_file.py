'''
Small module to read a vertex file of the rigid bodies.
'''
import numpy as np

def read_vertex_file(name_file):
  '''
  It reads a vertex file of the rigid bodies and return
  the coordinates as a numpy array with shape (Nblobs, 3).
  '''
  with open(name_file, 'r') as f:
    Nblobs = int(f.readline().split()[0])
    coor = np.empty((Nblobs, 3))
    for i in range(Nblobs):
      vec = f.readline().split()
      for j in range(3):
        coor[i, j] = float(vec[j])

  return coor

