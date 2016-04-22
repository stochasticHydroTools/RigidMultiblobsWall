'''
Small module to read a file with the initial locations and orientation
of the rigid bodies.
'''
import numpy as np
import sys
sys.path.append('../')
from quaternion_integrator.quaternion import Quaternion

def read_clones_file(name_file):
  '''
  It reads a file with the initial locations and orientation
  of the rigid bodies.
  Input:
  name_file = string.
  Output:
  locations = locations of rigid bodies, numpy array shape (Nbodies, 3).
  orientations = orientations of rigid bodies, numpy array of Quaternions,
                 shape (Nbodies).
  '''
  comment_symbols = ['#']   
  with open(name_file, 'r') as f:
    locations = []
    orientations = []
    for line in f:
      # Strip comments
      if comment_symbols[0] in line:
        line, comment = line.split(comment_symbols[0], 1)

      # Ignore blank lines
      line = line.strip()
      if line != '':
        data = line.split()
        location = [float(data[0]), float(data[1]), float(data[2])]
        orientation = [float(data[3]), float(data[4]), float(data[5]), float(data[6])]
        q = Quaternion(orientation)
        locations.append(location)
        orientations.append(q)

    # Creat and return numpy arrays
    locations = np.array(locations)
    orientations = np.array(orientations)
    return locations, orientations
