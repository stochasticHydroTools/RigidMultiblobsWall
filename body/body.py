'''
Small class to handle a single body.
'''
import numpy as np
from quaternion_integrator.quaternion import Quaternion 

class Body(object):
  '''
  Small class to handle a single body.
  '''  
  def __init__(self, name, location, orientation, reference_configuration, a):
    '''
    Constructor. Take arguments like ...
    '''
    # Name a string or number
    self.name = name
    # Location as np.array.shape = 3
    self.location = location
    # Orientation as Quaternion
    self.orientation = orientation
    # Reference configuration. Coordinates of blobs for quaternion [1, 0, 0, 0]
    # as np.array.shape = (Nblobs, 3) or np.array.shape = (Nblobs * 3)
    self.reference_configuration = reference_configuration
    # Number of blobs
    self.Nblobs = self.reference_configuration.size
    # Blob radius
    self.a = a

    
  

