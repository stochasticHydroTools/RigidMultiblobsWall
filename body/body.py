'''
Small class to handle a single body. The notation follows
loosely the paper Brownian dynamics of confined rigid
bodies, Steven Delong et al. The Journal of Chemical
Physics 143, 144107 (2015). doi: 10.1063/1.4932062
'''
import numpy as np
from quaternion_integrator.quaternion import Quaternion 

class Body(object):
  '''
  Small class to handle a single body.
  '''  
  def __init__(self, location, orientation, reference_configuration, blob_radius):
    '''
    Constructor. Take arguments like ...
    '''
    # Location as np.array.shape = 3
    self.location = location
    # Orientation as Quaternion
    self.orientation = orientation
    # Number of blobs
    self.Nblobs = reference_configuration.size / 3
    # Reference configuration. Coordinates of blobs for quaternion [1, 0, 0, 0]
    # and location = np.array[0, 0, 0]) as a np.array.shape = (Nblobs, 3) 
    # or np.array.shape = (Nblobs * 3)
    self.reference_configuration = np.reshape(reference_configuration, (self.Nblobs, 3))
    # Blob radius
    self.blob_radius = blob_radius
    # Name of body and type of body. A string or number
    self.name = None
    self.type = None
    self.mobility_blobs = None
    self.mobility_body = None
    # Geometrix matrix K (see paper Delong et al. 2015).
    self.K = None
    self.rotation_matrix = None
    # Some default functions
    self.function_slip = self.default_zero_blobs
    self.function_force = self.default_none
    self.function_torque = self.default_none
    self.function_force_blobs = self.default_zero_blobs
    
    
  
  def get_r_vectors(self):
    '''
    Return the coordinates of the blobs.
    '''
    rotation_matrix = self.orientation.rotation_matrix()
    r_vectors = np.empty([self.Nblobs, 3])
    for i, vec in enumerate(self.reference_configuration):
      r_vectors[i] = np.dot(rotation_matrix, vec) + self.location
    return r_vectors


  def calc_rot_matrix(self):
    ''' 
    Calculate the matrix R, where the i-th 3x3 block of R gives
    (R_i x) = -1 (r_i cross x).
    R has shape (3*Nblobs, 3).
    '''
    rot_matrix = np.empty((self.Nblobs, 3, 3))
    for k, vec in enumerate(self.reference_configuration):
      # Create block
      block = np.array([[0.0, vec[2], -1.0 * vec[1]],
                        [-1.0 * vec[2], 0.0, vec[0]],
                        [vec[1], -1.0 * vec[0], 0.0]])
      # Assign block
      rot_matrix[k] = block
    return np.reshape(rot_matrix, (3*self.Nblobs, 3))


  def calc_J_matrix(self):
    '''
    Returns a block matrix with dimensions (Nblobs, 1)
    with each block being a 3x3 identity matrix.
    '''
    J = np.empty([self.Nblobs, 3, 3])
    for i in range(self.Nblobs):
      J[i] = np.eye(3)
    return np.reshape(J, (3*self.Nblobs, 3))

  def calc_K_matrix(self):
    '''
    Return geometric matrix K = [J, rot] with shape (3*Nblobs, 6)
    '''
    return np.concatenate([self.calc_J_matrix(), self.calc_rot_matrix()], axis=1)
                        

  def check_function(self, distance = None):
    ''' 
    Function to check that the body didn't cross the wall,
    i.e., all its blobs have z > distance. Default distance is blob_radius.
    '''
    # Define distance
    if not distance:
      distance = self.blob_radius

    # Get current configuration
    r_vectors = self.get_r_vectors()

    # Loop over blobs
    for vec in r_vectors:
      if vec[2] < distance:
        return False
    return True


  def calc_slip(self):
    '''
    Return the slip on the blobs.
    '''
    return self.function_slip()


  def calc_force(self):
    '''
    Return the force on the body.
    '''
    return self.function_force()


  def calc_torque(self):
    '''
    Return the torque on the body.
    '''
    return self.function_torque()


  def calc_force_blobs(self):
    '''
    Return the force on the blobs.
    '''
    return self.function_force_blobs()
  
  def default_zero_blobs(self):
    return np.zeros((self.Nblobs, 3))

  def default_none(self):
    return None


