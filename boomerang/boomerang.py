'''
Set up the mobility, torque, and force functions for the Boomerang
from:
"Chakrabarty et. al - Brownian Motion of Boomerang Colloidal
Particles"
'''

import numpy as np
import sys
sys.path.append('..')

from fluids import mobility as mb

# Parameters
A = 0.2625  # Radius of blobs in um
ETA = 1.0  # This needs to be changed to match the paper in um, s, etc.



def boomerang_mobility(locations, orientations):
  ''' 
  Calculate the force and torque mobility for the
  boomerang.
  '''
  r_vectors = get_boomerang_r_vectors(locations[0], orientations[0])
  return force_and_torque_boomerang_mobility(r_vectors, locations[0])


def force_and_torque_boomerang_mobility(r_vectors, location):
  '''
  Calculate the mobility: (torque, force) -> (angular velocity, velocity) at position 
  In this case, position has orientation and location data, each of length 1.
  The mobility is equal to the inverse of: 
    [ J^T M^-1 J,   J^T M^-1 R ]
    [ R^T M^-1 J,   R^T M^-1 R ]
  where R is 3N x 3 (18 x 3) Rx = r cross x and J is a 3N x 3 matrix with 
  each 3x3 block being the identity.
  r is the distance from the apex blob of the boomerang to
  each other blob (a length 3N vector).
  M (3N x 3N) is the finite size single wall mobility taken from the
  Swan and Brady paper:
   "Simulation of hydrodynamically interacting particles near a no-slip
    boundary."
  Here location is the dereferenced list with 3 entries.
  '''  
  mobility = mb.boosted_single_wall_fluid_mobility(r_vectors, ETA, A)
  rotation_matrix = calc_rot_matrix(r_vectors, location)
  J = np.concatenate([np.identity(3) for _ in range(6)])
  J_rot_combined = np.concatenate([J, rotation_matrix], axis=1)
  total_mobility = np.linalg.inv(np.dot(J_rot_combined.T,
                                        np.dot(np.linalg.inv(mobility),
                                               J_rot_combined)))
  return total_mobility

def get_boomerang_r_vectors(location, orientation):
  '''Get the vectors of the 7 blobs used to discretize the boomerang.
  
         1 2 3     
         O-O-O-O
               O 4
               O 5
               O 6
   
  The location is the location of the Blob at the apex.. 
  Initial configuration is in the
  x-y plane, with  arm 1-2-3  pointing in the positive x direction, and arm
  4-5-6 pointing in the positive y direction.
  Seperation between blobs is currently hard coded at 0.525 um
  '''
    
  initial_configuration = [np.array([1.575, 0., 0.]),
                           np.array([1.05, 0., 0.]),
                           np.array([0.525, 0., 0.]),
                           np.array([0., 0.525, 0.]),
                           np.array([0., 1.05, 0.]),
                           np.array([0., 1.575, 0.])]

  rotation_matrix = orientation.rotation_matrix()
  rotated_configuration = []
  for vec in initial_configuration:
    rotated_configuration.append(np.dot(rotation_matrix, vec)
                                 + np.array(location))

  return rotated_configuration


def calc_rot_matrix(r_vectors, location):
  ''' 
  Calculate the matrix R, where the i-th 3x3 block of R gives
  (R_i x) = r_i cross x.
  R will be 3N by 3 (18 x 3). The r vectors point from the center
  of the icosohedron to the other vertices.
  '''
  rot_matrix = None
  for k in range(len(r_vectors)):
    # Here the cross is relative to the center.
    adjusted_r_vector = r_vectors[k] - location
    block = np.array(
        [[0.0, -1.*adjusted_r_vector[2], adjusted_r_vector[1]],
        [adjusted_r_vector[2], 0.0, -1.*adjusted_r_vector[0]],
        [-1.*adjusted_r_vector[1], adjusted_r_vector[0], 0.0]])
    if rot_matrix is None:
      rot_matrix = block
    else:
      rot_matrix = np.concatenate([rot_matrix, block], axis=0)
  return rot_matrix
