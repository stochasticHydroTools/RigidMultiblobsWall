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
from quaternion_integrator.quaternion import Quaternion

# Parameters
A = 0.2625  # Radius of blobs in um
ETA = 1.0  # This needs to be changed to match the paper in um, s, etc.

# Made these up for now.
M = [0.1/7. for _ in range(7)] 
KT = 0.2
REPULSION_STRENGTH = 2.0
DEBYE_LENGTH = 0.15


def boomerang_mobility(locations, orientations):
  ''' 
  Calculate the force and torque mobility for the
  boomerang.  Here location is the cross point.
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
  J = np.concatenate([np.identity(3) for _ in range(7)])
  J_rot_combined = np.concatenate([J, rotation_matrix], axis=1)
  total_mobility = np.linalg.inv(np.dot(J_rot_combined.T,
                                        np.dot(np.linalg.inv(mobility),
                                               J_rot_combined)))
  return total_mobility


def get_boomerang_r_vectors(location, orientation):
  '''Get the vectors of the 7 blobs used to discretize the boomerang.
  
         1 2 3 4    
         O-O-O-O
               O 5
               O 6
               O 7
   
  The location is the location of the Blob at the apex.. 
  Initial configuration is in the
  x-y plane, with  arm 1-2-3  pointing in the positive x direction, and arm
  4-5-6 pointing in the positive y direction.
  Seperation between blobs is currently hard coded at 0.525 um
  '''
    
  initial_configuration = [np.array([1.575, 0., 0.]),
                           np.array([1.05, 0., 0.]),
                           np.array([0.525, 0., 0.]),
                           np.array([0., 0., 0.]),
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


def boomerang_force_calculator(location, orientation):
  ''' 
  Calculate force exerted on the boomerang given 
  it's location and orientation.
  location - list of length 1 with location of tracking point of 
             boomerang.
  orientation - list of length 1 with orientation (as a Quaternion)
                of boomerang.
  '''
  gravity = [0., 0., -1.*sum(M)]
  h = location[0][2]
  repulsion = np.array([0., 0., 
                        (REPULSION_STRENGTH*((h - A)/DEBYE_LENGTH + 1)*
                         np.exp(-1.*(h - A)/DEBYE_LENGTH)/
                         ((h - A)**2))])
  return repulsion + gravity


def boomerang_torque_calculator(location, orientation):
  ''' 
  Calculate torque based on Boomerang location and orientation.
  location - list of length 1 with location of tracking point of 
             boomerang.
  orientation - list of length 1 with orientation (as a Quaternion)
                of boomerang.
  '''
  r_vectors = get_boomerang_r_vectors(location[0], orientation[0])
  forces = []
  for mass in M:
    forces += [0., 0., -1.*mass]
  R = calc_rot_matrix(r_vectors, location[0])
  return np.dot(R.T, forces)

def generate_boomerang_equilibrium_sample():
  ''' 
  Use accept-reject to generate a sample
  with location and orientation from the Gibbs Boltzmann 
  distribution for the Boomerang.
  '''
  while True:
    theta = np.random.normal(0., 1., 4)
    orientation = Quaternion(theta/np.linalg.norm(theta))
    location = [0., 0., np.random.uniform(A, 10.0)]
    accept_prob = boomerang_gibbs_boltzmann_distribution(location, orientation)/7.7e-1
    if accept_prob > 1.:
      print 'Accept probability %s is greater than 1' % accept_prob
    
    if np.random.uniform(0., 1.) < accept_prob:
      return [location, orientation]


def boomerang_gibbs_boltzmann_distribution(location, orientation):
  ''' Return exp(-U/kT) for the given location and orientation.'''
  r_vectors = get_boomerang_r_vectors(location, orientation)
  # Add gravity to potential.
  U = 0
  for k in range(7):
    U += M[k]*r_vectors[k][2]
  # Add repulsion to potential.
  U += (REPULSION_STRENGTH*np.exp(-1.*(location[2] -A)/DEBYE_LENGTH)/
        (location[2] - A))

  return np.exp(-1.*U/KT)
  
  
  
  


  
