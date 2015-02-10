''' Functions used for the Icosohedron structure near a wall. '''

import numpy as np
import os

from fluids import mobility as mb
from quaternion_integrator.quaternion import Quaternion


# Parameters
ETA = 1.0             # Viscosity.
A = 0.5               # 'Radius' of entire Icosohedron.
VERTEX_A = 0.05       # radius of individual vertices
M = [0.1/13. for _ in range(13)]  #Masses of particles
KT = 0.2              # Temperature

# Repulsion potential paramters.  Using Yukawa potential.
REPULSION_STRENGTH = 2.0
DEBYE_LENGTH = 0.5

# Make directory for logs if it doesn't exist.
if not os.path.isdir(os.path.join(os.getcwd(), 'logs')):
  os.mkdir(os.path.join(os.getcwd(), 'logs'))
# Make directory for figures if it doesn't exist.
if not os.path.isdir(os.path.join(os.getcwd(), 'figures')):
  os.mkdir(os.path.join(os.getcwd(), 'figures'))
# Make directory for data if it doesn't exist.
if not os.path.isdir(os.path.join(os.getcwd(), 'data')):
  os.mkdir(os.path.join(os.getcwd(), 'data'))

def icosohedron_mobility(location, orientation):
  ''' 
  Mobility for the rigid icosohedron, return a 6x6 matrix
  that takes Force + Torque and returns velocity and angular velocity.
  '''
  r_vectors = get_icosohedron_r_vectors(location[0], orientation[0])
  return force_and_torque_icosohedron_mobility(r_vectors, location[0])


def force_and_torque_icosohedron_mobility(r_vectors, location):
  '''
  Calculate the mobility: (torque, force) -> (angular velocity, velocity) at position 
  The mobility is equal to the inverse of: 
    [ J^T M^-1 J,   J^T M^-1 R ]
    [ R^T M^-1 J,   R^T M^-1 R ]
  where R is 3N x 3 (36 x 3) Rx = r cross x and J is a 3N x 3 matrix with 
  each 3x3 block being the identity.
  r is the distance from the center vertex of the icosohedron to
  each other vertex (a length 3N vector).
  M (3N x 3N) is the finite size single wall mobility taken from the
  Swan and Brady paper:
   "Simulation of hydrodynamically interacting particles near a no-slip
    boundary."
  Here location is the dereferenced list with 3 entries.
  '''  
  mobility = mb.boosted_single_wall_fluid_mobility(r_vectors, ETA, VERTEX_A)
  rotation_matrix = calc_icosohedron_rot_matrix(r_vectors, location)
  J = np.concatenate([np.identity(3) for _ in range(12)])
  J_rot_combined = np.concatenate([J, rotation_matrix], axis=1)
  total_mobility = np.linalg.inv(np.dot(J_rot_combined.T,
                                        np.dot(np.linalg.inv(mobility),
                                               J_rot_combined)))
  return total_mobility


def get_icosohedron_r_vectors(location, orientation):
  ''' Get the locations of each individual vertex of the icosohedron. '''
  # These values taken from an IBAMR vertex file. 'Radius' of 
  # Entire structure is ~1.
  initial_setup = [np.array([0.276393, 0.850651, 0.447214]),
                   np.array([1e-12, 1e-12, 1]),
                   np.array([-0.723607, 0.525731, 0.447214]),
                   np.array([0.276393, -0.850651, 0.447214]),
                   np.array([-0.276393, -0.850651, -0.447214]),
                   np.array([-0.723607, -0.525731, 0.447214]),
                   np.array([-0.276393, 0.850651, -0.447214]),
                   np.array([-0.894427, 1.00011e-12, -0.447214]),
                   np.array([0.723607, -0.525731, -0.447214]),
                   np.array([0.723607, 0.525731, -0.447214]),
                   np.array([0.894427, 9.99781e-13, 0.447214]),
                   np.array([1e-12, 1e-12, -1])]
  
  rotation_matrix = orientation.rotation_matrix()

  # TODO: Maybe don't do this on the fly every single time.
  for k in range(len(initial_setup)):
    initial_setup[k] = A*(initial_setup[k])

  rotated_setup = []
  for r in initial_setup:
    rotated_setup.append(np.dot(rotation_matrix, r) + np.array(location))
    
  return rotated_setup


def calc_icosohedron_rot_matrix(r_vectors, location):
  ''' 
  Calculate the matrix R, where the i-th 3x3 block of R gives
  (R_i x) = r_i cross x.
  R will be 3N by 3 (36 x 3). The r vectors point from the center
  of the icosohedron to the other vertices.
  '''
  rot_matrix = None
  for k in range(len(r_vectors)):
    # Here the cross is relative to the center.
    adjusted_r_vector = r_vectors[k] - location
    block = np.array(
        [[0.0, adjusted_r_vector[2], -1.*adjusted_r_vector[1]],
        [-1.*adjusted_r_vector[2], 0.0, adjusted_r_vector[0]],
        [adjusted_r_vector[1], -1.*adjusted_r_vector[0], 0.0]])
    if rot_matrix is None:
      rot_matrix = block
    else:
      rot_matrix = np.concatenate([rot_matrix, block], axis=0)
  return rot_matrix


def icosohedron_force_calculator(location, orientation):
  ''' Force on the Icosohedron center. 
  args: 
  location:   list of length 1, only entry is a list of
              length 3 with coordinates of tetrahedon "top" vertex.
  orientation: list of length 1, only entry is a quaternion with the 
               tetrahedron orientation
  '''
  gravity = [0., 0., -1.*sum(M)]
  h = location[0][2]
  repulsion = np.array([0., 0., 
                        (REPULSION_STRENGTH*((h - A)/DEBYE_LENGTH + 1)*
                         np.exp(-1.*(h - A)/DEBYE_LENGTH)/
                         ((h - A)**2))])
  return repulsion + gravity


def icosohedron_torque_calculator(location, orientation):
  ''' For now, approximate torque as zero, which is true for the sphere.'''
  return [0., 0., 0.]


def icosohedron_check_function(location, orientation):
  ''' Check that the Icosohedron is not overlapping the wall. '''
  if location[0][2] < A + VERTEX_A:
    return False
  else:
    return True
