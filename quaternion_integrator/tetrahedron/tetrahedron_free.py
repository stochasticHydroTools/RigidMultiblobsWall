'''

A free tetrahedron is allowed to diffuse in a domain with a single
wall (below the tetrahedron) in the presence of gravity and a quadratic potential
repelling from the wall.
'''

import numpy as np
import tetrahedron as tdn

def free_tetrahedron_mobility(location, orientation):
  ''' 
  Wrapper for torque mobility that takes a quaternion and location for
  use with quaternion_integrator. 
  '''
  r_vectors = get_free_r_vectors(location[0], orientation[0])
  return force_and_torque_mobility(r_vectors, location[0])


def force_and_torque_mobility(r_vectors, location):
  '''
  Calculate the mobility: (torque, force) -> (angular velocity, velocity) at position 
  In this case, position has orientation and location data, each of length 1.
  The mobility is equal to the inverse of: 
    [ J^T M^-1 J,   J^T M^-1 R ]
    [ R^T M^-1 J,   R^T M^-1 R ]
  where R is 3N x 3 (9 x 3) Rx = r cross x and J is a 3N x 3 matrix with 
  each 3x3 block being the identity.
  r is the distance from the fixed vertex of the tetrahedron to
  each other vertex (a length 3N vector).
  M (3N x 3N) is the finite size single wall mobility taken from the
  Swan and Brady paper:
   "Simulation of hydrodynamically interacting particles near a no-slip
    boundary."
  Here location is the dereferenced list with 3 entries.
  '''  
  mobility = tdn.single_wall_fluid_mobility(r_vectors, ETA, A)
  rotation_matrix = calculate_free_rot_matrix(r_vectors, location)
  J = np.concatenate([np.identity(3), np.identity(3), np.identity(3)])
  J_rot_combined = np.concatenate([J, rotation_matrix], axis=1)
  total_mobility = np.linalg.inv(np.dot(J_rot_combined.T,
                                        np.dot(np.linalg.inv(mobility),
                                               J_rot_combined)))
  return total_mobility


def get_free_r_vectors(location, quaternion):
  ''' Calculate r_i from a given quaternion. 
  The initial configuration is hard coded here but can be changed by
  considering an initial quaternion not equal to the identity rotation.
  initial configuration (top down view, the top vertex is fixed at the origin):

                         O r_1 = (0, 2/sqrt(3), -(2 sqrt(2))/3)
                        / \
                       /   \
                      /     \
                     /   O(location, everything else relative to this location)
                    /          \
                   /            \
               -> O--------------O  r_3 = (1, -1/sqrt(3),-(2 sqrt(2))/3)
             /
           r_2 = (-1, -1/sqrt(3),-(2 sqrt(2))/3)

  Each side of the tetrahedron has length 2.
  location is a 3-dimensional list giving the location of the "top" vertex.
  quaternion is a quaternion representing the tetrahedron orientation.
  '''
  initial_r1 = np.array([0., 2./np.sqrt(3.), -2.*np.sqrt(2.)/np.sqrt(3.)])
  initial_r2 = np.array([-1., -1./np.sqrt(3.), -2.*np.sqrt(2.)/np.sqrt(3.)])
  initial_r3 = np.array([1., -1./np.sqrt(3.), -2.*np.sqrt(2.)/np.sqrt(3.)])
  
  rotation_matrix = quaternion.rotation_matrix()

  r1 = np.dot(rotation_matrix, initial_r1) + np.array(location)
  r2 = np.dot(rotation_matrix, initial_r2) + np.array(location)
  r3 = np.dot(rotation_matrix, initial_r3) + np.array(location)
  
  return [r1, r2, r3]


def calc_free_rot_matrix(r_vectors, location):
  ''' 
  Calculate rotation matrix (r cross) based on the free tetrahedron.
  In this case, the R vectors point from the "top" vertex to the others.
  '''
  rot_matrix = None
  for k in range(len(r_vectors)):
    # Current r cross x matrix block.
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
