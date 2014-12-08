'''

A free tetrahedron is allowed to diffuse in a domain with a single
wall (below the tetrahedron) in the presence of gravity and a quadratic potential
repelling from the wall.
'''
import sys
import numpy as np
import tetrahedron as tdn
from quaternion import Quaternion
from quaternion_integrator import QuaternionIntegrator
import math
import cPickle

PROFILE = False  # Do we profile this run?

ETA = 1.0   # Fluid viscosity.
A = 0.5     # Particle Radius.
H = 3.0     # Distance to wall.

# Masses of particles.
M1 = 0.2
M2 = 0.4
M3 = 0.6


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
  rotation_matrix = calc_free_rot_matrix(r_vectors, location)
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


def free_gravity_torque_calculator(location, orientation):
  ''' 
  Calculate torque based on location, given as a length 1 list of 
  a 3-vector, and orientation, given as a length
  1 list of quaternions (1 quaternion).  This assumes the masses
  of particles 1, 2, and 3 are M1, M2, and M3 respectively.
  '''
  r_vectors = get_free_r_vectors(location[0], orientation[0])
  R = calc_free_rot_matrix(r_vectors, location[0])
  # Gravity
  g = np.array([0., 0., -1.*M1, 0., 0., -1.*M2, 0., 0., -1.*M3])
  return np.dot(R.T, g)


def free_gravity_force_calculator(location, orientation):
  '''
  Calculate force on tetrahedron given it's location and
  orientation.  
  args: 
  location:   list of length 1, only entry is a list of
              length 3 with coordinates of tetrahedon "top" vertex.
  orientation: list of length 1, only entry is a quaternion with the 
               tetrahedron orientation
  '''
  # TODO: Tune repulsion from the wall to keep tetrahedron away.
  # TODO: add a mass at the top vertex, make all vertices repel
  potential_force = np.array([0., 0., (8./(location[0][2]**2))])
  gravity_force = np.array([0., 0., -1.*(M1 + M2 + M3)])
  return potential_force + gravity_force


def bin_free_particle_heights(location, orientation, bin_width, 
                              height_histogram):
  '''Bin heights of the free particle based on a location and an orientaiton.'''
  r_vectors = get_free_r_vectors(location, orientation)
  for k in range(3):
    # Bin each particle height.
    idx = (int(math.floor((r_vectors[k][2])/bin_width)))
    if idx < len(height_histogram[k]):
      height_histogram[k][idx] += 1
    else:
      print 'index is: ', idx
      print 'r_vectors are: ', r_vectors
      print 'Index exceeds histogram length.'
  
  
def generate_free_equilibrum_sample():
  '''
  Generate an equilibrium sample of location and orientation, according
  to the distribution exp(-\beta U(heights)).
  Do this by generating a uniform quaternion and location, 
  then accept/rejecting with probability
  exp(-U(heights))
  '''
  max_gibbs_term = 0.
  while True:
    # First generate a uniform quaternion on the 4-sphere.
    theta = np.random.normal(0., 1., 4)
    theta = Quaternion(theta/np.linalg.norm(theta))
    # For location, set x and y to 0, since these are not affected
    # by the potential.
    location = [0., 0., np.random.uniform(2.5, 20.0)]
    r_vectors = get_free_r_vectors(location, theta)
    #TODO: add potential from wall to this.
    U = (M1*r_vectors[0][2] + M2*r_vectors[1][2] + M3*r_vectors[2][2] + 
         8./location[2])
    # Roughly the smallest height.
    smallest_height = 2.0
    normalization_constant = np.exp(-1.*smallest_height*(M1 + M2 + M3) - 
                                    8./(3*smallest_height))
    # For now, we set the normalization to 1e-2 for masses:
    #       M1 = 0.1, M2 = 0.2, M3 = 0.3
    gibbs_term = np.exp(-1.*U)
    if gibbs_term > max_gibbs_term:
      max_gibbs_term = gibbs_term
    accept_prob = np.exp(-1.*(U))/normalization_constant
    if accept_prob > 1:
      print "Warning: acceptance probability > 1."
      print "accept_prob = ", accept_prob
    if np.random.uniform() < accept_prob:
      return [location, theta]


if __name__ == '__main__':
  if PROFILE:
    pr = cProfile.Profile()
    pr.enable()
    
  # Script to run the various integrators on the quaternion.
  initial_location = [[0., 0., H]]
  initial_orientation = [Quaternion([1., 0., 0., 0.])]
  fixman_integrator = QuaternionIntegrator(free_tetrahedron_mobility,
                                           initial_orientation, 
                                           free_gravity_torque_calculator, 
                                           has_location = True,
                                           initial_location = initial_location,
                                           force_calculator = free_gravity_force_calculator)
  

  # Get command line parameters
  dt = float(sys.argv[1])
  n_steps = int(sys.argv[2])
  print_increment = max(int(n_steps/10.), 1)

  # For now hard code bin width.  Number of bins is equal to
  # 4 over bin_width, since the particle can be in a -2, +2 range around
  # the fixed vertex.
  bin_width = 1./2.
  fixman_heights = np.array([np.zeros(int(25./bin_width)) for _ in range(3)])
  equilibrium_heights = np.array([np.zeros(int(25./bin_width)) for _ in range(3)])

  for k in range(n_steps):
    # Fixman step and bin result.
    fixman_integrator.fixman_time_step(dt)
    bin_free_particle_heights(fixman_integrator.location[0],
                              fixman_integrator.orientation[0], 
                              bin_width, 
                              fixman_heights)
    # Bin equilibrium sample.
    sample = generate_free_equilibrum_sample()
    bin_free_particle_heights(sample[0], 
                              sample[1],
                              bin_width, 
                              equilibrium_heights)

    if k % print_increment == 0:
      print "At step:", k

  heights = [fixman_heights, equilibrium_heights]

    # Optional name for data provided
  if len(sys.argv) > 3:
    data_name = './data/free-tetrahedron-dt-%g-N-%d-%s.pkl' % (dt, n_steps, sys.argv[3])
  else:
    data_name = './data/free-tetrahedron-dt-%g-N-%d.pkl' % (dt, n_steps)

  with open(data_name, 'wb') as f:
    cPickle.dump(heights, f)
  
  if PROFILE:
    pr.disable()
    s = StringIO.StringIO()
    sortby = 'cumulative'
    ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
    ps.print_stats()
    print s.getvalue()

