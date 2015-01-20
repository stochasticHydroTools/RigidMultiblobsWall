'''
Script to test a tetrahedron near a wall.  The wall is at z = 0, and
the tetrahedron's "top" vertex is fixed at (0, 0, H).

This file has the mobility and torque calculator used for any tetrahedron
test.  Running this script will run a trajectory and bin the heights of each
of the three non-fixed vertices for Fixman, RFD, and EM timestepping, as well as
for the equilibrium distribution.  

Before running this script, you must compile mobility_ext.cc in 
/constrained_diffusion/fluids.  Just run make in the fluids folder.
'''

import sys
sys.path.append('..')
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot
import argparse
import cPickle
import cProfile, pstats, StringIO
import math
import time
import logging

from quaternion_integrator.quaternion import Quaternion
from quaternion_integrator.quaternion_integrator import QuaternionIntegrator
from fluids import mobility as mb
import uniform_analyzer as ua

ETA = 1.0   # Fluid viscosity.
A = 0.5     # Particle Radius.
H = 2.5     # Distance to wall.

# Masses of particles.
M1 = 0.1
M2 = 0.2
M3 = 0.3

# Fake log-like class to redirect stdout to log file.
class StreamToLogger(object):
   """
   Fake file-like stream object that redirects writes to a logger instance.
   """
   def __init__(self, logger, log_level=logging.INFO):
      self.logger = logger
      self.log_level = log_level
      self.linebuf = ''
 
   def write(self, buf):
      for line in buf.rstrip().splitlines():
         self.logger.log(self.log_level, line.rstrip())


def identity_mobility(orientation):
  ''' Simple identity mobility for testing. '''
  return np.identity(3)


def test_mobility(orientation):
  ''' Simple mobility that's not divergence free. '''
  r_vectors = get_r_vectors(orientation[0])
  total_mobility = np.array([np.zeros(3) for _ in range(3)])
  for k in range(3):
    total_mobility[k, k] = r_vectors[k][2]**2 + 1.
  return total_mobility


def tetrahedron_mobility(orientation):
  ''' 
  Wrapper for torque mobility that takes a quaternion for
  use with quaternion_integrator. 
  '''
  r_vectors = get_r_vectors(orientation[0])
  return torque_mobility(r_vectors)

def torque_oseen_mobility(r_vectors):
  '''
  Calculate the mobility, torque -> angular velocity, at orientation 
  In this case, orientation is length 1, as there is just 1 quaternion.
  The mobility is equal to R M^-1 R^t where R is 3N x 3 (9 x 3)
  Rx = r cross x
  r is the distance from the fixed vertex of the tetrahedron to
  each other vertex (a length 3N vector).
  M (3N x 3N) is the singular image stokeslet for a point force near a wall, but
  we've replaced the diagonal piece by 1/(6 pi eta a).
  '''  
  mobility = mb.image_singular_stokeslet(r_vectors)
  rotation_matrix = calculate_rot_matrix(r_vectors)
  total_mobility = np.linalg.inv(np.dot(rotation_matrix.T,
                                        np.dot(np.linalg.inv(mobility),
                                               rotation_matrix)))
  return total_mobility


def torque_mobility(r_vectors):
  '''
  Calculate the mobility, torque -> angular velocity, at orientation 
  In this case, orientation is length 1, as there is just 1 quaternion.
  The mobility is equal to R M^-1 R^t where R is 3N x 3 (9 x 3)
  Rx = r cross x
  r is the distance from the fixed vertex of the tetrahedron to
  each other vertex (a length 3N vector).
  M (3N x 3N) is the finite size single wall mobility taken from the
  Swan and Brady paper:
   "Simulation of hydrodynamically interacting particles near a no-slip
    boundary."
  '''  
  mobility = mb.boosted_single_wall_fluid_mobility(r_vectors, ETA, A)
  rotation_matrix = calculate_rot_matrix(r_vectors)
  total_mobility = np.linalg.inv(np.dot(rotation_matrix.T,
                                        np.dot(np.linalg.inv(mobility),
                                               rotation_matrix)))
  return total_mobility


def rpy_torque_mobility(r_vectors):
  '''
  Calculate the mobility, torque -> angular velocity, at orientation 
  In this case, orientation is length 1, as there is just 1 quaternion.
  The mobility is equal to R M^-1 R^t where R is 3N x 3 (9 x 3)
  Rx = r cross x
  r is the distance from the fixed vertex of the tetrahedron to
  each other vertex (a length 3N vector).
  M (3N x 3N) is the RPY tensor.
  '''  
  mobility = mb.rotne_prager_tensor(r_vectors, ETA, A)
  rotation_matrix = calculate_rot_matrix_cm(r_vectors)
  total_mobility = np.linalg.inv(np.dot(rotation_matrix.T,
                                        np.dot(np.linalg.inv(mobility),
                                               rotation_matrix)))
  return total_mobility


def calculate_rot_matrix(r_vectors):
  ''' Calculate R, 3N by 3 matrix of cross products for r_i. '''
  
  # Create the 3N x 3 matrix.  Each 3x3 block is the matrix for a cross
  # product with one of the r_vectors.  Cross is relative to (0, 0, H) the location
  # of the fixed vertex.
  
  # Adjust so we take the cross relative to (0, 0, H)
  rot_matrix = None
  for k in range(len(r_vectors)):
    r_vectors[k] = r_vectors[k] - np.array([0., 0., H])

    # Current r cross x matrix block.
    block = np.array(
        [[0.0, r_vectors[k][2], -1.*r_vectors[k][1]],
        [-1.*r_vectors[k][2], 0.0, r_vectors[k][0]],
        [r_vectors[k][1], -1.*r_vectors[k][0], 0.0]])

    if rot_matrix is None:
      rot_matrix = block
    else:
      rot_matrix = np.concatenate([rot_matrix, block], axis=0)

  return rot_matrix


def calculate_rot_matrix_cm(r_vectors):
  ''' Calculate R, 3N by 3 matrix of cross products for r_i relative to 0, 0, 0 '''
  # Create the 3N x 3 matrix.  Each 3x3 block is the matrix for a cross
  # product with one of the r_vectors.  Cross is relative to (0, 0, 0)   
  rot_matrix = None
  for k in range(len(r_vectors)):
    # Current r cross x matrix block.
    block = np.array(
        [[0.0, r_vectors[k][2], -1.*r_vectors[k][1]],
        [-1.*r_vectors[k][2], 0.0, r_vectors[k][0]],
        [r_vectors[k][1], -1.*r_vectors[k][0], 0.0]])

    if rot_matrix is None:
      rot_matrix = block
    else:
      rot_matrix = np.concatenate([rot_matrix, block], axis=0)

  return rot_matrix


def get_r_vectors(quaternion):
  ''' Calculate r_i from a given quaternion. 
  The initial configuration is hard coded here but can be changed by
  considering an initial quaternion not equal to the identity rotation.
  initial configuration (top down view, the top vertex is fixed at the origin):

                         O r_1 = (0, 2/sqrt(3), -(2 sqrt(2))/3)
                        / \
                       /   \
                      /     \
                     /   O(0, 0, H)
                    /          \
                   /            \
               -> O--------------O  r_3 = (1, -1/sqrt(3),-(2 sqrt(2))/3)
             /
           r_2 = (-1, -1/sqrt(3),-(2 sqrt(2))/3)

  Each side of the tetrahedron has length 2.
  '''
  initial_r1 = np.array([0., 2./np.sqrt(3.), -2.*np.sqrt(2.)/np.sqrt(3.)])
  initial_r2 = np.array([-1., -1./np.sqrt(3.), -2.*np.sqrt(2.)/np.sqrt(3.)])
  initial_r3 = np.array([1., -1./np.sqrt(3.), -2.*np.sqrt(2.)/np.sqrt(3.)])
  
  rotation_matrix = quaternion.rotation_matrix()

  r1 = np.dot(rotation_matrix, initial_r1) + np.array([0., 0., H])
  r2 = np.dot(rotation_matrix, initial_r2) + np.array([0., 0., H])
  r3 = np.dot(rotation_matrix, initial_r3) + np.array([0., 0., H])
  
  return [r1, r2, r3]

def gravity_torque_calculator(orientation):
  ''' 
  Calculate torque based on orientation, given as a length
  1 list of quaternions (1 quaternion).  This assumes the masses
  of particles 1, 2, and 3 are M1, M2, and M3 respectively.
  '''
  r_vectors = get_r_vectors(orientation[0])
  R = calculate_rot_matrix(r_vectors)
  # Gravity
  g = np.array([0., 0., -1.*M1, 0., 0., -1.*M2, 0., 0., -1.*M3])
  return np.dot(R.T, g)


def zero_torque_calculator(orientation):
  ''' Return 0 torque. '''
  # Gravity
  return np.array([0., 0., 0.])


def generate_equilibrium_sample():
  ''' 
  Generate a sample according to the equilibrium distribution, exp(-\beta U(heights)).
  Do this by generating a uniform quaternion, then accept/rejecting with probability
  exp(-U(heights))'''
  max_gibbs_term = 0.
  while True:
    # First generate a uniform quaternion on the 4-sphere.
    theta = np.random.normal(0., 1., 4)
    theta = Quaternion(theta/np.linalg.norm(theta))
  
    r_vectors = get_r_vectors(theta)
    U = M1*r_vectors[0][2] + M2*r_vectors[1][2] + M3*r_vectors[2][2]
    # Roughly the smallest height.
    smallest_height = H - 1.8
    normalization_constant = np.exp(-1.*smallest_height*(M1 + M2 + M3))
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
      return theta

def bin_particle_heights(orientation, bin_width, height_histogram):
  ''' 
  Given a quaternion orientation, bin the heights of the three particles.
  '''
  r_vectors = get_r_vectors(orientation)
  for k in range(3):
    # Bin each particle height.
    idx = (int(math.floor((r_vectors[k][2] - H)/bin_width)) + 
           len(height_histogram[k])/2)
    height_histogram[k][idx] += 1


if __name__ == "__main__":
  # Get command line arguments.
  parser = argparse.ArgumentParser(description='Run Simulation of fixed '
                                   'tetrahedron with Fixman, EM, and RFD '
                                   'schemes, and bin the resulting '
                                   'height distribution.')
  parser.add_argument('-dt', dest='dt', type=float,
                      help='Timestep to use for runs.')
  parser.add_argument('-N', dest='n_steps', type=int,
                      help='Number of steps to take for runs.')
  parser.add_argument('--data-name', dest='data_name', type=str,
                      default='',
                      help='Optional name added to the end of the '
                      'data file.  Useful for multiple runs '
                      '(--data_name=run-1).')
  parser.add_argument('--profile', dest='profile', type=bool, default=False,
                      help='True or False: Do we profile this run or not.')
  
  args = parser.parse_args()
  if args.profile:
    pr = cProfile.Profile()
    pr.enable()

  # Script to run the various integrators on the quaternion.
  initial_orientation = [Quaternion([1., 0., 0., 0.])]
  fixman_integrator = QuaternionIntegrator(tetrahedron_mobility,
                                           initial_orientation, 
                                           gravity_torque_calculator)

  rfd_integrator = QuaternionIntegrator(tetrahedron_mobility, 
                                        initial_orientation, 
                                        gravity_torque_calculator)

  em_integrator = QuaternionIntegrator(tetrahedron_mobility, 
                                       initial_orientation, 
                                       gravity_torque_calculator)
  # Get command line parameters
  dt = args.dt #float(sys.argv[1])
  n_steps = args.n_steps #int(sys.argv[2])
  print_increment = max(int(n_steps/20.), 1)

  # Set up logging.
  # Make directory for logs if it doesn't exist.
  if not os.path.isdir(os.path.join(os.getcwd(), 'logs')):
    os.mkdir(os.path.join(os.getcwd(), 'logs'))

  log_filename = './logs/tetrahedron-dt-%d-N-%d-%s.log' % (
    dt, n_steps, args.data_name)
  progress_logger = logging.getLogger('progress_logger')
  progress_logger.setLevel(logging.INFO)
  # Add the log message handler to the logger
  logging.basicConfig(filename=log_filename,
                      level=logging.INFO,
                      filemode='w')
  sl = StreamToLogger(progress_logger, logging.INFO)
  sys.stdout = sl
  sl = StreamToLogger(progress_logger, logging.ERROR)
  sys.stderr = sl
  
  # For now hard code bin width.  Number of bins is equal to
  # 4 over bin_width, since the particle can be in a -2, +2 range around
  # the fixed vertex.
  bin_width = 1./5.
  fixman_heights = np.array([np.zeros(int(4./bin_width)) for _ in range(3)])
  rfd_heights = np.array([np.zeros(int(4./bin_width)) for _ in range(3)])
  em_heights = np.array([np.zeros(int(4./bin_width)) for _ in range(3)])
  equilibrium_heights = np.array([np.zeros(int(4./bin_width)) for _ in range(3)])

  start_time = time.time()
  for k in range(n_steps):
    # Fixman step and bin result.
    fixman_integrator.fixman_time_step(dt)
    bin_particle_heights(fixman_integrator.orientation[0], 
                         bin_width, 
                         fixman_heights)
    # RFD step and bin result.
    rfd_integrator.rfd_time_step(dt)
    bin_particle_heights(rfd_integrator.orientation[0],
                         bin_width, 
                         rfd_heights)    
    # EM step and bin result.
    em_integrator.additive_em_time_step(dt)
    bin_particle_heights(em_integrator.orientation[0],
                         bin_width, 
                         em_heights)
    # Bin equilibrium sample.
    bin_particle_heights(generate_equilibrium_sample(), 
                         bin_width, 
                         equilibrium_heights)
    
    if k % print_increment == 0:
      elapsed_time = time.time() - start_time
      if elapsed_time < 60.:
        progress_logger.info('At step: %d. Time Taken: %.2f Seconds' % 
                             (k, float(elapsed_time)))
        if k > 0:
          progress_logger.info('Estimated Total time required: %.2f Seconds.' %
                               (elapsed_time*float(n_steps)/float(k)))
      else:
        progress_logger.info('At step: %d. Time Taken: %.2f Minutes.' %
                             (k, (float(elapsed_time)/60.)))
        if k > 0:
          progress_logger.info('Estimated Total time required: %.2f Minutes.' %
                               (elapsed_time*float(n_steps)/float(k)/60.))

  elapsed_time = time.time() - start_time
  if elapsed_time > 60:
    progress_logger.info('Finished timestepping. Total Time: %.2f minutes.' % 
                         (float(elapsed_time)/60.))
  else:
    progress_logger.info('Finished timestepping. Total Time: %.2f seconds.' % 
                         float(elapsed_time))

  heights = [fixman_heights/(n_steps*bin_width),
             rfd_heights/(n_steps*bin_width),
             em_heights/(n_steps*bin_width),
             equilibrium_heights/(n_steps*bin_width)]

  # Make directory for data if it doesn't exist.
  if not os.path.isdir(os.path.join(os.getcwd(), 'data')):
    os.mkdir(os.path.join(os.getcwd(), 'data'))

  # Optional name for data provided    
  if len(args.data_name) > 0:
    data_name = './data/tetrahedron-dt-%g-N-%d-%s.pkl' % (dt, n_steps, args.data_name)
  else:
    data_name = './data/tetrahedron-dt-%g-N-%d.pkl' % (dt, n_steps)

  height_data = dict()
  height_data['params'] = {'A': A, 'ETA': ETA, 'H': H, 'M1': M1, 'M2': M2, 
                           'M3': M3}
  height_data['heights'] = heights
  height_data['names'] = ['Fixman', 'RFD', 'EM', 'Gibbs-Boltzmann']
  height_data['buckets'] = H + np.linspace(-2., 2., len(heights[0][0]))

  with open(data_name, 'wb') as f:
    cPickle.dump(height_data, f)
  
  if args.profile:
    pr.disable()
    s = StringIO.StringIO()
    sortby = 'cumulative'
    ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
    ps.print_stats()
    print s.getvalue()


