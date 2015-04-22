''' Functions used for the sphere discretized with 42 blobs near a wall. '''

import argparse
import cPickle
import logging
import math
import numpy as np
import os
import sys
sys.path.append('..')
import time

from fluids import mobility as mb
from quaternion_integrator.quaternion import Quaternion
from quaternion_integrator.quaternion_integrator import QuaternionIntegrator
from utils import StreamToLogger
from utils import static_var
from utils import log_time_progress

# Parameters
ETA = 1.0                               # Viscosity.
VERTEX_A = 0.2732666211340000206320155  # radius of individual vertices
A = 1                                   # 'Radius' of entire sphere_42_blobs.
M = [1/42. for _ in range(42)]          # Masses of particles
KT = 1                                  # Temperature

# Repulsion potential paramters. Using Yukawa potential.
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

def sphere_42_blobs_mobility(location, orientation):
  ''' 
  Mobility for the rigid sphere_42_blobs, return a 6x6 matrix
  that takes Force + Torque and returns velocity and angular velocity.
  '''
  r_vectors = get_sphere_42_blobs_r_vectors(location[0], orientation[0])
  return force_and_torque_sphere_42_blobs_mobility(r_vectors, location[0])

def sphere_42_blobs_center_mobility(location, orientation):
  ''' 
  Mobility for the rigid sphere_42_blobs, return a 6x6 matrix
  that takes Force + Torque and returns velocity and angular velocity.
  '''
  r_vectors = get_sphere_42_blobs_center_r_vectors(location[0], orientation[0])
  return force_and_torque_sphere_42_blobs_mobility(r_vectors, location[0])


def force_and_torque_sphere_42_blobs_mobility(r_vectors, location):
  '''
  Calculate the mobility: (torque, force) -> (angular velocity, velocity) at position 
  The mobility is equal to the inverse of: 
    [ J^T M^-1 J,   J^T M^-1 R ]
    [ R^T M^-1 J,   R^T M^-1 R ]
  where R is 3N x 3 (126 x 3) Rx = r cross x and J is a 3N x 3 matrix with 
  each 3x3 block being the identity.
  r is the distance from the center vertex of the sphere_42_blobs to
  each other vertex (a length 3N vector).
  M (3N x 3N) is the finite size single wall mobility taken from the
  Swan and Brady paper:
   "Simulation of hydrodynamically interacting particles near a no-slip
    boundary."
  Here location is the dereferenced list with 3 entries.
  '''  
  mobility = mb.boosted_single_wall_fluid_mobility(r_vectors, ETA, VERTEX_A)
  rotation_matrix = calc_sphere_42_blobs_rot_matrix(r_vectors, location)
  J = np.concatenate([np.identity(3) for _ in range(len(r_vectors))])
  J_rot_combined = np.concatenate([J, rotation_matrix], axis=1)
  total_mobility = np.linalg.inv(np.dot(J_rot_combined.T,
                                        np.dot(np.linalg.inv(mobility),
                                               J_rot_combined)))
  return total_mobility


def get_sphere_42_blobs_r_vectors(location, orientation):
  ''' Get the locations of each individual vertex of the 42-blobs-sphere. '''
  # These values taken from an IBAMR vertex file. 'Radius' of 
  # Entire structure is ~1.
  initial_setup = [np.array([0.000000000000000e+00, 0.000000000000000e+00, 9.999999999993719e-01]),
                   np.array([2.763932022498475e-01, 8.506508083515056e-01, 4.472135954996771e-01]),
                   np.array([-7.236067977495244e-01, 5.257311121188035e-01, 4.472135954996771e-01]),
                   np.array([-7.236067977495246e-01, -5.257311121188033e-01, 4.472135954996771e-01]),	
                   np.array([2.763932022498473e-01, -8.506508083515057e-01, 4.472135954996771e-01]),
                   np.array([8.944271909993541e-01, -2.190714793055435e-16, 4.472135954996771e-01]),
                   np.array([7.236067977495245e-01, 5.257311121188034e-01, -4.472135954996771e-01]),
                   np.array([-2.763932022498474e-01, 8.506508083515057e-01, -4.472135954996771e-01]),
                   np.array([-8.944271909993541e-01, 1.095357396527717e-16, -4.472135954996771e-01]),
                   np.array([-2.763932022498475e-01, -8.506508083515056e-01, -4.472135954996771e-01]),
                   np.array([7.236067977495244e-01, -5.257311121188036e-01, -4.472135954996771e-01]),
                   np.array([0.000000000000000e+00, 0.000000000000000e+00, -9.999999999993719e-01]),
                   np.array([-2.628655560594016e-01, 8.090169943744392e-01, 5.257311121188034e-01]),
                   np.array([-4.253254041757528e-01, 3.090169943747534e-01, 8.506508083515057e-01]),
                   np.array([1.624598481163512e-01, 4.999999999996860e-01, 8.506508083515057e-01]),
                   np.array([-8.506508083515057e-01, 1.305145441259601e-16, 5.257311121188035e-01]),
                   np.array([-4.253254041757529e-01, -3.090169943747533e-01, 8.506508083515057e-01]),
                   np.array([-2.628655560594019e-01,-8.090169943744392e-01, 5.257311121188035e-01]),
                   np.array([1.624598481163510e-01, -4.999999999996861e-01, 8.506508083515057e-01]),
                   np.array([6.881909602351544e-01, -4.999999999996862e-01, 5.257311121188035e-01]),
                   np.array([5.257311121188034e-01, -1.287669847336942e-16, 8.506508083515057e-01]),
                   np.array([6.881909602351546e-01, 4.999999999996858e-01, 5.257311121188034e-01]),
                   np.array([-5.877852522921038e-01, 8.090169943744394e-01, 0.000000000000000e+00]),
                   np.array([6.525727206298004e-17, 9.999999999993719e-01, 0.000000000000000e+00]),
                   np.array([-9.510565162945563e-01, -3.090169943747532e-01, 0.000000000000000e+00]),
                   np.array([-9.510565162945562e-01, 3.090169943747535e-01, 0.000000000000000e+00]),
                   np.array([-1.631431801574501e-16, -9.999999999993719e-01, 0.000000000000000e+00]),
                   np.array([-5.877852522921041e-01, -8.090169943744392e-01, 0.000000000000000e+00]),
                   np.array([9.510565162945562e-01, -3.090169943747536e-01, 0.000000000000000e+00]),
                   np.array([5.877852522921039e-01, -8.090169943744395e-01, 0.000000000000000e+00]),
                   np.array([5.877852522921040e-01, 8.090169943744392e-01, 0.000000000000000e+00]),
                   np.array([9.510565162945563e-01, 3.090169943747532e-01, 0.000000000000000e+00]),
                   np.array([2.628655560594017e-01, 8.090169943744392e-01, -5.257311121188034e-01]),
                   np.array([-6.881909602351545e-01, 4.999999999996861e-01, -5.257311121188034e-01]),
                   np.array([-6.881909602351546e-01, -4.999999999996859e-01, -5.257311121188034e-01]),
                   np.array([2.628655560594015e-01, -8.090169943744394e-01, -5.257311121188034e-01]),
                   np.array([8.506508083515057e-01, -1.305145441259601e-16, -5.257311121188035e-01]),
                   np.array([-1.624598481163511e-01, 4.999999999996861e-01, -8.506508083515057e-01]),
                   np.array([4.253254041757529e-01, 3.090169943747534e-01, -8.506508083515057e-01]),
                   np.array([-5.257311121188034e-01, 6.438349236684708e-17, -8.506508083515057e-01]),
                   np.array([-1.624598481163512e-01, -4.999999999996860e-01, -8.506508083515057e-01]),
                   np.array([4.253254041757528e-01, -3.090169943747535e-01, -8.506508083515057e-01])]

  rotation_matrix = orientation.rotation_matrix()

  # TODO: Maybe don't do this on the fly every single time.
  for k in range(len(initial_setup)):
    initial_setup[k] = A*(initial_setup[k])

  rotated_setup = []
  for r in initial_setup:
    rotated_setup.append(np.dot(rotation_matrix, r) + np.array(location))
    
  return rotated_setup


def get_sphere_42_blobs_center_r_vectors(location, orientation):
  ''' Get the locations of each individual vertex of the sphere_42_blobs. 
  Same as above, but now we have a blob in the center '''
  # These values
  # taken from an IBAMR vertex file. 'Radius' of 
  # Entire structure is ~1.
  initial_setup = [np.array([0.0, 0.0, 0.0]),
                   np.array([0.000000000000000e+00, 0.000000000000000e+00, 9.999999999993719e-01]),
                   np.array([2.763932022498475e-01, 8.506508083515056e-01, 4.472135954996771e-01]),
                   np.array([-7.236067977495244e-01, 5.257311121188035e-01, 4.472135954996771e-01]),
                   np.array([-7.236067977495246e-01, -5.257311121188033e-01, 4.472135954996771e-01]),	
                   np.array([2.763932022498473e-01, -8.506508083515057e-01, 4.472135954996771e-01]),
                   np.array([8.944271909993541e-01, -2.190714793055435e-16, 4.472135954996771e-01]),
                   np.array([7.236067977495245e-01, 5.257311121188034e-01, -4.472135954996771e-01]),
                   np.array([-2.763932022498474e-01, 8.506508083515057e-01, -4.472135954996771e-01]),
                   np.array([-8.944271909993541e-01, 1.095357396527717e-16, -4.472135954996771e-01]),
                   np.array([-2.763932022498475e-01, -8.506508083515056e-01, -4.472135954996771e-01]),
                   np.array([7.236067977495244e-01, -5.257311121188036e-01, -4.472135954996771e-01]),
                   np.array([0.000000000000000e+00, 0.000000000000000e+00, -9.999999999993719e-01]),
                   np.array([-2.628655560594016e-01, 8.090169943744392e-01, 5.257311121188034e-01]),
                   np.array([-4.253254041757528e-01, 3.090169943747534e-01, 8.506508083515057e-01]),
                   np.array([1.624598481163512e-01, 4.999999999996860e-01, 8.506508083515057e-01]),
                   np.array([-8.506508083515057e-01, 1.305145441259601e-16, 5.257311121188035e-01]),
                   np.array([-4.253254041757529e-01, -3.090169943747533e-01, 8.506508083515057e-01]),
                   np.array([-2.628655560594019e-01,-8.090169943744392e-01, 5.257311121188035e-01]),
                   np.array([1.624598481163510e-01, -4.999999999996861e-01, 8.506508083515057e-01]),
                   np.array([6.881909602351544e-01, -4.999999999996862e-01, 5.257311121188035e-01]),
                   np.array([5.257311121188034e-01, -1.287669847336942e-16, 8.506508083515057e-01]),
                   np.array([6.881909602351546e-01, 4.999999999996858e-01, 5.257311121188034e-01]),
                   np.array([-5.877852522921038e-01, 8.090169943744394e-01, 0.000000000000000e+00]),
                   np.array([6.525727206298004e-17, 9.999999999993719e-01, 0.000000000000000e+00]),
                   np.array([-9.510565162945563e-01, -3.090169943747532e-01, 0.000000000000000e+00]),
                   np.array([-9.510565162945562e-01, 3.090169943747535e-01, 0.000000000000000e+00]),
                   np.array([-1.631431801574501e-16, -9.999999999993719e-01, 0.000000000000000e+00]),
                   np.array([-5.877852522921041e-01, -8.090169943744392e-01, 0.000000000000000e+00]),
                   np.array([9.510565162945562e-01, -3.090169943747536e-01, 0.000000000000000e+00]),
                   np.array([5.877852522921039e-01, -8.090169943744395e-01, 0.000000000000000e+00]),
                   np.array([5.877852522921040e-01, 8.090169943744392e-01, 0.000000000000000e+00]),
                   np.array([9.510565162945563e-01, 3.090169943747532e-01, 0.000000000000000e+00]),
                   np.array([2.628655560594017e-01, 8.090169943744392e-01, -5.257311121188034e-01]),
                   np.array([-6.881909602351545e-01, 4.999999999996861e-01, -5.257311121188034e-01]),
                   np.array([-6.881909602351546e-01, -4.999999999996859e-01, -5.257311121188034e-01]),
                   np.array([2.628655560594015e-01, -8.090169943744394e-01, -5.257311121188034e-01]),
                   np.array([8.506508083515057e-01, -1.305145441259601e-16, -5.257311121188035e-01]),
                   np.array([-1.624598481163511e-01, 4.999999999996861e-01, -8.506508083515057e-01]),
                   np.array([4.253254041757529e-01, 3.090169943747534e-01, -8.506508083515057e-01]),
                   np.array([-5.257311121188034e-01, 6.438349236684708e-17, -8.506508083515057e-01]),
                   np.array([-1.624598481163512e-01, -4.999999999996860e-01, -8.506508083515057e-01]),
                   np.array([4.253254041757528e-01, -3.090169943747535e-01, -8.506508083515057e-01])]
  
  rotation_matrix = orientation.rotation_matrix()

  # TODO: Maybe don't do this on the fly every single time.
  for k in range(len(initial_setup)):
    initial_setup[k] = A*(initial_setup[k])

  rotated_setup = []
  for r in initial_setup:
    rotated_setup.append(np.dot(rotation_matrix, r) + np.array(location))
    
  return rotated_setup


def calc_sphere_42_blobs_rot_matrix(r_vectors, location):
  ''' 
  Calculate the matrix R, where the i-th 3x3 block of R gives
  (R_i x) = r_i cross x.
  R will be 3N by 3 (126 x 3). The r vectors point from the center
  of the sphere_42_blobs to the other vertices.
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
  return -1.*rot_matrix


def sphere_42_blobs_force_calculator(location, orientation):
  ''' Force on the sphere_42_blobs center. 
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


def sphere_42_blobs_torque_calculator(location, orientation):
  ''' For now, approximate torque as zero, which is true for the sphere.'''
  return [0., 0., 0.]


def sphere_42_blobs_check_function(location, orientation):
  ''' Check that the sphere_42_blobs is not overlapping the wall. '''
  if location[0][2] < A + VERTEX_A:
    return False
  else:
    return True

def generate_sphere_42_blobs_equilibrium_sample():
  '''Generate a sample sphere_42_blobs location and orientation according to the 
  Gibbs Boltzmann distribution.'''
  while True:
    theta = np.random.normal(0., 1., 4)
    orientation = Quaternion(theta/np.linalg.norm(theta))
    location = [0., 0., np.random.uniform(A, 20.0)]
    accept_prob = gibbs_boltzmann_distribution(location, orientation)/(3.0e-1)
    if accept_prob > 1.:
      print 'Accept probability %s is greater than 1' % accept_prob
    
    if np.random.uniform(0., 1.) < accept_prob:
      return [location, orientation]

  
def gibbs_boltzmann_distribution(location, orientation):
  ''' Calculate Gibbs Boltzmann Distribution at location and orientation.'''
  if location < A:
    return 0.0
  else:
    U = sum(M)*location[2]
    U += (REPULSION_STRENGTH*np.exp(-1.*(location[2] - A)/DEBYE_LENGTH)/
          (location[2] - A))

  return np.exp(-1.*U/KT)
  
@static_var('max_index', 0)
def bin_sphere_42_blobs_height(location, bin_width, height_histogram):
  ''' Bin the z coordinate of location and add to height_histogram.'''
  idx = int(math.floor((location[2])/bin_width))
  if idx < len(height_histogram):
    height_histogram[idx] += 1
  else:
    if idx > bin_sphere_42_blobs_height.max_index:
      bin_sphere_42_blobs_height.max_index = idx
      print "New maximum Index  %d is beyond histogram length " % idx


if __name__ == '__main__':
  # Get command line arguments.
  parser = argparse.ArgumentParser(description='Run Simulation of Uniform '
                                   'sphere_42_blobs with Fixman and RFD '
                                   'schemes, and bin the resulting '
                                   'height distribution.  sphere_42_blobs is '
                                   'affected by gravity, and repulsed from '
                                   'the wall gently.')
  parser.add_argument('-dt', dest='dt', type=float,
                      help='Timestep to use for runs.')
  parser.add_argument('-N', dest='n_steps', type=int,
                      help='Number of steps to take for runs.')
  parser.add_argument('--data-name', dest='data_name', type=str,
                      default='',
                      help='Optional name added to the end of the '
                      'data file.  Useful for multiple runs '
                      'To analyze multiple runs and compute MSD, you must '
                      'specify this, and it must end with "-#" '
                      ' for # starting at 1 and increasing successively. e.g. '
                      'heavy-masses-1, heavy-masses-2, heavy-masses-3 etc.')
  parser.add_argument('--profile', dest='profile', type=bool, default=False,
                      help='True or False: Do we profile this run or not.')

  args=parser.parse_args()
  if args.profile:
    pr = cProfile.Profile()
    pr.enable()


  # Get command line parameters
  dt = args.dt
  n_steps = args.n_steps
  print_increment = max(int(n_steps/20.), 1)

  # Set up logging.
  log_filename = './logs/sphere_42_blobs-dt-%f-N-%d-%s.log' % (
    dt, n_steps, args.data_name)
  progress_logger = logging.getLogger('Progress Logger')
  progress_logger.setLevel(logging.INFO)
  # Add the log message handler to the logger
  logging.basicConfig(filename=log_filename,
                      level=logging.INFO,
                      filemode='w')
  sl = StreamToLogger(progress_logger, logging.INFO)
  sys.stdout = sl
  sl = StreamToLogger(progress_logger, logging.ERROR)
  sys.stderr = sl

  # Script to run the various integrators on the quaternion.
  initial_location = [[0., 0., 1.5]]
  initial_orientation = [Quaternion([1., 0., 0., 0.])]
  fixman_integrator = QuaternionIntegrator(sphere_42_blobs_mobility,
                                           initial_orientation, 
                                           sphere_42_blobs_torque_calculator, 
                                           has_location=True,
                                           initial_location=initial_location,
                                           force_calculator=
                                           sphere_42_blobs_force_calculator)
  fixman_integrator.kT = KT
  fixman_integrator.check_function = sphere_42_blobs_check_function
  rfd_integrator = QuaternionIntegrator(sphere_42_blobs_mobility,
                                           initial_orientation, 
                                           sphere_42_blobs_torque_calculator, 
                                           has_location=True,
                                           initial_location=initial_location,
                                           force_calculator=
                                           sphere_42_blobs_force_calculator)
  rfd_integrator.kT = KT
  rfd_integrator.check_function = sphere_42_blobs_check_function
  
  # Set up histogram for heights.
  bin_width = 1./5.
  fixman_heights = np.zeros(int(15./bin_width))
  rfd_heights = np.zeros(int(15./bin_width))

  start_time = time.time()
  progress_logger.info('Starting run...')
  for k in range(n_steps):
    # Fixman step and bin result.
    fixman_integrator.fixman_time_step(dt)
    bin_sphere_42_blobs_height(fixman_integrator.location[0],
                           bin_width, 
                           fixman_heights)

    # RFD step and bin result.
    rfd_integrator.rfd_time_step(dt)
    bin_sphere_42_blobs_height(rfd_integrator.location[0],
                           bin_width, 
                           rfd_heights)

    if k % print_increment == 0 and k > 0:
      elapsed_time = time.time() - start_time
      log_time_progress(elapsed_time, k, n_steps)

  progress_logger.info('Finished Runs.')
  # Gather data to save.
  heights = [fixman_heights/(n_steps*bin_width),
             rfd_heights/(n_steps*bin_width)]

  height_data = dict()
  # Save parameters just in case they're useful in the future.
  # TODO: Make sure you check all parameters when plotting to avoid
  # issues there.
  height_data['params'] = {'A': A, 'ETA': ETA, 'VERTEX_A': VERTEX_A, 'M': M, 
                           'REPULSION_STRENGTH': REPULSION_STRENGTH,
                           'DEBYE_LENGTH': DEBYE_LENGTH, 'KT': KT,}
  height_data['heights'] = heights
  height_data['buckets'] = (bin_width*np.array(range(len(fixman_heights)))
                            + 0.5*bin_width)
  height_data['names'] = ['Fixman', 'RFD']


  # Optional name for data provided    
  if len(args.data_name) > 0:
    data_name = './data/sphere_42_blobs-dt-%g-N-%d-%s.pkl' % (dt, n_steps, args.data_name)
  else:
    data_name = './data/sphere_42_blobs-dt-%g-N-%d.pkl' % (dt, n_steps)

  with open(data_name, 'wb') as f:
    cPickle.dump(height_data, f)
  
  if args.profile:
    pr.disable()
    s = StringIO.StringIO()
    sortby = 'cumulative'
    ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
    ps.print_stats()
    print s.getvalue()  
