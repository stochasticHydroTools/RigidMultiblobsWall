'''
Set up the mobility, torque, and force functions for the Boomerang
from:
"Chakrabarty et. al - Brownian Motion of Boomerang Colloidal
Particles"

This file defines several functions needed to simulate
the boomerang, and contains several parameters for the run.

Running this script will generate a boomerang trajectory
which can be analyzed with other python scripts in this folder.
'''

import argparse
import cProfile
import numpy as np
import logging
import os
import pstats
import StringIO
import sys
sys.path.append('..')
import time


from config_local import DATA_DIR
from fluids import mobility as mb
from quaternion_integrator.quaternion import Quaternion
from quaternion_integrator.quaternion_integrator import QuaternionIntegrator
from utils import log_time_progress
from utils import StreamToLogger
from utils import write_trajectory_to_txt

# Make sure figures folder exists
if not os.path.isdir(os.path.join(os.getcwd(), 'figures')):
  os.mkdir(os.path.join(os.getcwd(), 'figures'))
# Make sure logs folder exists
if not os.path.isdir(os.path.join(os.getcwd(), 'logs')):
  os.mkdir(os.path.join(os.getcwd(), 'logs'))


# Parameters.  Units are um, s, mg.
A = 0.275   # Radius of individual blobs in um
ETA = 8.9e-4  # Pa s = kg/(m s) = mg/(um s)

# 0.2 g/cm^3 = 0.0000000002 mg/um^3.  Volume is ~1.0238 um^3.  Include gravity in this.
TOTAL_MASS = 1.023825*0.0000000002*(9.8*1.e6)
# Here we have 3 different sets of parameters for mass.  
#  One is the approximate mass of the boomerang in earth's gravity.
#  The second is 3 times earth gravity.  The third is 5 times.
M = [TOTAL_MASS/7. for _ in range(7)] 
# M = [3*TOTAL_MASS/7. for _ in range(7)]
# M = [5*TOTAL_MASS/7. for _ in range(7)]

KT = 300.*1.3806488e-5  # T = 300K
# Made these up for now.
REPULSION_STRENGTH = 1.0
DEBYE_LENGTH = 0.13


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
  
         7 O  
         6 O  
         5 O
           O-O-O-O
           4 3 2 1
   
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
  # Get a rough upper bound on max height.
  max_height = (KT/sum(M))*6.
  while True:
    theta = np.random.normal(0., 1., 4)
    orientation = Quaternion(theta/np.linalg.norm(theta))
    location = [0., 0., np.random.uniform(A, max_height)]
    accept_prob = boomerang_gibbs_boltzmann_distribution(location, orientation)/(7.0e-1)
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
  

def boomerang_check_function(location, orientation):
  ''' 
  Function called after timesteps to check that the boomerang
  is in a viable location (not through the wall).
  '''
  r_vectors = get_boomerang_r_vectors(location[0], orientation[0])
  for k in range(7):
    if r_vectors[k][2] < (A + 0.05): 
      return False
  return True
  
  
if __name__ == '__main__':
  # Get command line arguments.
  parser = argparse.ArgumentParser(description='Run Simulation of Boomerang '
                                   'particle with Fixman, EM, and RFD '
                                   'schemes, and save trajectory.  Boomerang '
                                   'is affected by gravity and repulsed from '
                                   'the wall gently.')
  parser.add_argument('-dt', dest='dt', type=float,
                      help='Timestep to use for runs.')
  parser.add_argument('-N', dest='n_steps', type=int,
                      help='Number of steps to take for runs.')
  parser.add_argument('--data-name', dest='data_name', type=str,
                      default='',
                      help='Optional name added to the end of the '
                      'data file.  Useful for multiple runs. '
                      'To analyze multiple runs and compute MSD, you must '
                      'specify this, and it must end with "-#" '
                      ' for # starting at 1 and increasing successively. e.g. '
                      'heavy-masses-1, heavy-masses-2, heavy-masses-3 etc.')
  parser.add_argument('--profile', dest='profile', type=bool, default=False,
                      help='True or False: Profile this run or not.')

  args=parser.parse_args()
  if args.profile:
    pr = cProfile.Profile()
    pr.enable()

  # Get command line parameters
  dt = args.dt
  n_steps = args.n_steps
  print_increment = max(int(n_steps/20.), 1)

  # Set up logging.
  log_filename = './logs/boomerang-dt-%f-N-%d-%s.log' % (
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
  sample = generate_boomerang_equilibrium_sample()
  initial_location = [sample[0]]
  initial_orientation = [sample[1]]
  fixman_integrator = QuaternionIntegrator(boomerang_mobility,
                                           initial_orientation, 
                                           boomerang_torque_calculator, 
                                           has_location=True,
                                           initial_location=initial_location,
                                           force_calculator=
                                           boomerang_force_calculator)
  fixman_integrator.kT = KT
  fixman_integrator.check_function = boomerang_check_function
  rfd_integrator = QuaternionIntegrator(boomerang_mobility,
                                        initial_orientation, 
                                        boomerang_torque_calculator, 
                                        has_location=True,
                                        initial_location=initial_location,
                                        force_calculator=
                                        boomerang_force_calculator)
  rfd_integrator.kT = KT
  rfd_integrator.check_function = boomerang_check_function
  em_integrator = QuaternionIntegrator(boomerang_mobility,
                                        initial_orientation, 
                                        boomerang_torque_calculator, 
                                        has_location=True,
                                        initial_location=initial_location,
                                        force_calculator=
                                        boomerang_force_calculator)
  em_integrator.kT = KT
  em_integrator.check_function = boomerang_check_function

  # Lists of location and orientation.
  fixman_trajectory = [[], []]
  rfd_trajectory = [[], []]
  em_trajectory = [[], []]



  start_time = time.time()
  for k in range(n_steps):
    # Fixman step and bin result.
    fixman_integrator.fixman_time_step(dt)
    fixman_trajectory[0].append(fixman_integrator.location[0])
    fixman_trajectory[1].append(fixman_integrator.orientation[0].entries)

    rfd_integrator.rfd_time_step(dt)
    rfd_trajectory[0].append(rfd_integrator.location[0])
    rfd_trajectory[1].append(rfd_integrator.orientation[0].entries)

    # EM step and bin result.
    em_integrator.additive_em_time_step(dt)
    em_trajectory[0].append(em_integrator.location[0])
    em_trajectory[1].append(em_integrator.orientation[0].entries)

    if k % print_increment == 0:
      elapsed_time = time.time() - start_time
      print 'At step %s out of %s' % (k, n_steps)
      log_time_progress(elapsed_time, k, n_steps)
      

  elapsed_time = time.time() - start_time
  if elapsed_time > 60:
    progress_logger.info('Finished timestepping. Total Time: %.2f minutes.' % 
                         (float(elapsed_time)/60.))
  else:
    progress_logger.info('Finished timestepping. Total Time: %.2f seconds.' % 
                         float(elapsed_time))
  progress_logger.info('Fixman Rejection rate: %s' % 
                       (float(fixman_integrator.rejections)/
                        float(fixman_integrator.rejections + n_steps)))
  progress_logger.info('RFD Rejection rate: %s' % 
                       (float(rfd_integrator.rejections)/
                        float(rfd_integrator.rejections + n_steps)))

  # Gather parameters to save
  params = {'A': A, 'ETA': ETA, 'M': M,
            'REPULSION_STRENGTH': REPULSION_STRENGTH,
            'DEBYE_LENGTH': DEBYE_LENGTH, 'dt': dt, 'n_steps': n_steps}

  # Set up naming for data files for trajectories.
  if len(args.data_name) > 0:
    def generate_trajectory_name(scheme):
      trajectory_dat_name = 'boomerang-trajectory-dt-%g-N-%d-scheme-%s-%s.txt' % (
        dt, n_steps, scheme, args.data_name)
      return trajectory_dat_name
  else:
    def generate_trajectory_name(scheme):
      trajectory_dat_name = 'boomerang-trajectory-dt-%g-N-%d-scheme-%s.txt' % (
        dt, n_steps, scheme)
      return trajectory_dat_name

  fixman_data_file = os.path.join(
    DATA_DIR, 'boomerang', generate_trajectory_name('FIXMAN'))
  write_trajectory_to_txt(fixman_data_file, fixman_trajectory, params)

  rfd_data_file = os.path.join(
    DATA_DIR, 'boomerang', generate_trajectory_name('RFD'))
  write_trajectory_to_txt(rfd_data_file, rfd_trajectory, params)

  em_data_file = os.path.join(
    DATA_DIR, 'boomerang', generate_trajectory_name('EM'))
  write_trajectory_to_txt(em_data_file, em_trajectory, params)

  if args.profile:
    pr.disable()
    s = StringIO.StringIO()
    sortby = 'cumulative'
    ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
    ps.print_stats()
    print s.getvalue()

 
