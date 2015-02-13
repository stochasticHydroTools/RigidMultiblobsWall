''' Script to run the Icosohedron and calculate the MSD. '''

import argparse
import cPickle
import cProfile
import logging
import numpy as np
import os

import icosohedron as ic
from quaternion_integrator.quaternion import Quaternion
from quaternion_integrator.quaternion_integrator import QuaternionIntegrator
from utils import StreamToLogger
from utils import MSDStatistics


def calc_icosohedron_msd_from_equilibrium(initial_orientation,
                                     scheme,
                                     dt,
                                     end_time,
                                     n_steps,
                                     location=None,
                                     n_runs=4):
  ''' 
  Do a few long runs, and along the way gather statistics
  about the average rotational Mean Square Displacement 
  by calculating it from time lagged data.  This is icosohedron specific.
  args:
    initial_orientation: list of length 1 quaternion where 
                 the run starts.  This shouldn't effect results.
    scheme: FIXMAN, RFD, or EM, scheme for the integrator to use.
    dt:  float, timestep used by the integrator.
    end_time: float, how much time to track the evolution of the MSD.
    n_steps:  How many total steps to take.
    location: initial location of icosohedron.
    n_runs:  How many separate runs to do in order to get std deviation.  
             4 should be fine.
  '''
  progress_logger = logging.getLogger('Progress Logger')
  burn_in = int(end_time*4./dt)
  rot_msd_list = []
  print_increment = n_steps/20
  dim = 6
  for run in range(n_runs):
    integrator = QuaternionIntegrator(ic.icosohedron_mobility,
                                      initial_orientation, 
                                      ic.icosohedron_torque_calculator,
                                      has_location=True,
                                      initial_location=location,
                                      force_calculator=
                                      ic.icosohedron_force_calculator)
    integrator.kT = ic.KT
    integrator.check_function = ic.icosohedron_check_function

    trajectory_length = int(end_time/dt) + 1
    if trajectory_length > n_steps:
      raise Exception('Trajectory length is greater than number of steps.  '
                      'Do a longer run.')
    lagged_trajectory = []   # Store rotation matrices to avoid re-calculation.
    lagged_location_trajectory = [] 
    average_rotational_msd = np.array([np.zeros((dim, dim)) 
                                       for _ in range(trajectory_length)])
    for step in range(burn_in + n_steps):
      if scheme == 'FIXMAN':
        integrator.fixman_time_step(dt)
      elif scheme == 'RFD':
        integrator.rfd_time_step(dt)
      elif scheme == 'EM':
        integrator.additive_em_time_step(dt)

      if step > burn_in:
        lagged_trajectory.append(integrator.orientation[0].rotation_matrix())
        lagged_location_trajectory.append(integrator.location[0])

      if len(lagged_trajectory) > trajectory_length:
        lagged_trajectory = lagged_trajectory[1:]
        lagged_location_trajectory = lagged_location_trajectory[1:]
        for k in range(trajectory_length):
          current_rot_msd = (calc_total_icosohedron_msd(
            lagged_location_trajectory[0],
            lagged_trajectory[0],
            lagged_location_trajectory[k],
            lagged_trajectory[k]))
          average_rotational_msd[k] += current_rot_msd

      if (step % print_increment) == 0:
        progress_logger.info('At step: %d in run %d of %d' % (step, run + 1, n_runs))

    progress_logger.info('Integrator Rejection rate: %s' % 
                         (float(integrator.rejections)/
                          float(integrator.rejections + n_steps)))
    average_rotational_msd = average_rotational_msd/(n_steps - trajectory_length)
    rot_msd_list.append(average_rotational_msd)
  
  progress_logger.info('Done with Equilibrium MSD runs.')
  # Average results to get time, mean, and std of rotational MSD.
  # For now, std = 0.  Will figure out a good way to calculate this later.
  results = [[], [], []]
  results[0] = np.arange(0, trajectory_length)*dt
  results[1] = np.mean(rot_msd_list, axis=0)
  results[2] = np.std(rot_msd_list, axis=0)/np.sqrt(n_runs)

  progress_logger = logging.getLogger('progress_logger')  
  progress_logger.info('Rejection Rate: %s' % 
                       (float(integrator.rejections)/
                        float(n_steps*n_runs + integrator.rejections)))
  return results


def calc_total_icosohedron_msd(initial_location, initial_rotation, 
                               location, rotation):
  ''' 
  Calculate 6x6 MSD from initial location and rotation matrix, 
  and final location and final rotation matrix.
  '''
  dx = np.array(location) - np.array(initial_location)
  # Get rotational displacement, u_hat.
  u_hat = np.zeros(3)
  for i in range(3):
    e = np.zeros(3)
    e[i] = 1.
    u_hat += 0.5*np.cross(np.inner(initial_rotation, e),
                          np.inner(rotation, e))

  displacement = np.concatenate([dx, u_hat])
  return np.outer(displacement, displacement)


if __name__ == '__main__':
  # Get command line arguments.
  parser = argparse.ArgumentParser(description='Run Simulations to calculate '
                                   'icosohedron rotational MSD using the '
                                   'Fixman, Random Finite Difference, and '
                                   'Euler-Maruyama schemes at a given '
                                   'timestep')
  parser.add_argument('-dt', dest='dt', type=float,
                      help='Timestep to use for runs. specify as a list, '
                      'e.g. -dt 0.8')
  parser.add_argument('-N', dest='n_steps', type=int,
                      help='Number of steps to take for runs')
  parser.add_argument('-end', dest='end_time', type=float, default = 128.0,
                      help='How far to calculate the time dependent MSD.')
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
  
  # Set initial conditions.
  initial_orientation = [Quaternion([1., 0., 0., 0.])]
  initial_location = [[0., 0., 4.0]]
  
  # Extract parameters from Arguments.
  scheme = 'RFD'
  dt = args.dt
  end_time = args.end_time
  n_steps = args.n_steps
  # Set up buckets for histogram.
  bin_width = 1./10.
  buckets = np.arange(0, int(20./bin_width))*bin_width + bin_width/2.

  # Set up logging.
  log_filename = './logs/icosohedron-rotation-dt-%f-N-%d-%s.log' % (
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

  height_histogram_run = np.zeros(len(buckets))
  params = {'M': ic.M, 'A': ic.A, 'VERTEX_A': ic.VERTEX_A,
            'REPULSION_STRENGTH': ic.REPULSION_STRENGTH, 
            'DEBYE_LENGTH': ic.DEBYE_LENGTH, 'KT': ic.KT}
  msd_statistics = MSDStatistics(['FIXMAN'], [dt], params)

  run_data = calc_icosohedron_msd_from_equilibrium(
    initial_orientation,
    scheme,
    dt, 
    end_time,
    n_steps,
    location=initial_location)

  progress_logger.info('Completed equilibrium runs.')
  msd_statistics.add_run(scheme, dt, run_data)

  data_name = './data/icosohedron-msd-dt-%s-N-%d-%s.pkl' % (
    dt, n_steps, args.data_name)

  with open(data_name, 'wb') as f:
    cPickle.dump(msd_statistics, f)
