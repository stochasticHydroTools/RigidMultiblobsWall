''' 
Estimate the rotational MSD based on:

u_hat(dt) = \sum_i u_i(0) cross u_i(dt)

For what it's worth, the derivative,  

msd slope = <u_hat_i u_hat_j>/dt

should go to 2kBT * Mobility as dt -> 0.
Evaluate mobility at point with no torque, and take several steps to
get a curve of MSD(t).  Alternatively calculate the time dependent MSD 
at equilibrium by doing a long run and calculating MSD from the time 
lagged trajectory, then average.
'''
import argparse
import cPickle
import logging
import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot
import os
import sys
sys.path.append('..')
import time

from quaternion_integrator.quaternion import Quaternion
from quaternion_integrator.quaternion_integrator import QuaternionIntegrator
import tetrahedron as tdn
import tetrahedron_free as tf

class MSDStatistics(object):
  '''
  Class to hold the means and std deviations of the time dependent
  MSD for multiple schemes and timesteps.  data is a dictionary of
  dictionaries, holding runs indexed by scheme and timestep in that 
  order.
  Each run is organized as a list of 3 arrays: [time, mean, std]
  mean and std are matrices (the rotational MSD).
  '''
  def __init__(self, schemes, dts, params):
    self.data = {}
    self.params = params

  def add_run(self, scheme_name, dt, run_data):
    ''' 
    Add a run.  Create the entry if need be. 
    run is organized as a list of 3 arrays: [time, mean, std]
    In that order.
    '''
    if scheme_name not in self.data:
      self.data[scheme_name] = dict()

    self.data[scheme_name][dt] = run_data

def calculate_msd_from_fixed_initial_condition(initial_orientation,
                                               scheme,
                                               dt,
                                               end_time,
                                               n_runs,
                                               has_location=False,
                                               location=None):
  ''' 
  Calculate MSD by starting at an initial condition, and doing short runs
  to time = end_time.
  Average over these trajectories to get the curve of MSD v. time.
  '''
  if has_location:
    mobility = tf.free_tetrahedron_mobility
    torque_calculator = tf.free_gravity_torque_calculator
    KT = tf.KT
  else:
    mobility = tdn.tetrahedron_mobility
    torque_calculator = tdn.gravity_torque_calculator
    KT = 1.0

  integrator = QuaternionIntegrator(mobility,
                                    initial_orientation, 
                                    torque_calculator,
                                    has_location=has_location,
                                    initial_location=location,
                                    force_calculator=
                                    tf.free_gravity_force_calculator)
  integrator.kT = KT
  if has_location:
    integrator.check_function = tf.check_particles_above_wall
  n_steps = int(end_time/dt) + 1
  trajectories = []
  for run in range(n_runs):
    integrator.orientation = initial_orientation
    integrator.location = initial_location
    trajectories.append([])
    # Calculate rotational MSD and add to trajectory.
    if has_location:
      trajectories[run].append(
        calc_total_msd(initial_location[0], initial_orientation[0],
                       integrator.location[0], integrator.orientation[0]))
    else:
      trajectories[run].append(
        calc_rotational_msd(initial_orientation[0],
                            integrator.orientation[0]))
    for step in range(n_steps):
      if scheme == 'FIXMAN':
        integrator.fixman_time_step(dt)
      elif scheme == 'RFD':
        integrator.rfd_time_step(dt)
      elif scheme == 'EM':
        integrator.additive_em_time_step(dt)

      if has_location:
        trajectories[run].append(
          calc_total_msd(initial_location[0], initial_orientation[0],
                         integrator.location[0], integrator.orientation[0]))
      else:
        trajectories[run].append(
          calc_rotational_msd(initial_orientation[0],
                              integrator.orientation[0]))
  # Average results to get time, mean, and std of rotational MSD.
  results = [[], [], []]
  step = 0
  for step in range(n_steps):
    time = dt*step
    mean_msd = np.mean([trajectories[run][step] for run in range(n_runs)], axis=0)
    std_msd = np.std([trajectories[run][step] for run in range(n_runs)], axis=0)
    results[0].append(time)
    results[1].append(mean_msd)
    results[2].append(std_msd/np.sqrt(n_runs))

  progress_logger = logging.getLogger('progress_logger')  
  progress_logger.info('Rejection Rate: %s' % 
                       (float(integrator.rejections)/
                        float(n_steps*n_runs + integrator.rejections)))

  return results


def calc_rotational_msd_from_equilibrium(initial_orientation,
                                         scheme,
                                         dt,
                                         end_time,
                                         n_steps,
                                         has_location=False,
                                         location=None):

  ''' 
  Do one long run, and along the way gather statistics
  about the average rotational Mean Square Displacement 
  by calculating it from time lagged data. 
  args:
    initial_orientation: list of length 1 quaternion where 
                 the run starts.  This shouldn't effect results.
    scheme: FIXMAN, RFD, or EM, scheme for the integrator to use.
    dt:  float, timestep used by the integrator.
    end_time: float, how much time to track the evolution of the MSD.
    n_steps:  How many total steps to take.
    has_location: boolean, do we let the tetrahedron move and track location?
    location: initial location of tetrahedron, only used if has_location = True.
  '''
  if has_location:
    mobility = tf.free_tetrahedron_mobility
    torque_calculator = tf.free_gravity_torque_calculator
    KT = tf.KT
    dim = 6
  else:
    mobility = tdn.tetrahedron_mobility
    torque_calculator = tdn.gravity_torque_calculator
    KT = 1.0
    dim = 3

  integrator = QuaternionIntegrator(mobility,
                                    initial_orientation, 
                                    torque_calculator,
                                    has_location=has_location,
                                    initial_location=location,
                                    force_calculator=
                                    tf.free_gravity_force_calculator)
  integrator.kT = KT

  trajectory_length = int(end_time/dt) + 1
  if trajectory_length > n_steps:
    raise Exception('Trajectory length is greater than number of steps.  '
                    'Do a longer run.')
  lagged_trajectory = []
  lagged_location_trajectory = []
  average_rotational_msd = np.array([np.zeros((dim, dim)) 
                                     for _ in range(trajectory_length)])
  for step in range(n_steps):
    if scheme == 'FIXMAN':
      integrator.fixman_time_step(dt)
    elif scheme == 'RFD':
      integrator.rfd_time_step(dt)
    elif scheme == 'EM':
      integrator.additive_em_time_step(dt)


    lagged_trajectory.append(integrator.orientation[0])
    if has_location:
      lagged_location_trajectory.append(integrator.location[0])

    if len(lagged_trajectory) > trajectory_length:
      lagged_trajectory = lagged_trajectory[1:]
      if has_location:
        lagged_location_trajectory = lagged_location_trajectory[1:]
      for k in range(trajectory_length):
        if has_location:
          average_rotational_msd[k] += (calc_total_msd(
            lagged_location_trajectory[0],
            lagged_trajectory[0],
            lagged_location_trajectory[k],
            lagged_trajectory[k]))
        else:
          average_rotational_msd[k] += (calc_rotational_msd(
            lagged_trajectory[0],
            lagged_trajectory[k]))
    
  average_rotational_msd = average_rotational_msd/(n_steps - trajectory_length)
  
  # Average results to get time, mean, and std of rotational MSD.
  # For now, std = 0.  Will figure out a good way to calculate this later.
  results = [[], [], []]
  results[0] = np.arange(0, trajectory_length)*dt
  results[1] = average_rotational_msd
  results[2] = np.zeros((trajectory_length, dim, dim))
      
  return results

  
def calc_rotational_msd(initial_orientation, orientation):
  ''' 
  Calculate the rotational MSD from an initial configuration to
  a final orientation.  Orientations are given as single quaternion objects.
  '''  
  u_hat = np.zeros(3)
  rot_matrix = orientation.rotation_matrix()
  original_rot_matrix = initial_orientation.rotation_matrix()
  for i in range(3):
    e = np.zeros(3)
    e[i] = 1.
    u_hat += 0.5*np.cross(np.inner(original_rot_matrix, e),
                          np.inner(rot_matrix, e))
  return np.outer(u_hat, u_hat)


def calc_total_msd(initial_location, initial_orientation, 
                   location, orientation):
  ''' Calculate 6x6 MSD including orientation and location. '''
  u_hat = np.zeros(3)
  rot_matrix = orientation.rotation_matrix()
  original_rot_matrix = initial_orientation.rotation_matrix()
  original_center_of_mass = tf.get_free_center_of_mass(initial_location, 
                                                       initial_orientation)
  final_center_of_mass = tf.get_free_center_of_mass(location, 
                                                    orientation)
  for i in range(3):
    e = np.zeros(3)
    e[i] = 1.
    u_hat += 0.5*np.cross(np.inner(original_rot_matrix, e),
                          np.inner(rot_matrix, e))
    
  dx = np.array(final_center_of_mass) - np.array(original_center_of_mass)
  displacement = np.concatenate([dx, u_hat])
  return np.outer(displacement, displacement)


def plot_msd_convergence(dts, msd_list, names):
  ''' 
  Log-log plot of error in MSD v. dt.  This is for single
  step MSD compared to theoretical MSD slope (mobility).
  '''
  fig = pyplot.figure()
  ax = fig.add_subplot(1, 1, 1)
  for k in range(len(msd_list)):
    pyplot.plot(dts, msd_list[k], label=names[k])

  first_order = msd_list[0][0]*((np.array(dts)))/(dts[0])
  pyplot.plot(dts, first_order, 'k--', label='1st Order')
  pyplot.ylabel('Error')
  pyplot.xlabel('dt')
  pyplot.legend(loc='best', prop={'size': 9})
  pyplot.title('Error in Rotational MSD')
  ax.set_yscale('log')
  ax.set_xscale('log')
  pyplot.savefig('./plots/RotationalMSD.pdf')


def log_time_progress(elapsed_time, time_units, total_time_units):
  ''' Write elapsed time and expected duration to progress log.'''
  progress_logger = logging.getLogger('progress_logger')  
  expected_duration = elapsed_time*total_time_units/time_units
  if elapsed_time > 60.0:
    progress_logger.info('Elapsed Time: %.2f Minutes.' % 
                         (float(elapsed_time/60.)))
  else:
    progress_logger.info('Elapsed Time: %.2f Seconds' % float(elapsed_time))
  if expected_duration > 60.0:
    progress_logger.info('Expected Duration: %.2f Minutes.' % 
                         (float(expected_duration/60.)))
  else:
      progress_logger.info('Expected Duration: %.2f Seconds' % 
                           float(expected_duration))


if __name__ == "__main__":
  # Get command line arguments.
  parser = argparse.ArgumentParser(description='Run Simulations to calculate '
                                   'fixed tetrahedron rotational MSD using the '
                                   'Fixman, Random Finite Difference, and '
                                   'Euler-Maruyama schemes at multiple '
                                   'timesteps.')
  parser.add_argument('-dts', dest='dts', type=float, nargs='+',
                      help='Timesteps to use for runs. specify as a list, e.g. '
                      '-dts 4.0 2.0')
  parser.add_argument('-N', dest='n_steps', type=int,
                      help='Number of steps to take for runs or number of runs '
                      'to perform in the case of fixed initial condition.')
  parser.add_argument('-end', dest='end_time', type=float, default = 128.0,
                      help='How far to calculate the time dependent MSD.')
  parser.add_argument('-initial', dest='initial', type=bool, default=False,
                      help='Indicate whether to do multiple runs starting at '
                      'a fixed initial condition.  If false, will do one '
                      'run and calculate the average time dependent MSD at '
                      'equilibrium.')
  parser.add_argument('-location', dest='has_location', type=bool,
                      default=False,
                      help='Whether or not the tetrahedron is allowed '
                      'to move (has a location).  If it is allowed to move, '
                      'then the MSD includes translational and rotational '
                      'displacement.')
  parser.add_argument('--data-name', dest='data_name', type=str,
                      default='',
                      help='Optional name added to the end of the '
                      'data file.  Useful for multiple runs '
                      '(--data_name=run-1).')
  args = parser.parse_args()

  # Set masses and initial position.  
  # These only matter if there is no location.
  tdn.M1 = 0.1
  tdn.M2 = 0.1
  tdn.M3 = 0.1
  tf.M1 = 0.225
  tf.M2 = 0.225
  tf.M3 = 0.225
  tf.M4 = 0.225
  initial_orientation = [Quaternion([1., 0., 0., 0.])]
  initial_location = [[0., 0., 4.0]]

  schemes = ['FIXMAN', 'RFD']
  if not args.has_location:
    schemes.append('EM')

  dts = args.dts
  end_time = args.end_time
  n_runs = args.n_steps

  # Setup logging.
  # Make directory for logs if it doesn't exist.
  if not os.path.isdir(os.path.join(os.getcwd(), 'logs')):
    os.mkdir(os.path.join(os.getcwd(), 'logs'))
  log_filename = './logs/rotational-msd-initial-%s-location-%s-dts-%s-N-%d-%s.log' % (
    args.initial, args.has_location, dts, n_runs, args.data_name)
  progress_logger = logging.getLogger('progress_logger')
  progress_logger.setLevel(logging.INFO)
  # Add the log message handler to the logger
  logging.basicConfig(filename=log_filename,
                      level=logging.INFO,
                      filemode='w')
  sl = tdn.StreamToLogger(progress_logger, logging.INFO)
  sys.stdout = sl
  sl = tdn.StreamToLogger(progress_logger, logging.ERROR)
  sys.stderr = sl

  if args.has_location:
    params = {'M1': tf.M1, 'M2': tf.M2, 'M3': tf.M3, 'M4': tf.M4,
              'A': tf.A, 'REPULSION_STRENGTH': tf.REPULSION_STRENGTH,
              'REPULSION_CUTOFF': tf.REPULSION_CUTOFF,
              'KT': tf.KT, 'end_time': end_time}
  else:
    params = {'M1': tf.M1, 'M2': tf.M2, 'M3': tf.M3,
              'A': tf.A, 'KT': tf.KT, 'end_time': end_time}
    
  msd_statistics = MSDStatistics(schemes, dts, params)
  # Measure time, and estimate how long runs will take.
  # One time unit is n_runs timesteps.
  start_time = time.time()
  total_time_units = sum(end_time/np.array(dts))*len(schemes)
  time_units = 0
  for scheme in schemes:
    for dt in dts:
      if args.initial:
        run_data = calculate_msd_from_fixed_initial_condition(
          initial_orientation,
          scheme,
          dt,
          end_time,
          n_runs,
          has_location=args.has_location,
          location=initial_location)
      else:
        run_data = calc_rotational_msd_from_equilibrium(initial_orientation,
                                                        scheme,
                                                        dt,
                                                        end_time,
                                                        n_runs,
                                                        has_location=
                                                        args.has_location,
                                                        location=
                                                        initial_location)
      msd_statistics.add_run(scheme, dt, run_data)
      elapsed_time = time.time() - start_time
      time_units += end_time/dt
      progress_logger.info('finished timestepping dt= %f for scheme %s' % (
      dt, scheme))
      log_time_progress(elapsed_time, time_units, total_time_units)

  progress_logger.info('Runs complete.')

  # Make directory for data if it doesn't exist.
  if not os.path.isdir(os.path.join(os.getcwd(), 'data')):
    os.mkdir(os.path.join(os.getcwd(), 'data'))

  # Optional name for data provided
  data_name = args.data_name
  if len(data_name) > 3:
    data_name = './data/rot-msd-initial-%s-location-%s-dt-%s-N-%d-%s.pkl' % (
      args.initial, args.has_location, dts, n_runs, data_name)
  else:
    data_name = './data/rot-msd-initial-%s-location-%s-dt-%s-N-%d.pkl' % (
      args.initial, args.has_location, dts, n_runs)

  with open(data_name, 'wb') as f:
    cPickle.dump(msd_statistics, f)

