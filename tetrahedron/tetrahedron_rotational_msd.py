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
import cProfile
import logging
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot
import numpy as np
import os
import pstats
import sys
import StringIO
import time

import tetrahedron as tdn
import tetrahedron_free as tf

from quaternion_integrator.quaternion import Quaternion
from quaternion_integrator.quaternion_integrator import QuaternionIntegrator
from utils import static_var
from utils import MSDStatistics
from utils import log_time_progress


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


def calc_total_msd_from_matrix_and_com(original_center_of_mass, original_rot_matrix, 
                                       final_center_of_mass, rot_matrix):
  ''' 
  Calculate 6x6 MSD including orientation and location.  This is calculated from
  precomputed center of mass and rotation matrix data to avoid repeating computation.
  '''
  u_hat = np.zeros(3)
  for i in range(3):
    e = np.zeros(3)
    e[i] = 1.
    u_hat += 0.5*np.cross(np.inner(original_rot_matrix, e),
                          np.inner(rot_matrix, e))
    
  dx = np.array(final_center_of_mass) - np.array(original_center_of_mass)
  displacement = np.concatenate([dx, u_hat])
  return np.outer(displacement, displacement)


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
  progress_logger = logging.getLogger('progress_logger')  
  print_increment = n_runs/20.
  if has_location:
    mobility = tf.free_tetrahedron_mobility
    torque_calculator = tf.free_gravity_torque_calculator
    KT = tf.KT
  else:
    mobility = tdn.tetrahedron_mobility
    torque_calculator = tdn.gravity_torque_calculator
    KT = 0.2
  
  r_vectors = tdn.get_r_vectors(initial_orientation[0])
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
  # Why do I have to do this?
  start_time = time.time()
  progress_logger.info('Started runs...')
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
        calc_rotational_msd_from_likely_position(
          integrator.orientation[0]))
    for step in range(n_steps):
      if scheme == 'FIXMAN':
        integrator.fixman_time_step(dt)
      elif scheme == 'RFD':
        integrator.rfd_time_step(dt)
      elif scheme == 'EM':
        integrator.additive_em_time_step(dt)
              
      if has_location:
#        raise NotImplementedError('Need to fix calc total msd for initial '
#                                  'position run with location.')
        trajectories[run].append(
          calc_total_msd(initial_location[0], initial_orientation[0],
                         integrator.location[0], integrator.orientation[0]))
      else:
        trajectories[run].append(
          calc_rotational_msd_from_likely_position(
            integrator.orientation[0]))

    if (run % print_increment) == 0 and (run > 0):
      elapsed_time = time.time() - start_time
      progress_logger.info('finished run %s' % run)
      log_time_progress(elapsed_time, run, n_runs)
      
  # Average results to get time, mean, and std of rotational MSD.
  results = [[], [], []]
  step = 0
  for step in range(n_steps):
    current_time = dt*step
    mean_msd = np.mean([trajectories[run][step] for run in range(n_runs)], axis=0)
    std_msd = np.std([trajectories[run][step] for run in range(n_runs)], axis=0)
    results[0].append(current_time)
    results[1].append(mean_msd)
    results[2].append(std_msd/np.sqrt(n_runs))

 
  progress_logger.info('Rejection Rate: %s' % 
                       (float(integrator.rejections)/
                        float(n_steps*n_runs + integrator.rejections)))

#  for l in range(3):
#    pyplot.figure(l)
#    pyplot.plot(results[0], [dat[l] for dat in particle_position], 'g--', label='Python')
#    pyplot.plot(floren_time, [dat[l] for dat in floren_position], 'r:', label='IBAMR')
#    pyplot.legend(loc='best', prop={'size': 9})
#    pyplot.savefig('./figures/BlobPosition-Coordinate-' + str(l) + '.pdf')
      
  return results


def calc_rotational_msd_from_equilibrium(initial_orientation,
                                         scheme,
                                         dt,
                                         end_time,
                                         n_steps,
                                         has_location=False,
                                         location=None,
                                         n_runs=8):
  ''' 
  Do a few long run, and along the way gather statistics
  about the average rotational Mean Square Displacement 
  by calculating it from time lagged data.  This is Tetrahedron Specific.
  args:
    initial_orientation: list of length 1 quaternion where 
                 the run starts.  This shouldn't effect results.
    scheme: FIXMAN, RFD, or EM, scheme for the integrator to use.
    dt:  float, timestep used by the integrator.
    end_time: float, how much time to track the evolution of the MSD.
    n_steps:  How many total steps to take.
    has_location: boolean, do we let the tetrahedron move and track location?
    location: initial location of tetrahedron, only used if has_location = True.
    n_runs:  How many separate runs to do in order to estimate std deviation.  
  '''
  burn_in = 0
  progress_logger = logging.getLogger('Progress Logger')
  # Instead of burn in we generate a sample start point using accept-reject.
  if has_location:
    mobility = tf.free_tetrahedron_mobility
    torque_calculator = tf.free_gravity_torque_calculator
    KT = tf.KT
    dim = 6
    sample = tf.generate_free_equilibrium_sample()
    location = [sample[0]]
    initial_orientation = [sample[1]]
  else:
    mobility = tdn.tetrahedron_mobility
    torque_calculator = tdn.gravity_torque_calculator
    KT = tdn.KT
    dim = 3

  rot_msd_list = []
  print_increment = n_steps/10
  start_time = time.time()
  for run in range(n_runs):
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

    # choose number of steps to take before saving data.
    # Want 100 points on our plot.
    #HACK
    data_interval = int((end_time/dt)/100.)
    trajectory_length = int(end_time/dt)
    if data_interval == 0:
      data_interval = 1
    data_interval = 1

    if trajectory_length*data_interval > n_steps:
      raise Exception('Trajectory length is greater than number of steps.  '
                      'Do a longer run.')
    lagged_trajectory = []   # Store rotation matrices to avoid re-calculation.
    lagged_location_trajectory = []  # Locations of geometric center
    average_rotational_msd = np.array([np.zeros((dim, dim)) 
                                     for _ in range(trajectory_length)])
    for step in range(n_steps + burn_in):
      if scheme == 'FIXMAN':
        integrator.fixman_time_step(dt)
      elif scheme == 'RFD':
        integrator.rfd_time_step(dt)
      elif scheme == 'EM':
        integrator.additive_em_time_step(dt)

      if step % data_interval == 0:
        lagged_trajectory.append(integrator.orientation[0].rotation_matrix())
        if has_location:
          geometric_center = tf.get_free_geometric_center(integrator.location[0], 
                                                               integrator.orientation[0])
          lagged_location_trajectory.append(geometric_center)

      if len(lagged_trajectory) > trajectory_length:
        lagged_trajectory = lagged_trajectory[1:]
        if has_location:
          lagged_location_trajectory = lagged_location_trajectory[1:]
        for k in range(trajectory_length):
          if has_location:
            current_rot_msd = (calc_total_msd_from_matrix_and_com(
                lagged_location_trajectory[0],
                lagged_trajectory[0],
                lagged_location_trajectory[k],
                lagged_trajectory[k]))
            average_rotational_msd[k] += current_rot_msd
          else:
            current_rot_msd = (calc_rotational_msd(
                lagged_trajectory[0],
                lagged_trajectory[k]))
            average_rotational_msd[k] += current_rot_msd
      if (step % print_increment) == 0:
        progress_logger.info(
          'At step: %d in run %d of %d ' %
          (step, run + 1, n_runs))
        if step > 0:
          time_elapsed = time.time() - start_time
          log_time_progress(time_elapsed, step + (burn_in + n_steps)*run,
                            (burn_in + n_steps)*n_runs)

    progress_logger.info('Integrator Rejection rate: %s' % 
                         (float(integrator.rejections)/
                          float(integrator.rejections + n_steps)))
    average_rotational_msd = average_rotational_msd/(n_steps/data_interval - trajectory_length)
    rot_msd_list.append(average_rotational_msd)
  progress_logger.info('Done with Equilibrium MSD runs.')
  # Average results to get time, mean, and std of rotational MSD.
  # For now, std = 0.  Will figure out a good way to calculate this later.
  results = [[], [], []]
  results[0] = np.arange(0, trajectory_length)*dt*data_interval
  results[1] = np.mean(rot_msd_list, axis=0)
  results[2] = np.std(rot_msd_list, axis=0)/np.sqrt(n_runs)

  progress_logger = logging.getLogger('progress_logger')  
  progress_logger.info('Rejection Rate: %s' % 
                       (float(integrator.rejections)/
                        float(n_steps*n_runs + integrator.rejections)))
  return results

  
def calc_rotational_msd(original_rot_matrix, rot_matrix):
  ''' 
  Calculate the rotational MSD from an initial configuration to
  a final orientation.  Orientations are given as single quaternion objects.
  '''  
  u_hat = np.zeros(3)
  for i in range(3):
    e = np.zeros(3)
    e[i] = 1.
    u_hat += 0.5*np.cross(np.inner(original_rot_matrix, e),
                          np.inner(rot_matrix, e))
  return np.outer(u_hat, u_hat)


def calc_rotational_msd_from_likely_position(orientation):
  ''' Calculate rotational MSD from the quaternion Identity.'''
  rot_matrix = orientation.rotation_matrix()
  u_hat = np.zeros(3)
  for i in range(3):
    e = np.zeros(3)
    e[i] = 1.
    u_hat += 0.5*np.cross(e, np.inner(rot_matrix, e))

  msd = np.outer(u_hat, u_hat)
  return msd


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


if __name__ == "__main__":
  # Get command line arguments.
  parser = argparse.ArgumentParser(description='Run Simulations to calculate '
                                   'fixed tetrahedron rotational MSD using the '
                                   'Fixman, Random Finite Difference, and '
                                   'Euler-Maruyama schemes at multiple '
                                   'timesteps.')
  parser.add_argument('-dt', dest='dt', type=float,
                      help='Timestep to use for runs.')
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
  parser.add_argument('-free', dest='has_location', type=bool,
                      default=True,
                      help='Whether or not the tetrahedron is allowed '
                      'to move (is free).  If it is allowed to move, '
                      'then the MSD includes translational and rotational '
                      'displacement.')
  parser.add_argument('-scheme', dest='scheme', type=str,
                      default='RFD',
                      help='Scheme to use for timestepping. Must be '
                      'RFD, FIXMAN, or EM (Euler Maruyama)')
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

  if args.scheme not in ['RFD', 'FIXMAN', 'EM']:
    raise Exception('Scheme must be one of RFD, FIXMAN, or EM')

  # Set initial conditions.
  initial_orientation = [Quaternion([1., 0., 0., 0.])]
  r_vectors = tdn.get_r_vectors(initial_orientation[0])
  initial_location = [[0., 0., 3.5]]

  dt = args.dt
  end_time = args.end_time
  n_runs = args.n_steps

  # Setup logging.
  log_filename = ('./logs/rotational-msd-initial-%s-location-%s-'
                  'scheme-%s-dt-%s-N-%d-%s.log' % (
      args.initial, args.has_location, args.scheme, dt, n_runs, args.data_name))
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
              'DEBYE_LENGTH': tf.DEBYE_LENGTH,
              'KT': tf.KT, 'end_time': end_time, 'N': n_runs}
  else:
    params = {'M1': tf.M1, 'M2': tf.M2, 'M3': tf.M3,
              'A': tf.A, 'KT': tf.KT, 'end_time': end_time,
              'N': n_runs}


  msd_statistics = MSDStatistics(params)
  # Measure time, and estimate how long runs will take.
  # One time unit is n_runs timesteps.
  if args.initial:
    run_data = calculate_msd_from_fixed_initial_condition(
      initial_orientation,
      args.scheme,
      dt,
      end_time,
      n_runs,
      has_location=args.has_location,
      location=initial_location)
  else:
    run_data = calc_rotational_msd_from_equilibrium(initial_orientation,
                                                    args.scheme,
                                                    dt,
                                                    end_time,
                                                    n_runs,
                                                    has_location=
                                                    args.has_location,
                                                    location=
                                                    initial_location,
                                                    n_runs=10)
  msd_statistics.add_run(args.scheme, dt, run_data)
  progress_logger.info('finished timestepping dt= %f for scheme %s' % (
      dt, args.scheme))
  progress_logger.info('Runs complete.')

  # Optional name for data provided
  data_name = args.data_name
  if len(data_name) > 3:
    data_name = ( 
      './data/rot-msd-initial-%s-location-%s-scheme-%s-dt-%s-N-%d-%s.pkl' % (
        args.initial, args.has_location, args.scheme, dt, n_runs, data_name))
  else:
    data_name = (
      './data/rot-msd-initial-%s-location-%s-scheme-%s-dt-%s-N-%d.pkl' % (
        args.initial, args.has_location, args.scheme, dt, n_runs))

  with open(data_name, 'wb') as f:
    cPickle.dump(msd_statistics, f)

  if args.profile:
    pr.disable()
    s = StringIO.StringIO()
    sortby = 'cumulative'
    ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
    ps.print_stats()
    print s.getvalue()
