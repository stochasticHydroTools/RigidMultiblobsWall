'''
Estimate the total time dependent MSD (with std dev) for a sphere
near a single wall, and save to a pkl file in the data subfolder.  This
file can then be used to plot any component of time dependent mobility
with the plot_sphere_rotational_msd.py script.

We care most about the x-x diffusion and how it relates to
the average parallel mobility, which will be reported and 
plotted automatically, along with the equilibrium distribution
of sphere height.
'''

import argparse
import pickle
import logging
import math
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot
import numpy as np
import os
import sys
sys.path.append('..')
import time

from fluids import mobility as mb
from quaternion_integrator.quaternion import Quaternion
from quaternion_integrator.quaternion_integrator import QuaternionIntegrator
from . import sphere as sph
from general_application_utils import log_time_progress
from general_application_utils import static_var
from general_application_utils import MSDStatistics
from general_application_utils import StreamToLogger







def gibbs_boltzmann_distribution(location):
  '''
  Evaluate the equilibrium distribution at a given location for
  a single sphere above a wall.  Location is given as a list with
  [x, y, z] components of sphere position.
  '''
  # Calculate potential.
  if location[2] > sph.A:
    U = sph.M*location[2]
    U += (sph.REPULSION_STRENGTH*np.exp(-1.*(location[2] - sph.A)/sph.DEBYE_LENGTH)/
          (location[2] - sph.A))
  else:
    return 0.0
  return np.exp(-1.*U/sph.KT)  


def calc_total_sphere_msd(initial_location, initial_rot_matrix, 
                          location, rot_matrix):
  ''' Calulate 6x6 MSD for a sphere.'''
  dx = np.array(location) - np.array(initial_location)
  # Get rotational displacement, u_hat.
  u_hat = np.zeros(3)
  for i in range(3):
    e = np.zeros(3)
    e[i] = 1.
    u_hat += 0.5*np.cross(np.inner(initial_rot_matrix, e),
                          np.inner(rot_matrix, e))

  displacement = np.concatenate([dx, u_hat])
  return np.outer(displacement, displacement)

def calc_sphere_msd_from_equilibrium(initial_orientation,
                                     scheme,
                                     dt,
                                     end_time,
                                     n_steps,
                                     location=None,
                                     n_runs=10):
  ''' 
  Do a few long run, and along the way gather statistics
  about the average rotational Mean Square Displacement 
  by calculating it from time lagged data.  This is sphere specific.
  args:
    initial_orientation: list of length 1 quaternion where 
                 the run starts.  This shouldn't effect results.
    scheme: FIXMAN, RFD, or EM, scheme for the integrator to use.
    dt:  float, timestep used by the integrator.
    end_time: float, how much time to track the evolution of the MSD.
    n_steps:  How many total steps to take.
    location: initial location of sphere.
    n_runs:  How many separate runs to do in order to get std deviation.  
             10 by default.
  '''
  progress_logger = logging.getLogger('Progress Logger')
  burn_in = int(end_time*4./dt)
  rot_msd_list = []
  print_increment = n_steps/20
  dim = 6
  progress_logger.info('Starting runs...')
  start_time = time.time()
  for run in range(n_runs):
    integrator = QuaternionIntegrator(sph.sphere_mobility,
                                      initial_orientation, 
                                      sph.null_torque_calculator,
                                      has_location=True,
                                      initial_location=location,
                                      force_calculator=
                                      sph.sphere_force_calculator)
    integrator.kT = sph.KT
    integrator.check_function = sph.sphere_check_function
    # choose number of steps to take before saving data.
    # Want 100 points on our plot.
    data_interval = int((end_time/dt)/200.)
    trajectory_length = 200
    if data_interval == 0:
      data_interval = 1

    if trajectory_length*data_interval > n_steps:
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

      if (step > burn_in) and (step % data_interval == 0):
        lagged_trajectory.append(integrator.orientation[0].rotation_matrix())
        lagged_location_trajectory.append(integrator.location[0])

      if len(lagged_trajectory) > trajectory_length:
        lagged_trajectory = lagged_trajectory[1:]
        lagged_location_trajectory = lagged_location_trajectory[1:]
        for k in range(trajectory_length):
          current_rot_msd = (calc_total_sphere_msd(
            lagged_location_trajectory[0],
            lagged_trajectory[0],
            lagged_location_trajectory[k],
            lagged_trajectory[k]))
          average_rotational_msd[k] += current_rot_msd

      if (step % print_increment == 0 ) and (step > 0): 
        progress_logger.info('At step: %d in run %d of %d' % (step, run + 1, n_runs))
        elapsed_time = time.time() - start_time
        elapsed_units = (n_steps + burn_in)*run + step
        total_units = (n_steps + burn_in)*n_runs
        log_time_progress(elapsed_time, elapsed_units, total_units)

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



def plot_x_and_y_msd(msd_statistics, mob_and_friction, n_steps):
  '''  
  Plot Fixman and RFD x and y MSD. Also calculate the slope of the
  MSD at later times to compare to equilibrium mobility.
  '''
  scheme_colors = ['b','g','r']
  ind_styles = ['', ':']
  scheme_num = 0
  num_err_bars = 12
  average_msd_slope = 0.  # Calculate average slope of MSD.
  num_series = 0
  for scheme in list(msd_statistics.data.keys()):
    dt = min(msd_statistics.data[scheme].keys())
    for ind in [[0, 0], [1, 1]]:
      # Extract the entry specified by ind to plot.
      num_steps = len(msd_statistics.data[scheme][dt][0])
      # Don't put error bars at every point
      err_idx = [int(num_steps*k/num_err_bars) for k in range(num_err_bars)]
      msd_entries = np.array([msd_statistics.data[scheme][dt][1][_][ind[0]][ind[1]]
                     for _ in range(num_steps)])
      msd_entries_std = np.array([msd_statistics.data[scheme][dt][2][_][ind[0]][ind[1]]
                                  for _ in range(num_steps)])
      for k in range(5):
        average_msd_slope += (msd_entries[-1 - k] - msd_entries[-2 - k])/dt

      num_series += 1

      pyplot.plot(msd_statistics.data[scheme][dt][0],
                  msd_entries,
                  scheme_colors[scheme_num] + ind_styles[ind[0]],
                  label = '%s, ind=%s' % (scheme, ind))
      pyplot.errorbar(np.array(msd_statistics.data[scheme][dt][0])[err_idx],
                      msd_entries[err_idx],
                      yerr = 2.*msd_entries_std[err_idx],
                      fmt = scheme_colors[scheme_num] + '.')
    scheme_num += 1

  # Annotate plot and add theory.
  pyplot.plot(msd_statistics.data[scheme][dt][0], 
              2.*sph.KT*mob_and_friction[0]*np.array(msd_statistics.data[scheme][dt][0]),
              'k-',
              label='Slope=2 kT Mu Parallel')
  pyplot.plot(msd_statistics.data[scheme][dt][0], 
              2.*sph.KT*np.array(msd_statistics.data[scheme][dt][0]/mob_and_friction[1]),
              'r--',
              label='Slope=2 kT/Friction')
  pyplot.title('MSD(t) for spere in X and Y directions')
  pyplot.ylabel('MSD')
  pyplot.xlabel('time')
  pyplot.legend(loc='best', prop={'size': 9})
  pyplot.savefig('./figures/SphereTranslationalMSDComponent-N-%d.pdf' % n_steps)
  # Return average slope
  average_msd_slope /= num_series*5
  return average_msd_slope


def calculate_mu_friction_and_height_distribution(bin_width, height_histogram):
  ''' 
  Calculate average mu parallel and fricton using rectangle rule. 
  Populate height histogram with equilibrium distribution.
  TODO: Make this use trapezoidal rule.
  '''
  for k in range(len(height_histogram)):
    h = sph.A + bin_width*(k + 0.5)
    height_histogram[k] = gibbs_boltzmann_distribution([0., 0., h])
  
  # Normalize to get ~PDF.
  height_histogram /= sum(height_histogram)*bin_width
  # Calculate Mu and gamma.
  average_mu = 0.
  average_gamma = 0.
  # Just choose an arbitrary orientation, since it won't affect the
  # distribution.
  initial_orientation = [Quaternion([1., 0., 0., 0.])]
  for k in range(len(height_histogram)):
    h = sph.A + bin_width*(k + 0.5)    
    mobility = sph.sphere_mobility([np.array([0., 0., h])], initial_orientation)
    average_mu += (mobility[0, 0] + mobility[1, 1])*height_histogram[k]*bin_width
    average_gamma += height_histogram[k]*bin_width/mobility[0, 0]

  return [average_mu, average_gamma]


def bin_sphere_height(sample, height_histogram, bin_width):
  ''' 
  Bin the height (last component, idx = 2) of a sample, and
  add the count to height_histogram.
  '''
  idx = int(math.floor((sample[2])/bin_width)) 
  if idx < len(height_histogram):
    height_histogram[idx] += 1
  else:
    # Extend histogram to allow for this index.
    print('Index %d exceeds histogram length' % idx)


def plot_height_histograms(buckets, height_histograms, labels):
  ''' Plot buckets v. heights of eq and run pdf and save the figure.'''
  pyplot.figure()
  start_ind = 0.4/bin_width
  for k in range(len(height_histograms)):
    pyplot.plot(buckets[start_ind:], height_histograms[k][start_ind:],
                label=labels[k])
  pyplot.plot(sph.A*np.ones(2), [1e-5, 0.45], label="Touching Wall")
  pyplot.gca().set_yscale('log')
  pyplot.title('Height Distribution for Sphere')
  pyplot.legend(loc='best', prop={'size': 9})
  pyplot.xlabel('Height')
  pyplot.ylabel('PDF')
  # Make directory for figures if it doesn't exist.
  if not os.path.isdir(os.path.join(os.getcwd(), 'figures')):
    os.mkdir(os.path.join(os.getcwd(), 'figures'))
  pyplot.savefig('./figures/SphereHeights.pdf')
  

if __name__ == '__main__':
  # Get command line arguments.
  parser = argparse.ArgumentParser(
    description='Run Simulation of Sphere '
    'using the RFD scheme, and bin the resulting '
    'height distribution + calculate the MSD.  The MSD '
    'data is saved in the /data folder, and also plotted. '
    'The sphere is repulsed from the wall with the Yukawa '
    'potential.')
  parser.add_argument('-dt', dest='dt', type=float,
                      help='Timestep to use for runs.')
  parser.add_argument('-N', dest='n_steps', type=int,
                      help='Number of steps to take for runs.')
  parser.add_argument('-end', dest='end_time', type=float, default = 128.0,
                      help='How far to calculate the time dependent MSD.')
  parser.add_argument('--data-name', dest='data_name', type=str,
                      default='',
                      help='Optional name added to the end of the '
                      'data file.  Useful for multiple runs '
                      '(--data_name=run-1).')
  args=parser.parse_args()
  initial_orientation = [Quaternion([1., 0., 0., 0.])]
  initial_location = [np.array([0., 0., sph.H])]
  scheme = 'RFD'
  dt = args.dt
  end_time = args.end_time
  n_steps = args.n_steps
  bin_width = 1./10.
  buckets = np.arange(0, int(20./bin_width))*bin_width + bin_width/2.

  log_filename = './logs/sphere-rotation-dt-%f-N-%d-%s.log' % (
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
  params = {'M': sph.M, 'A': sph.A,
            'REPULSION_STRENGTH': sph.REPULSION_STRENGTH, 
            'DEBYE_LENGTH': sph.DEBYE_LENGTH,
            'KT': sph.KT}

  msd_statistics = MSDStatistics(params)

  run_data = calc_sphere_msd_from_equilibrium(
    initial_orientation,
    scheme,
    dt, 
    end_time,
    n_steps,
    location=initial_location)

  progress_logger.info('Completed equilibrium runs.')
  msd_statistics.add_run(scheme, dt, run_data)

  data_name = './data/sphere-msd-dt-%s-N-%d-%s.pkl' % (
    dt, n_steps, args.data_name)

  with open(data_name, 'wb') as f:
    pickle.dump(msd_statistics, f)

  height_histograms = []
  labels = []
  height_histograms.append(np.zeros(len(buckets)))
  labels.append('strength=%s, b=%s' % (sph.REPULSION_STRENGTH, sph.DEBYE_LENGTH))
  average_mob_and_friction = calculate_mu_friction_and_height_distribution(
    bin_width, height_histograms[-1])
  avg_slope = plot_x_and_y_msd(msd_statistics, 
                               [average_mob_and_friction[0], average_mob_and_friction[1]],
                               n_steps)

  plot_height_histograms(buckets, height_histograms, labels)
  print("Mobility is ", average_mob_and_friction[0])
  print("Average friction is ", average_mob_and_friction[1])
  print("1/Friction is %f" % (1./average_mob_and_friction[1]))
  print("Slope/2kT is ", avg_slope/2./sph.KT)
