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
import cPickle
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

from quaternion_integrator.quaternion import Quaternion
from quaternion_integrator.quaternion_integrator import QuaternionIntegrator
from tetrahedron.tetrahedron_rotational_msd import MSDStatistics
import tetrahedron.tetrahedron_rotational_msd as trm
from tetrahedron.tetrahedron_free import static_var
from utils import StreamToLogger
from fluids import mobility as mb


#Parameters
ETA = 1.0  # Viscosity.
A = 0.5    # Radius of sphere.
M  = 0.1   # Mass*g of sphere.
H = 3.5    # Initial Distance from Wall.
KT = 0.2   # Temperature.
# Parameters for Yukawa potential
REPULSION_STRENGTH = 2.0
DEBYE_LENGTH = 0.25  



def sphere_check_function(location, orientation):
  ''' Check that sphere is not overlapping the wall. '''
  if location[0][2] < A:
    return False
  else:
    return True
  

def null_torque_calculator(location, orientation):
  return [0., 0., 0.]


def sphere_force_calculator(location, orientation):
  gravity = -1*M
  h = location[0][2]
  repulsion = (REPULSION_STRENGTH*((h - A)/DEBYE_LENGTH + 1)*
               np.exp(-1.*(h - A)/DEBYE_LENGTH)/((h - A)**2))
  return [0., 0., gravity + repulsion]


def sphere_mobility(location, orientation):
  location = location[0]
  fluid_mobility = mb.single_wall_self_mobility_with_rotation(location, ETA, A)
  return fluid_mobility


def gibbs_boltzmann_distribution(location):
  '''
  Evaluate the equilibrium distribution at a given location for
  a single sphere above a wall.  Location is given as a list with
  [x, y, z] components of sphere position.
  '''
  # Calculate potential.
  if location[2] > A:
    U = M*location[2]
    U += (REPULSION_STRENGTH*np.exp(-1.*(location[2] - A)/DEBYE_LENGTH)/
          (location[2] - A))
  else:
    return 0.0
  return np.exp(-1.*U/KT)  


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
  for scheme in msd_statistics.data.keys():
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
              2.*KT*mob_and_friction[0]*np.array(msd_statistics.data[scheme][dt][0]),
              'k-',
              label='Slope=2 kT Mu Parallel')
  pyplot.plot(msd_statistics.data[scheme][dt][0], 
              2.*KT*np.array(msd_statistics.data[scheme][dt][0]/mob_and_friction[1]),
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
    h = A + bin_width*(k + 0.5)
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
    h = A + bin_width*(k + 0.5)    
    mobility = sphere_mobility([np.array([0., 0., h])], initial_orientation)
    average_mu += mobility[0, 0]*height_histogram[k]*bin_width
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
    print 'Index %d exceeds histogram length' % idx


def plot_height_histograms(buckets, height_histograms, labels):
  ''' Plot buckets v. heights of eq and run pdf and save the figure.'''
  pyplot.figure()
  start_ind = 0.4/bin_width
  for k in range(len(height_histograms)):
    pyplot.plot(buckets[start_ind:], height_histograms[k][start_ind:],
                label=labels[k])
  pyplot.plot(A*np.ones(2), [0., 0.45], label="Touching Wall")
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
  initial_location = [np.array([0., 0., H])]

  scheme = 'RFD'
  dt = args.dt
  end_time = args.end_time
  n_steps = args.n_steps
  bin_width = 1./10.
  buckets = np.arange(0, int(20./bin_width))*bin_width + bin_width/2.

  # Set up logging.
  # Make directory for logs if it doesn't exist.
  if not os.path.isdir(os.path.join(os.getcwd(), 'logs')):
    os.mkdir(os.path.join(os.getcwd(), 'logs'))
  if not os.path.isdir(os.path.join(os.getcwd(), 'figures')):
    os.mkdir(os.path.join(os.getcwd(), 'figures'))

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
  params = {'M': M, 'A': A,
            'REPULSION_STRENGTH': REPULSION_STRENGTH, 
            'DEBYE_LENGTH': DEBYE_LENGTH}

  msd_statistics = MSDStatistics(['FIXMAN'], [dt], params)

  run_data = trm.calc_rotational_msd_from_equilibrium(
    initial_orientation,
    scheme,
    dt, 
    end_time,
    n_steps,
    has_location=True,
    location=initial_location,
    check_fcn=sphere_check_function)

  progress_logger.info('Completed equilibrium runs.')
  msd_statistics.add_run(scheme, dt, run_data)
  # Make directory for data if it doesn't exist.
  if not os.path.isdir(os.path.join(os.getcwd(), 'data')):
    os.mkdir(os.path.join(os.getcwd(), 'data'))

  data_name = './data/sphere-msd-dt-%s-N-%d-%s.pkl' % (
    dt, n_steps, args.data_name)

  with open(data_name, 'wb') as f:
    cPickle.dump(msd_statistics, f)

  repulsion_strengths = [2.0]
  debye_lengths = [0.25]
  height_histograms = []
  labels = []
  for param_idx in range(len(repulsion_strengths)):
    REPULSION_STRENGTH = repulsion_strengths[param_idx]
    DEBYE_LENGTH = debye_lengths[param_idx]
    height_histograms.append(np.zeros(len(buckets)))
    labels.append('strength=%s, b=%s' % (REPULSION_STRENGTH, DEBYE_LENGTH))
    average_mob_and_friction = calculate_mu_friction_and_height_distribution(
      bin_width, height_histograms[-1])
  avg_slope = plot_x_and_y_msd(msd_statistics, 
                               [average_mob_and_friction[0], average_mob_and_friction[1]],
                               n_steps)

  # height_histogram_run /= sum(height_histogram_run)*bin_width
  # height_histogram.append(height_histogram_run)
  # labels.append('Run')
  plot_height_histograms(buckets, height_histograms, labels)
  print "Mobility is ", average_mob_and_friction[0]
  print "Average friction is ", average_mob_and_friction[1]
  print "1/Friction is %f" % (1./average_mob_and_friction[1])
  print "Slope/2kT is ", avg_slope/2./KT
