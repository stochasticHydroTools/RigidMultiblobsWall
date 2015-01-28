''' 
Estimate the total MSD (we are mainly concerned with rotation).

u_hat(dt) = \sum_i u_i(0) cross u_i(dt)
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
import math

from quaternion_integrator.quaternion import Quaternion
from quaternion_integrator.quaternion_integrator import QuaternionIntegrator
from tetrahedron_rotational_msd import MSDStatistics
from tetrahedron_free import static_var
from fluids import mobility as mb

#Parameters
ETA = 1.0
A = 0.5
M  = 0.05
H = 3.5
# Parameters for Yukawa potential
REPULSION_STRENGTH = 3.0
REPULSION_CUTOFF = 0.25  # This is the Debye length, TODO: rename.
KT = 0.5


def null_torque_calculator(location, orientation):
  return [0., 0., 0.]

def sphere_force_calculator(location, orientation):
  gravity = -1*M
  h = location[0][2]
  repulsion = (REPULSION_STRENGTH*((h - A)/REPULSION_CUTOFF + 1)*
               np.exp(-1.*(h - A)/REPULSION_CUTOFF)/((h - A)**2))
  return [0., 0., gravity + repulsion]

def sphere_mobility(location, orientation):
  location = [location[0]]
  fluid_mobility = mb.boosted_single_wall_fluid_mobility(location, ETA, A)
  mobility = np.concatenate([fluid_mobility, np.zeros([3, 3])])
  mobility = np.concatenate([mobility, 
                             np.concatenate([np.zeros([3, 3]), np.identity(3)])],
                            axis=1)
  return mobility

@static_var('samples', 0)  
@static_var('accepts', 0)  
def generate_sphere_equilibrium_sample_mcmc(current_sample):
  '''
  Generate an equilibrium sample of location and orientation, according
  to the distribution exp(-\beta U(heights)) by using MCMC.
  '''
  generate_sphere_equilibrium_sample_mcmc.samples += 1
  location = current_sample
  # Tune this dt parameter to try to achieve acceptance rate of ~50%.
  dt = 0.1
  # Take a step using Metropolis.
  velocity = np.random.normal(0., 1., 3)
  new_location = location + velocity*dt
  accept_probability = (gibbs_boltzmann_distribution(new_location)/
                        gibbs_boltzmann_distribution(location))

  if np.random.uniform() < accept_probability:
    generate_sphere_equilibrium_sample_mcmc.accepts += 1
    return new_location
  else:
    return location
                          

def gibbs_boltzmann_distribution(location):
  '''
  Evaluate the equilibrium distribution at a given location for
  a single sphere above a wall.  Location is given as a list with
  [x, y, z] components of sphere position.
  '''
  # Calculate potential.
  U = M*location[2]
  U += (REPULSION_STRENGTH*np.exp(-1.*(location[2] - A)/REPULSION_CUTOFF)/
        (location[2] - A))
  return np.exp(-1.*U/KT)  

def calc_rotational_msd_from_equilibrium(initial_orientation,
                                         scheme,
                                         dt,
                                         end_time,
                                         n_steps,
                                         has_location=False,
                                         location=None,
                                         n_runs=4):

  ''' 
  Do a few long runs, and along the way gather statistics
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
  Copied from tetrahedron_rotational_msd and modified slightly.
  '''
  dim = 3
  rot_msd_list = []
  for k in range(n_runs):
    integrator = QuaternionIntegrator(sphere_mobility,
                                      initial_orientation, 
                                      null_torque_calculator,
                                      has_location=has_location,
                                      initial_location=location,
                                      force_calculator=
                                      sphere_force_calculator)
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
            current_rot_msd = (calc_translation_msd(
                lagged_location_trajectory[0],
                lagged_location_trajectory[k]))
            average_rotational_msd[k] += current_rot_msd
          else:
            current_rot_msd = (calc_rotational_msd(
                lagged_trajectory[0],
                lagged_trajectory[k]))
            average_rotational_msd[k] += current_rot_msd

    average_rotational_msd = average_rotational_msd/(n_steps - trajectory_length)
    rot_msd_list.append(average_rotational_msd)


  # Average results to get time, mean, and std of rotational MSD.
  # For now, std = 0.  Will figure out a good way to calculate this later.
  results = [[], [], []]
  results[0] = np.arange(0, trajectory_length)*dt
  results[1] = np.mean(rot_msd_list, axis=0)
  results[2] = np.std(rot_msd_list, axis=0)/np.sqrt(n_runs)

  progress_logger = logging.getLogger('progress_logger')  
  progress_logger.info('Rejection Rate: %s' % 
                       (float(integrator.rejections)/
                        float(n_steps + integrator.rejections)))
  return results

def calc_translation_msd(initial_location, location):
  ''' Calculate 3x3 MSD including just location.'''
  dx = np.array(location) - np.array(initial_location)
  return np.outer(dx, dx)


def plot_x_and_y_msd(msd_statistics, mob_and_friction):
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
  pyplot.savefig('./figures/SphereTranslationalMSDComponent.pdf')
  # Return average slope
  average_msd_slope /= num_series*5
  return average_msd_slope


def calculate_average_mu_parallel_and_bin_heights(n_samples, height_histogram,
                                                  bin_width):
  ''' 
  Generate random samples from equilibrium to
  calculate the average parallel mobility and friction. 
  Do this with masses equal for comparison to MSD data.
  '''
  initial_location = [np.array([0., 0., H])]
  initial_orientation = [Quaternion([1., 0., 0., 0.])]
  sample = initial_location[0]
  average_mu_parallel = 0.0
  average_gamma_parallel = 0.0
  average_sphere_height = 0.0
  for k in range(n_samples):
    sample = generate_sphere_equilibrium_sample_mcmc(sample)
    mobility_sample = sphere_mobility([sample], initial_orientation)
    average_mu_parallel += mobility_sample[0, 0]
    average_gamma_parallel += (1.0/mobility_sample[0, 0])
    bin_sphere_height(sample, height_histogram, bin_width)
    
  average_mu_parallel /= n_samples
  average_gamma_parallel /= n_samples

  return [average_mu_parallel, average_gamma_parallel]


def bin_sphere_height(sample, height_histogram, bin_width):
  ''' 
  Bin the height (last component, idx = 2) of a sample, and
  add the count to height_histogram.
  '''
  idx = int(math.floor((sample[2])/bin_width)) 
  if idx < len(height_histogram):
    height_histogram[idx] += 1
  else:
    print 'index is: ', idx
    print 'Index exceeds histogram length.'

def plot_height_histogram(buckets, height_histogram):
  ''' Plot buckets v. height pdf and save the figure.'''
  pyplot.figure()
  pyplot.plot(buckets, height_histogram)
  pyplot.title('Equilibrium Height Distribution')
  pyplot.xlabel('Height')
  pyplot.ylabel('PDF')
  # Make directory for data if it doesn't exist.
  if not os.path.isdir(os.path.join(os.getcwd(), 'figures')):
    os.mkdir(os.path.join(os.getcwd(), 'figures'))
  pyplot.savefig('./figures/SphereHeights.pdf')
  

if __name__ == '__main__':

  initial_orientation = [Quaternion([1., 0., 0., 0.])]
  initial_location = [np.array([0., 0., H])]

  scheme = 'FIXMAN'
  dt = 0.5
  end_time = 180.0
  n_steps = 200000
  bin_width = 1./10.
  buckets = np.arange(0, int(20./bin_width))*bin_width + bin_width/2.
  height_histogram = np.zeros(len(buckets))

  params = {'M': M, 'A': A,
            'REPULSION_STRENGTH': REPULSION_STRENGTH, 
            'REPULSION_CUTOFF': REPULSION_CUTOFF}

  msd_statistics = MSDStatistics(['FIXMAN'], [dt], params)

  run_data = calc_rotational_msd_from_equilibrium(initial_orientation,
                                                  scheme,
                                                  dt, 
                                                  end_time,
                                                  n_steps,
                                                  has_location=True,
                                                  location=initial_location)
  msd_statistics.add_run(scheme, dt, run_data)
  # Make directory for data if it doesn't exist.
  if not os.path.isdir(os.path.join(os.getcwd(), 'data')):
    os.mkdir(os.path.join(os.getcwd(), 'data'))


  data_name = './data/sphere-msd-dt-%s-N-%d.pkl' % (
    dt, n_steps)

  with open(data_name, 'wb') as f:
    cPickle.dump(msd_statistics, f)

  n_runs = 16
  mobilities = []
  frictions = []
  for k in range(n_runs):
    average_mob_and_friction = calculate_average_mu_parallel_and_bin_heights(
      15000, height_histogram, bin_width)
    mobilities.append(average_mob_and_friction[0])
    frictions.append(average_mob_and_friction[1])

  average_mobility = np.mean(mobilities)
  mobility_std = np.std(mobilities)/np.sqrt(n_runs)
  average_friction = np.mean(frictions)
  friction_std = np.std(frictions)/np.sqrt(n_runs)

  avg_slope = plot_x_and_y_msd(msd_statistics, 
                               [average_mobility, average_friction])

  height_histogram /= sum(height_histogram)*bin_width
  plot_height_histogram(buckets, height_histogram)
  print "Mobility is ", average_mobility, " +/- ", mobility_std
  print "1/Friction is %f to %f" %  (1./(average_friction + 2.*friction_std),
         1./(average_friction - 2.*friction_std))
  print "Slope/2kT is ", avg_slope/2./KT
