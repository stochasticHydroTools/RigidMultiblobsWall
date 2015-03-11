'''
Script to calculate, from free rotational MSD data, the translational
(in x and y) long time diffusion coefficient. This will be compared against
<mu_parallel>, the parallel entry of mobility averaged over the equilibrium 
distribution.
'''
import os

import numpy as np
import cPickle
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot
import sys
sys.path.append('..')

from quaternion_integrator.quaternion import Quaternion
import tetrahedron_free as tf
from utils import MSDStatistics
import sphere.sphere_rotational_msd as sph

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
              tf.KT*2.*mob_and_friction[0]*
              np.array(msd_statistics.data[scheme][dt][0]),
              'k--',
              label='Slope=2 kT Mu Parallel')
  pyplot.plot(msd_statistics.data[scheme][dt][0], 
              tf.KT*2.*np.array(msd_statistics.data[scheme][dt][0]/
                                mob_and_friction[1]),
              'r--',
              label='Slope=2 kT/Friction')
  pyplot.title('MSD(t) in X and Y directions')
  pyplot.ylabel('MSD')
  pyplot.xlabel('time')
  pyplot.legend(loc='best', prop={'size': 9})
  pyplot.savefig('./figures/TranslationalMSDComponent.pdf')
  # Return average slope
  average_msd_slope /= num_series*5
  return average_msd_slope


def calculate_average_mu_parallel_and_perpendicular(n_samples):
  ''' 
  Generate random samples from equilibrium to
  calculate the average parallel mobility and friction. 
  Do this with masses equal for comparison to MSD data.
  '''
#  initial_location = [np.array([0., 0., tf.H])]
#  initial_orientation = [Quaternion([1., 0., 0., 0.])]
#  sample = [initial_location[0], initial_orientation[0]]
  average_mu_parallel = 0.0
  average_mu_perp = 0.0
  average_gamma_parallel = 0.0
  for k in range(n_samples):
    sample = tf.generate_free_equilibrium_sample()
    mobility_sample = tf.free_tetrahedron_mobility([sample[0]], [sample[1]])
    average_mu_parallel += mobility_sample[0, 0] + mobility_sample[1, 1]
    average_mu_perp += mobility_sample[2, 2]
    average_gamma_parallel += (1.0/mobility_sample[0, 0] + 
                               1.0/mobility_sample[1, 1])
    
  average_mu_parallel /= (2*n_samples)
  average_mu_perp /= n_samples
  average_gamma_parallel /= (2*n_samples)

  return [average_mu_parallel, average_gamma_parallel, average_mu_perp]

  
if __name__ == "__main__":
  # Open data file.
  data_name = os.path.join('data', sys.argv[1])
  with open(data_name, 'rb') as f:
    msd_statistics = cPickle.load(f)  
  
  n_runs = 4
  mobilities = []
  frictions = []
  for k in range(n_runs):
    average_mob_and_friction = calculate_average_mu_parallel_and_perpendicular(4000)
    mobilities.append(average_mob_and_friction[0])
    frictions.append(average_mob_and_friction[1])

  average_mobility = np.mean(mobilities)
  mobility_std = np.std(mobilities)/np.sqrt(n_runs)
  average_friction = np.mean(frictions)
  friction_std = np.std(frictions)/np.sqrt(n_runs)

  avg_slope = plot_x_and_y_msd(msd_statistics, 
                               [average_mobility, average_friction])
  
  print "Mobility is ", average_mobility, " +/- ", mobility_std
  print "1/Friction is %f to %f" %  (1./(average_friction + 2.*friction_std),
         1./(average_friction - 2.*friction_std))
  print "Slope/2kT is ", avg_slope/2./tf.KT
  
  
