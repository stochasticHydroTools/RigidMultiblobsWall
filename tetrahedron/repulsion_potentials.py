''' Script to plot the equilibrium PDF of the free tetrahedron
for various repulsion potentials.'''

import sys
sys.path.append('..')
import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot
import math

import tetrahedron_free as tf
from quaternion_integrator.quaternion import Quaternion

def bin_center_of_mass(location, orientation, bin_width, 
                       height_histogram):
  '''Bin heights of the free particle based on a location and an orientaiton.'''
  center_of_mass = tf.get_free_center_of_mass(location, orientation)
  # Bin each particle height.
  idx = (int(math.floor((center_of_mass[2])/bin_width)))
  if idx < len(height_histogram):
    height_histogram[idx] += 1
  else:
    print 'index is: ', idx
    print 'Index exceeds histogram length.'


if __name__ == '__main__':
  repulsion_strengths = [1.5, 1.5, 1.5, 3.0, 3.0]
  repulsion_cutoffs = [0.25, 0.5, 1.0, 0.25, 0.5]
  n_samples = 1000000

  for k in range(len(repulsion_strengths)):
    tf.REPULSION_STRENGTH = repulsion_strengths[k]
    tf.REPULSION_CUTOFF = repulsion_cutoffs[k]
    initial_location = [0., 0., tf.H]
    initial_orientation = Quaternion([1., 0., 0., 0.])
    sample = [initial_location, initial_orientation]
    bin_width = 1/10.
    bins = bin_width*np.arange(int(24./bin_width)) + bin_width/2.
    height_histogram = np.zeros(int(24./bin_width))
    for k in range(n_samples):
      sample = tf.generate_free_equilibrium_sample_mcmc(sample)
      bin_center_of_mass(sample[0], sample[1], bin_width, height_histogram)
      
    acceptance_rate = (float(tf.generate_free_equilibrium_sample_mcmc.accepts)/
                       float(tf.generate_free_equilibrium_sample_mcmc.samples))
    print 'acceptance rate for MCMC: %f' % acceptance_rate
    print 'low rejections: %d' % tf.gibbs_boltzmann_distribution.low_rejections
    height_histogram = height_histogram/n_samples/bin_width
    pyplot.plot(bins, height_histogram, label='Strength=%s, Cutoff=%s' % 
                (tf.REPULSION_STRENGTH, tf.REPULSION_CUTOFF))
    tf.generate_free_equilibrium_sample_mcmc.accepts = 0
    tf.generate_free_equilibrium_sample_mcmc.samples = 0

  pyplot.legend(loc='best', prop={'size': 9})
  pyplot.savefig('./figures/PotentialPDFs-N-%s.pdf' % n_samples)



    
    
  
    
    

  
