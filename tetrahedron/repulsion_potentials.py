''' Script to plot the equilibrium PDF of the free tetrahedron
for various repulsion potentials.'''


import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot

import tetrahedron_free as tf

def bin_center_of_mass(location, orientation, bin_width, 
                       height_histogram):
  '''Bin heights of the free particle based on a location and an orientaiton.'''
  center_of_mass = tf.get_free_center_of_mass(location, orientation)
  # Bin each particle height.
  idx = (int(math.floor((center_of_mass[2])/bin_width)))
  if idx < len(height_histogram[k]):
    height_histogram[k][idx] += 1
  else:
    print 'index is: ', idx
    print 'Index exceeds histogram length.'


if __name__ == '__main__':
  repulsion_strengths = []
  repulsion_cutoffs =[]
  n_samples = 100000

  for k in range(len(repulsion_strengths)):
    tf.REPULSION_STRENGTH = repulsion_strengths[k]
    tf.REPULSION_CUTOFF = repulsion_cutoffs[k]
    initial_location = 
    initial_orientation = 
    current_sample = [initial_location, initial_orientation]
    bin_width = 
    height_histogram = 
    for k in range(n_samples):
      sample = tf.generate_free_equilibrium_sample_mcmc(sample)
      bin_center_of_mass(sample[0], sample[1], bin_width, height_histogram)

    pyplot.plot(bins, height_histogram, label='Strength=%s, Cutoff=%s' % 
                (tf.REPULSION_STRENGTH, tf.REPULSION_CUTOFF))
    
    
  
    
    

  
