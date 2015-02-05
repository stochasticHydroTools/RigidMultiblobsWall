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
  '''Bin heights of the free particle based on a location and an orientation.'''
  center_of_mass = tf.get_free_center_of_mass(location, orientation)
  # Bin each particle height.
  idx = (int(math.floor((center_of_mass[2])/bin_width)))
  if idx < len(height_histogram):
    height_histogram[idx] += 1
  else:
    # Extend histogram to allow for this index.
    print "index out of range: ", idx
    # print 'Extending histogram for particle %s. ' % k
    # new_entries = np.zeros(idx - len(height_histogram) + 1)
    # height_histogram[k] = np.concatenate([height_histogram, 
    #                                       new_entries])
    # height_histogram[k][idx] += 1


if __name__ == '__main__':
  repulsion_strengths = [2.0]
  repulsion_cutoffs = [0.25]
  n_samples = 200000
  write_data = True

  for k in range(len(repulsion_strengths)):
    tf.REPULSION_STRENGTH = repulsion_strengths[k]
    tf.DEBYE_LENGTH = repulsion_cutoffs[k]
    initial_location = [0., 0., tf.H]
    initial_orientation = Quaternion([1., 0., 0., 0.])
    sample = [initial_location, initial_orientation]
    bin_width = 1/5.
    bins = bin_width*np.arange(int(18./bin_width)) + bin_width/2.
    height_histogram = np.zeros(int(18./bin_width))
    for k in range(n_samples):
      sample = tf.generate_free_equilibrium_sample_mcmc(sample)
      bin_center_of_mass(sample[0], sample[1], bin_width, height_histogram)
      
    acceptance_rate = (float(tf.generate_free_equilibrium_sample_mcmc.accepts)/
                       float(tf.generate_free_equilibrium_sample_mcmc.samples))
    print 'acceptance rate for MCMC: %f' % acceptance_rate
    print 'low rejections: %d' % tf.gibbs_boltzmann_distribution.low_rejections
    height_histogram = height_histogram/n_samples/bin_width
    if write_data:
      with open ('./data/free-tetrahedron-pdf.txt', 'w+') as f:
        f.write('repulsion strength: %f \n' % tf.REPULSION_STRENGTH)
        f.write('debye length: %f \n ' % tf.DEBYE_LENGTH)
        f.write('Buckets:\n')
        f.write('%s \n' % bins)
        f.write('PDF:\n')
        f.write('%s \n' % height_histogram)
        
        
                         
                         
    pyplot.plot(bins, height_histogram, label='Strength=%s, Cutoff=%s' % 
                (tf.REPULSION_STRENGTH, tf.DEBYE_LENGTH))
    tf.generate_free_equilibrium_sample_mcmc.accepts = 0
    tf.generate_free_equilibrium_sample_mcmc.samples = 0

  pyplot.legend(loc='best', prop={'size': 9})
  pyplot.title('Distribution of center of mass')
  pyplot.savefig('./figures/PotentialPDFs-N-%s.pdf' % n_samples)



    
    
  
    
    

  
