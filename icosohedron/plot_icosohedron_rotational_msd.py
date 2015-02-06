''' 
Plot rotational msd data from a pickle file. 
'''
import os
import sys
sys.path.append('..')
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot
import numpy as np
import cPickle

import icosohedron as ic
from quaternion_integrator.quaternion import Quaternion
import sphere.sphere as sph
from tetrahedron.plot_rotational_msd import plot_time_dependent_msd
from utils import MSDStatistics


def gibbs_boltzmann_distribution(location):
  ''' 
  Return gibbs boltzmann distribution (without normalization)
  for the icosohedron when center of mass at location. 
  '''
    # Calculate potential.
  if location[2] > ic.A:
    U = sum(ic.M)*location[2]
    U += (ic.REPULSION_STRENGTH*np.exp(-1.*(location[2] - ic.A)/ic.DEBYE_LENGTH)/
          (location[2] - ic.A))
  else:
    return 0.0
  return np.exp(-1.*U/ic.KT)  

def calculate_mu_friction_and_height_distribution(bin_width, height_histogram):
  ''' 
  Calculate average mu parallel and fricton using rectangle rule. 
  Populate height histogram with equilibrium distribution.
  TODO: Make this use trapezoidal rule.
  '''
  for k in range(len(height_histogram)):
    h = ic.A + bin_width*(k + 0.5)
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
    h = ic.A + bin_width*(k + 0.5)    
    mobility = ic.icosohedron_mobility([np.array([0., 0., h])], initial_orientation)
    average_mu += mobility[0, 0]*height_histogram[k]*bin_width
    average_gamma += height_histogram[k]*bin_width/mobility[0, 0]

  return [average_mu, average_gamma]


if __name__ == "__main__":
  # Make sure figures folder exists
  if not os.path.isdir(os.path.join(os.getcwd(), 'figures')):
    os.mkdir(os.path.join(os.getcwd(), 'figures'))
  # Open data file.
  data_name = os.path.join('data', sys.argv[1])
  with open(data_name, 'rb') as f:
    msd_statistics = cPickle.load(f)


  bin_width = 1./10.
  buckets = np.arange(0, int(20./bin_width))*bin_width + bin_width/2.
  height_histogram = np.zeros(len(buckets))
  average_mob_and_friction = calculate_mu_friction_and_height_distribution(
    bin_width, height_histogram)
  

  ind = [0, 0]
  plot_time_dependent_msd(msd_statistics, ind, 1)
  if ind == [0, 0] or ind == [1, 1]:
    pyplot.plot([0.0, 180.0], [0.0, 180.*2.*sph.KT*0.0941541889044], 'r--', 
              label='Sphere Mobility')
    pyplot.plot([0., 180.], [0., 180.*average_mob_and_friction[0]*2.*ic.KT], 'k--',
                label='Icosohedron Mobility')
  pyplot.title('MSD(t) for Sphere')
  pyplot.legend(loc='best', prop={'size': 9})
  pyplot.savefig('./figures/IcosohedronTimeDependentMSD-Component-%s.pdf' % 
                   (ind))

  print "Icosohedron mobility is ", average_mob_and_friction[0]
