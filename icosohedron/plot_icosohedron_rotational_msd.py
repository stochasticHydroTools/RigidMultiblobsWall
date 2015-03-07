''' 
Plot rotational msd data from a pickle file. 
'''
import os
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot
import numpy as np
import os
import cPickle
import sys

import icosohedron as ic
from quaternion_integrator.quaternion import Quaternion
import sphere.sphere as sph
from utils import MSDStatistics
from utils import plot_time_dependent_msd

def calculate_zz_msd_at_equilibrium(n_steps):
  ''' Use MC to caluclate asymptotic (t -> inf) zz MSD at equilibrium'''
  zz_msd = 0.
  for k in range(n_steps):
    sample_1 = ic.generate_icosohedron_equilibrium_sample()
    sample_2 = ic.generate_icosohedron_equilibrium_sample()
    zz_msd += (sample_2[0][2] - sample_1[0][2])**2

  zz_msd /= n_steps
  return zz_msd


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
  # Open data file.
  data_name = os.path.join('data', sys.argv[1])
  with open(data_name, 'rb') as f:
    msd_statistics = cPickle.load(f)
    print 'Icosohedron parameters:'
    msd_statistics.print_params()


  # Open Sphere file to compare to.
  sphere_data_name = os.path.join('..', 'sphere', 'data',
                                  'sphere-msd-dt-1.0-N-500000-production.pkl')
  with open(sphere_data_name, 'rb') as f:
    sphere_statistics = cPickle.load(f)
    print 'Sphere parameters:'
    sphere_statistics.print_params()

    
  bin_width = 1./10.
  buckets = np.arange(0, int(20./bin_width))*bin_width + bin_width/2.
  height_histogram = np.zeros(len(buckets))
  average_mob_and_friction = calculate_mu_friction_and_height_distribution(
    bin_width, height_histogram)
  
  zz_msd = calculate_zz_msd_at_equilibrium(20000)
  
  figure_index = [1, 2, 1, 3, 4, 5]
  label_list = [' Icosohedron xx MSD', ' Icosohedron yy MSD', ' Icosohedron zz MSD', 
                ' Rotational MSD', ' Rotational MSD', ' Rotational MSD']
  sphere_label_list = [' Sphere xx MSD', ' Sphere yy MSD', ' Sphere zz MSD', 
                       ' Sphere Rotational MSD', ' Sphere Rotational MSD', ' Sphere Rotational MSD']
  style_list = ['.', 's', '^', '.', '.', '.']
  sphere_style_list = ['x', 'o', 'o', 'o', 'o', 'o']
  translation_plot_limit = 100.
  for l in range(6):
    ind = [l, l]
    plot_time_dependent_msd(msd_statistics, ind, figure_index[l], symbol=style_list[l], 
                            label=label_list[l])
    plot_time_dependent_msd(sphere_statistics, ind, figure_index[l], color='r', 
                            label=sphere_label_list[l], symbol=sphere_style_list[l],
                            data_name = "SphereMSDComponent-%s.txt" % l)
    if l == 0:
      pyplot.plot([0.0, translation_plot_limit], [0.0, translation_plot_limit*2.*sph.KT*0.0941541889044], 'r:', 
                  label='Slope = Sphere Mobility')
      pyplot.plot([0., translation_plot_limit], [0., translation_plot_limit*average_mob_and_friction[0]*2.*ic.KT], 'k--',
                  label='Slope = Icosohedron Mobility')
    if l == 2:
      pyplot.plot([0., translation_plot_limit], [zz_msd, zz_msd], 'b--',  
                  label='Asymptotic ZZ MSD')
      pyplot.xlim([0., translation_plot_limit,])
    if l == 3:
      pyplot.xlim([0., 150.])
    pyplot.title('MSD(t) for icosohedron')
    pyplot.legend(loc='best', prop={'size': 9})
    pyplot.savefig('./figures/IcosohedronTimeDependentMSD-Component-%s-%s.pdf' % 
                   (ind[0], ind[1]))
  print "Sphere mobility is ", 0.0941541889044
  print "Asymptotic zz MSD for Icosohedron is", zz_msd
  print "Icosohedron mobility is ", average_mob_and_friction[0]
