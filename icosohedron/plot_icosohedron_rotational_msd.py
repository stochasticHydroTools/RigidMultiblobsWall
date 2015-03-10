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

  # HACK, combine 0 and 1 component into parallel.
  for scheme in msd_statistics.data:
    for dt in msd_statistics.data[scheme]:
      for k in range(len(msd_statistics.data[scheme][dt][1])):
        msd_statistics.data[scheme][dt][1][k][0][0] = (
          msd_statistics.data[scheme][dt][1][k][0][0] +
          msd_statistics.data[scheme][dt][1][k][1][1])
        msd_statistics.data[scheme][dt][2][k][0][0] = np.sqrt(
          msd_statistics.data[scheme][dt][2][k][0][0]**2 +
          msd_statistics.data[scheme][dt][2][k][1][1]**2)

  # Open Sphere file to compare to.
  sphere_data_name = os.path.join('..', 'sphere', 'data',
                                  'sphere-msd-dt-1.0-N-500000-more-points-1.pkl')
  with open(sphere_data_name, 'rb') as f:
    sphere_statistics = cPickle.load(f)
    print 'Sphere parameters:'
    sphere_statistics.print_params()

  for scheme in sphere_statistics.data:
    for dt in sphere_statistics.data[scheme]:
      for k in range(len(sphere_statistics.data[scheme][dt][1])):
        sphere_statistics.data[scheme][dt][1][k][0][0] = (
          sphere_statistics.data[scheme][dt][1][k][0][0] +
          sphere_statistics.data[scheme][dt][1][k][1][1])
        sphere_statistics.data[scheme][dt][2][k][0][0] = np.sqrt(
          sphere_statistics.data[scheme][dt][2][k][0][0]**2 +
          sphere_statistics.data[scheme][dt][2][k][1][1]**2)
    
  bin_width = 1./10.
  buckets = np.arange(0, int(20./bin_width))*bin_width + bin_width/2.
  height_histogram = np.zeros(len(buckets))
  average_mob_and_friction = calculate_mu_friction_and_height_distribution(
    bin_width, height_histogram)
  
  zz_msd = calculate_zz_msd_at_equilibrium(20000)
  
  figure_index = [1, 2, 1, 3, 4, 5]
  label_list = [' Icosohedron Parallel MSD', ' Icosohedron yy MSD', ' Icosohedron Perpendicular MSD', 
                ' Icosohedron Rotational MSD', ' Icosohedron Rotational MSD', ' Icosohedron Rotational MSD']
  sphere_label_list = [' Sphere Parallel MSD', ' Sphere yy MSD', ' Sphere perpendicular MSD', 
                       ' Sphere Rotational MSD', ' Sphere Rotational MSD', ' Sphere Rotational MSD']
  style_list = ['.', 's', '^', '.', '.', '.']
  sphere_style_list = ['d', 'o', 'o', 'o', 'o', 'o']
  translation_plot_limit = 1100.
  for l in range(6):
    ind = [l, l]
    if l in [0, 2]:
      data_name = 'TranslationalMSDComponent.txt'
      num_err_bars = 30
    elif l == 3:
      data_name = 'RotationalMSDComponent.txt'
      num_err_bars = 60
    else:
      data_name = None
    plot_time_dependent_msd(msd_statistics, ind, figure_index[l], symbol=style_list[l], 
                            label=label_list[l], num_err_bars=num_err_bars)
    plot_time_dependent_msd(sphere_statistics, ind, figure_index[l], color='b', 
                            label=sphere_label_list[l], symbol=sphere_style_list[l],
                            data_name = "SphereMSDComponent-%s.txt" % l,
                            num_err_bars=num_err_bars)
    if l == 0:
      pyplot.plot([0.0, translation_plot_limit], 
                  [0.0, translation_plot_limit*2.*2.*sph.KT*0.0941541889044], 'k:', 
                  label='Slope = Sphere Mobility')
#      pyplot.plot([0., translation_plot_limit], [0., translation_plot_limit*average_mob_and_friction[0]*2.*ic.KT], 'k--',
#                  label='Slope = Icosohedron Mobility')
    if l == 2:
      pyplot.plot([0., translation_plot_limit], [zz_msd, zz_msd], 'k--',  
                  label='Asymptotic ZZ MSD')
      pyplot.xlim([0., translation_plot_limit,])
    if l in [3, 4, 5]:
      pyplot.xlim([0., 150.])
    pyplot.title('MSD(t) for icosohedron')
    pyplot.legend(loc='best', prop={'size': 9})
    pyplot.savefig('./figures/IcosohedronTimeDependentMSD-Component-%s-%s.pdf' % 
                   (ind[0], ind[1]))
  print "Sphere mobility is ", 0.0941541889044
  print "Asymptotic zz MSD for Icosohedron is", zz_msd
  print "Icosohedron mobility is ", average_mob_and_friction[0]
