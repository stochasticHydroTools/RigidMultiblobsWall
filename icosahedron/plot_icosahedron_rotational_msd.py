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

import icosahedron as ic
import icosahedron_nonuniform as icn
from quaternion_integrator.quaternion import Quaternion
import sphere.sphere as sph
from utils import calc_total_msd_from_matrix_and_center
from utils import MSDStatistics
from utils import plot_time_dependent_msd


def calculate_zz_and_rot_msd_at_equilibrium(n_steps):
  ''' Use MC to calculate asymptotic (t -> inf) zz MSD at equilibrium'''
  zz_msd = 0.
  rot_msd = 0.
  rot_perp_msd = 0.
  for k in range(n_steps):
    sample_1 = icn.generate_nonuniform_icosahedron_equilibrium_sample()
    sample_2 = icn.generate_nonuniform_icosahedron_equilibrium_sample()
    rot_mat_1 = sample_1[1].rotation_matrix().T
    rot_mat_2 = sample_2[1].rotation_matrix().T
    total_msd = calc_total_msd_from_matrix_and_center(sample_1[0], rot_mat_1,
                                                      sample_2[0], rot_mat_2)
    zz_msd += total_msd[2, 2]
    rot_msd += total_msd[3, 3]
    rot_perp_msd += total_msd[5, 5]

  zz_msd /= n_steps
  rot_msd /= n_steps
  rot_perp_msd /= n_steps
  return [zz_msd, rot_msd, rot_perp_msd]


def gibbs_boltzmann_distribution(location):
  ''' 
  Return gibbs boltzmann distribution (without normalization)
  for the icosahedron when center of mass at location. 
  '''
    # Calculate potential.
  if location[2] > ic.EFFECTIVE_A:
    # HACK, this is ICN now.
    U = sum(icn.M)*location[2]
    U += (ic.REPULSION_STRENGTH*np.exp(-1.*(location[2] - ic.EFFECTIVE_A)/ic.DEBYE_LENGTH)/
          (location[2] - ic.EFFECTIVE_A))
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
    h = ic.EFFECTIVE_A + bin_width*(k + 0.5)
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
    h = ic.EFFECTIVE_A + bin_width*(k + 0.5)    
    mobility = ic.icosahedron_mobility([np.array([0., 0., h])], initial_orientation)
    average_mu += mobility[0, 0]*height_histogram[k]*bin_width
    average_gamma += height_histogram[k]*bin_width/mobility[0, 0]

  return [average_mu, average_gamma]


if __name__ == "__main__":
  # Open data file.
  data_name = os.path.join(
    'data', 
    'icosahedron-msd-dt-0.05-N-500000-end-100.0-scheme-RFD-runs-16-fixed-repulsion.pkl')
  with open(data_name, 'rb') as f:
    msd_statistics = cPickle.load(f)
    print 'Icosahedron parameters:'
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
                                  'sphere-msd-dt-0.05-N-1000000-final.pkl')

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
  # This is for the mass = 0.5 Sphere and nonuniform Icosahedron.
#  average_mob_and_friction = [0.08735]
  
  [zz_msd, rot_msd, rot_perp_msd] = calculate_zz_and_rot_msd_at_equilibrium(15000)
  print 'rot_msd is ', rot_msd
  print 'rot_perp msd is ', rot_perp_msd
#  rot_msd = 0.16666
  # This is for the mass = 0.5 Sphere and nonuniform Icosahedron.
  zz_msd = 0.4557
  
  figure_index = [1, 2, 1, 3, 4, 3]
  label_list = [' icosahedron parallel MSD', ' icosahedron yy MSD', 
                ' blob perpendicular MSD', 
                ' icosahedron 4-4 MSD', ' icosahedron rotational MSD', 
                ' icosahedron 6-6 MSD']
  sphere_label_list = [' blob parallel MSD (a = 0.5)', ' blob yy MSD', 
                       ' blob perpendicular MSD', 
                       ' blob 4-4 MSD', ' blob rotational MSD',
                       ' blob 6-6 MSD']
  sphere_mobility = 0.17963
  
  style_list = ['*', 's', '^', 'd', '.', 'h']
  sphere_style_list = ['d', 'o', 'o', 'o', 'o', 's']
  translation_plot_limit = 110.
  for l in range(6):
    ind = [l, l]
    if l in [0, 2]:
      data_name = 'TranslationalMSDComponent.txt'
      num_err_bars = 120
    elif l == 3:
      data_name = 'RotationalMSDComponent.txt'
      num_err_bars = 120
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
                  [0.0, translation_plot_limit*2.*sph.KT*sphere_mobility], 'k-',
                  lw=2, label='blob parallel mobility')
#      pyplot.plot([0., translation_plot_limit], [0., translation_plot_limit*average_mob_and_friction[0]*2.*2.*ic.KT], 'k--', lw=2,
#                  label='Slope = Icosahedron Mobility')
    if l == 2:
      pyplot.plot([0., translation_plot_limit], [zz_msd, zz_msd], 'k--',  
                  lw=2, label='blob asymptotic perp MSD')
      pyplot.xlim([0., translation_plot_limit,])
      pyplot.ylim([0., translation_plot_limit*2.2*sph.KT*sphere_mobility])
    if l == 5:
      pyplot.plot([0., 60.], [rot_msd, rot_msd], 'k-', 
                  lw=2, label='blob asymptotic 4-4 MSD')
      pyplot.plot([0., 60.], [rot_perp_msd, rot_perp_msd], 'k--', 
                  lw=2, label='blob asymptotic 6-6 MSD')
      pyplot.xlim([0., 60.])

    pyplot.title('MSD(t) for Icosahedron with Hydrodynamic Radius = 0.5')
    pyplot.legend(loc='best', prop={'size': 12})
    pyplot.savefig('./figures/IcosahedronTimeDependentMSD-Component-%s-%s.pdf' % 
                   (ind[0], ind[1]))
  print "Blob mobility is ", sphere_mobility
  print "Asymptotic zz MSD for Icosahedron is", zz_msd
  print "Icosahedron mobility is ", average_mob_and_friction[0]
