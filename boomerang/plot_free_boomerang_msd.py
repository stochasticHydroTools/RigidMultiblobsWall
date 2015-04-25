''' 
Script to compare the MSD for a free and constrained boomerang,
looking at the parallel (or for the free boomerang, the total) MSD
using different tracking points.
'''

import cPickle
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import numpy as np
import os
import sys
sys.path.append('..')

import boomerang as bm
from quaternion_integrator.quaternion import Quaternion
from utils import plot_time_dependent_msd


def calculate_free_boomerang_avg_mobilities(n_samples):
  ''' Sample random orientations, and calculate trace of
  the translational mobility.  This is done using both CoH and
  CoD'''
  location = [0., 0., 9000000.]
  avg_coh_mobility = 0.
  avg_cod_mobility = 0.
  avg_tip_mobility = 0.
  for k in range(n_samples):
    orientation = np.random.normal(0., 1., 4)
    orientation = Quaternion(orientation/np.linalg.norm(orientation))
    coh = bm.calculate_boomerang_coh(location, orientation)
    mobility = bm.boomerang_mobility_at_arbitrary_point([location],
                                                        [orientation],
                                                        coh)
    avg_coh_mobility += mobility[0, 0] + mobility[1, 1] + mobility[2, 2]
    cod = bm.calculate_boomerang_cod(location, orientation)
    mobility = bm.boomerang_mobility_at_arbitrary_point([location],
                                                        [orientation],
                                                        cod)
    avg_cod_mobility += mobility[0, 0] + mobility[1, 1] + + mobility[2, 2]
    r_vectors = bm.get_boomerang_r_vectors_15(location, orientation)
    tip = r_vectors[0]
    mobility = bm.boomerang_mobility_at_arbitrary_point([location],
                                                        [orientation],
                                                        tip)
    avg_tip_mobility += mobility[0, 0] + mobility[1, 1] + + mobility[2, 2]

    
  avg_coh_mobility /= (3.*n_samples)
  avg_cod_mobility /= (3.*n_samples)
  avg_tip_mobility /= (3.*n_samples)
  return [avg_coh_mobility, avg_cod_mobility, avg_tip_mobility]

    
if __name__ == '__main__':

  # Specify MSD data files, located in ./data subfolder.
  cod_data = 'free-boomerang-msd-dt-0.01-N-300000-end-30.0-scheme-RFD-runs-4-final-CoD.pkl'
  coh_data = 'free-boomerang-msd-dt-0.01-N-300000-end-30.0-scheme-RFD-runs-4-final-CoH.pkl'
  tip_data = 'free-boomerang-msd-dt-0.01-N-300000-end-30.0-scheme-RFD-runs-4-final-tip.pkl'
  
  out_data_file = os.path.join('.', 'data', 'FreeBoomerangMSD-data.txt')
  with open(out_data_file, 'w') as f:
    f.write('Translational MSD for free boomerang (xx + yy + zz).\n')
    f.write('dt = 0.01 for all runs \n')
    f.write('time series: \n' )


  labels = [' CoH', ' CoD', ' tip']
  symbols = ['s', 'd', 'o']

  translation_limit = 10.
  
  ctr = 0
  for name in [cod_data, coh_data, tip_data]:
    file_name = os.path.join('.', 'data', 
                             name)
    with open(file_name, 'rb') as f:
      msd_statistics = cPickle.load(f)
      msd_statistics.print_params()

#     # Add xx and yy and zz to get translational data.
    for scheme in msd_statistics.data:
      for dt in msd_statistics.data[scheme]:
        for k in range(len(msd_statistics.data[scheme][dt][1])):
          msd_statistics.data[scheme][dt][1][k][0][0] = (
            msd_statistics.data[scheme][dt][1][k][0][0] +
            msd_statistics.data[scheme][dt][1][k][1][1] +
            msd_statistics.data[scheme][dt][1][k][2][2])
          msd_statistics.data[scheme][dt][2][k][0][0] = np.sqrt(
            msd_statistics.data[scheme][dt][2][k][0][0]**2 +
            msd_statistics.data[scheme][dt][2][k][1][1]**2 +
            msd_statistics.data[scheme][dt][2][k][2][2]**2)

    plot_time_dependent_msd(msd_statistics, [0, 0], 1, num_err_bars=120,
                            label=labels[ctr], symbol=symbols[ctr])
    with open(out_data_file, 'a') as f:
      f.write('time: \n')
      f.write('%s \n' % msd_statistics.data['RFD'][0.01][0])
      f.write('%s Parallel MSD \n' % labels[ctr])
      f.write('%s \n' % ([msd_statistics.data['RFD'][0.01][1][k][0][0] for k
                         in range(len(msd_statistics.data['RFD'][0.01][0]))]))
      
      
    ctr += 1

  # Plot mobility theory.
  [mu_coh, mu_cod, mu_tip] = calculate_free_boomerang_avg_mobilities(7000)
  # plt.plot([0., translation_limit], 
  #          [0., 6.*bm.KT*mu_coh*translation_limit],
  #          'k--', label='CoH theory')
  print "bulk diffusivity ", bm.KT*mu_coh 
  plt.plot([0., translation_limit], 
           [0., 6.*bm.KT*mu_cod*translation_limit],
           'k-', label='CoD Theory')
  plt.plot([0., translation_limit/5.], 
           [0., 6.*bm.KT*mu_tip*translation_limit/5.],
           'k--', label='tip Theory')
  plt.legend(loc='best', prop={'size': 10})
  plt.xlim([0., translation_limit])
  plt.ylim([0., translation_limit*1.7])
#  plt.title('Location MSD for different gravities of Boomerang.')
  plt.savefig(os.path.join('.', 'figures', 'FreeBoomerangMSD.pdf'))
    

