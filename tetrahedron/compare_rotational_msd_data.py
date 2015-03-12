''' 
Quick script to plot rotational msd data from specified pkl files.
usage: 
  python compare_rotational_msd_data.py rot-msd-data-file-1.pkl
    rot-msd-data-file-2.pkl etc.
''' 

import cPickle
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot
import numpy as np
import os
import sys
sys.path.append('..')

from plot_rotational_msd import calculate_zz_and_rot_msd_at_equilibrium
from translational_diffusion_coefficient import calculate_average_mu_parallel_and_perpendicular
import tetrahedron_free as tf
from utils import MSDStatistics
from utils import plot_time_dependent_msd

if __name__ == '__main__':
  # Don't care about paramters here, pass an empty dictionary.
  combined_msd_statistics = MSDStatistics({})
  label_list = ['vertex', ' center of mass', ' Old', 'new']
  colors = ['b', 'g', 'r', 'c']
  for k in range(1, len(sys.argv)):
    data_file = sys.argv[k]
    data_name = os.path.join('data', data_file)
    with open(data_name, 'rb') as f:
      msd_statistics = cPickle.load(f)
      msd_statistics.print_params()
      for l in range(6):
        ind = [l, l]
        plot_time_dependent_msd(msd_statistics, ind, l, color=colors[k-1],
                                label=label_list[k-1])


  average_mob_and_friction = calculate_average_mu_parallel_and_perpendicular(100)
  [zz_msd, rot_msd] = calculate_zz_and_rot_msd_at_equilibrium(200)
  translation_end = 300.0
  for l in range(6):
    pyplot.figure(l)
    if l in [0, 1]:
      pyplot.plot([0.0, translation_end], 
                  [0.0, translation_end*2.*tf.KT*average_mob_and_friction[0]], 
                  'k--', label='mu parallel vertex')
      pyplot.xlim([0.0, translation_end])
      pyplot.ylim([0., translation_end*2.*tf.KT*average_mob_and_friction[0]])
    elif l == 2:
      pyplot.plot([0.0, translation_end], [zz_msd, zz_msd], 'k--', label='Equilibrium Perp MSD')
      pyplot.xlim([0., translation_end])
      pyplot.ylim([0., translation_end*2.*tf.KT*average_mob_and_friction[0]])
    pyplot.title('MSD(t) for Tetrahedron')
    pyplot.legend(loc='best', prop={'size': 9})
    pyplot.savefig('./figures/TimeDependentRotationalMSD-Component-%s-%s.pdf' % 
                   (l, l))
  
    
    
  
