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
from utils import MSDStatistics
from utils import plot_time_dependent_msd

if __name__ == '__main__':
  # Don't care about paramters here, pass an empty dictionary.
  combined_msd_statistics = MSDStatistics({})
  label_list = ['EM', 'RFD']
  colors = ['b', 'g']
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

  for l in range(6):
    pyplot.figure(l)
    pyplot.title('MSD(t) for Tetrahedron')
    pyplot.legend(loc='best', prop={'size': 9})
    pyplot.savefig('./figures/TimeDependentRotationalMSD-Component-%s.pdf' % 
                   ([l, l]))
  
    
    
  
