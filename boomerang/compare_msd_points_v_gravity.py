''' Script to compare MSD calculated using different points 
for different gravities.
'''
import cPickle
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import numpy as np
import os
import sys
sys.path.append('..')
  

if __name__ == '__main__':
  
  gfactor = 10.0
  scheme = 'RFD'
  dt = 0.01
  N = 500000
  end = 60.0
  runs = 8
  data_name = 'final'

  colors = ['r', 'g', 'b']
  symbols = ['s', 'o', 'd', '^', 'v','h' ]
  labels = ['cross point', 'tip', 'CoH']
  
  point_ctr = 0
  for out_name in ['', 'tip', 'CoH']:
    if out_name:
      file_name = 'boomerang-msd-dt-%s-N-%s-end-%s-scheme-%s-g-%s-runs-%s-%s-%s.pkl' % (
        (dt, N, end, scheme, gfactor, runs, data_name, out_name))
    else:
      file_name = 'boomerang-msd-dt-%s-N-%s-end-%s-scheme-%s-g-%s-runs-%s-%s.pkl' % (
        (dt, N, end, scheme, gfactor, runs, data_name))
  
    file_name = os.path.join('.', 'data', file_name)

    with open(file_name, 'rb') as f:
      msd_statistics = cPickle.load(f)
      msd_statistics.print_params()

    # Add xx and yy to get translational data (D parallel).
    for sch in msd_statistics.data:
      for t in msd_statistics.data[scheme]:
        for k in range(len(msd_statistics.data[scheme][dt][1])):
          msd_statistics.data[sch][t][1][k][0][0] = (
            msd_statistics.data[sch][t][1][k][0][0] +
            msd_statistics.data[sch][t][1][k][1][1])
          msd_statistics.data[sch][t][2][k][0][0] = np.sqrt(
            msd_statistics.data[sch][t][2][k][0][0]**2 +
            msd_statistics.data[sch][t][2][k][1][1]**2)

    series_len = len(msd_statistics.data[scheme][dt][1])

    plt.errorbar([msd_statistics.data[scheme][dt][0][k] for k in range(series_len)],
                 [msd_statistics.data[scheme][dt][1][k][0][0] for k in range(series_len)],
                 yerr = 2.*np.array(
                   [msd_statistics.data[scheme][dt][2][k][0][0] for k in range(series_len)]),
                 c=colors[point_ctr],
                 marker=symbols[point_ctr],
                 label=labels[point_ctr] + ' parallel')

    plt.errorbar([msd_statistics.data[scheme][dt][0][k] for k in range(series_len)],
                 [msd_statistics.data[scheme][dt][1][k][2][2] for k in range(series_len)],
                 yerr = 2.*np.array(
                   [msd_statistics.data[scheme][dt][2][k][2][2] for k in range(series_len)]),
                 c=colors[point_ctr],
                 marker=symbols[point_ctr + 3],
                 label=labels[point_ctr] + ' perpendicular')
    point_ctr += 1
  plt.legend(loc='best', prop={'size': 10})
  plt.xlim([0., 30.])
  plt.ylim([0., 35.])
  plt.savefig(os.path.join('.', 'figures', 
                           'PointMSDComparison-g-%s.pdf' % gfactor))
