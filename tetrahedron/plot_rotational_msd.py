''' 
Plot rotational msd data from a pickle file. 
'''
import argparse
import cPickle
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot
import numpy as np
import os
import sys
sys.path.append('..')
import re

from utils import MSDStatistics
from utils import plot_time_dependent_msd


if __name__ == "__main__":

  parser = argparse.ArgumentParser(description='Plot results of Rotational MSD '
                                   'Simulations from pkl files created by '
                                   'tetrahedron_rotational_msd.py.')
  parser.add_argument('-dts', dest='dts', type=float, nargs = '+',
                      help='Timesteps to plot')
  parser.add_argument('-schemes', dest='schemes', type=str, nargs='+',
                      help='Schemes to plot')
  parser.add_argument('-initial', dest='initial', type=bool,default=False,
                      help='If true, plot runs that start at one fixed initial '
                      'condition.  If False, plot runs that give equilibrium '
                      'MSD.')
  parser.add_argument('-free', dest='has_location', type=bool,
                      default=True,
                      help='If true, plot runs where Tetrahedron is allowed '
                      'to move.  If False, plot runs where Tetrahedron '
                      'is fixed.')
  parser.add_argument('--data-name', dest='data_name', type=str,
                      help='Name of data runs to plot.  All runs must have '
                      'the same name specified when running '
                      'tetrahedron_rotational_msd.py to plot together. '
                      ' This is easy to change, by just renaming the pkl file.')
  args = parser.parse_args()
  
  # Open data file.
  data_path = os.path.join(os.getcwd(), 'data')
  data_files = os.listdir(data_path)
  for dt in args.dts:
    for scheme in args.schemes:
      wanted_file_name = ('rot-msd-initial-%s-location-%s-scheme-%s'
                          '-dt-%s-N-(.*)-%s.pkl' % (
          args.initial, args.has_location, scheme, dt,
          args.data_name))
      for data_file in data_files:
        if re.match(wanted_file_name, data_file):
          data_name = os.path.join('data', data_file)
          with open(data_name, 'rb') as f:
            msd_statistics = cPickle.load(f)
            msd_statistics.print_params()
          break
        
      for l in range(6):
        ind = [l, l]
        plot_time_dependent_msd(msd_statistics, ind, l)

  for l in range(6):
    pyplot.figure(l)
    pyplot.title('MSD(t) for Tetrahedron')
    pyplot.legend(loc='best', prop={'size': 9})
    pyplot.savefig('./figures/TimeDependentRotationalMSD-Component-%s.pdf' % 
                   ([l, l]))

