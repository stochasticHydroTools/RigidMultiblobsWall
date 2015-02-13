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

DT_STYLES = {}

def plot_time_dependent_msd(msd_statistics, ind, figure):
  ''' 
  Plot the <ind> entry of the rotational MSD as 
  a function of time on given figure (integer).  
  This uses the msd_statistics object
  that is saved by the tetrahedron_rotational_msd.py script.
  ind contains the indices of the entry of the MSD matrix to be plotted.
  ind = [row index, column index].
  '''
  scheme_colors = {'RFD': 'g', 'FIXMAN': 'b', 'EM': 'r'}
  pyplot.figure(figure)
  # Types of lines for different dts.
  write_data = False
  if write_data:
    np.set_printoptions(threshold=np.nan)
  num_err_bars = 12
  linestyles = ['', ':', '--', '-.']

  for scheme in msd_statistics.data.keys():
    dt_num = 0
    for dt in msd_statistics.data[scheme].keys():
      if dt in DT_STYLES.keys():
        dt_style = DT_STYLES[dt]
      else:
        dt_style = linestyles[len(DT_STYLES)]
        DT_STYLES[dt] = dt_style
      # Extract the entry specified by ind to plot.
      num_steps = len(msd_statistics.data[scheme][dt][0])
      # Don't put error bars at every point
      err_idx = [int(num_steps*k/num_err_bars) for k in range(num_err_bars)]
      msd_entries = np.array([msd_statistics.data[scheme][dt][1][_][ind[0]][ind[1]]
                              for _ in range(num_steps)])
      msd_entries_std = np.array(
        [msd_statistics.data[scheme][dt][2][_][ind[0]][ind[1]]
         for _ in range(num_steps)])
      pyplot.plot(msd_statistics.data[scheme][dt][0],
                  msd_entries,
                  scheme_colors[scheme] + dt_style,
                  label = '%s, dt=%s' % (scheme, dt))
      if write_data:
        with open("./MSD-component-%s-%s.txt" % (ind[0], ind[1]),'w+') as f:
          f.write("scheme %s \n" % scheme)
          f.write("dt %s \n" % dt)
          f.write("time: %s \n" % msd_statistics.data[scheme][dt][0])
          f.write("MSD component: %s \n" % msd_entries)
          f.write("Std Dev:  %s \n" % msd_entries_std)
        
      pyplot.errorbar(np.array(msd_statistics.data[scheme][dt][0])[err_idx],
                      msd_entries[err_idx],
                      yerr = 2.*msd_entries_std[err_idx],
                      fmt = scheme_colors[scheme] + '.')
      dt_num += 1
  pyplot.ylabel('MSD')
  pyplot.xlabel('time')



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

