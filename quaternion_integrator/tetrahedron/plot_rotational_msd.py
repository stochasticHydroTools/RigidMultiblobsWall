''' 
Plot rotational msd data from a pickle file. 
'''
import os
import sys
sys.path.append('../..')
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot
import numpy as np
import cPickle
from quaternion_integrator.tetrahedron.tetrahedron_rotational_msd import MSDStatistics

def plot_time_dependent_msd(msd_statistics, ind):
  ''' 
  Plot the <ind> entry of the rotational MSD as 
  a function of time.  This uses the msd_statistics object
  that is saved by tetrahedron_rotational_msd.py.
  
  ind contains the indices of the entry of the MSD matrix to be plotted.
  ind = [row index, column index].
  '''
  # Types of lines for different dts.
  dt_styles = ['', ':', '--']
  scheme_colors = ['b','g','r']
  scheme_num = 0

  for scheme in msd_statistics.data.keys():
    dt_num = 0
    pyplot.figure(scheme_num)
    for dt in msd_statistics.data[scheme].keys():
      # Extract the entry specified by ind to plot.
      num_steps = len(msd_statistics.data[scheme][dt][0])
      msd_entries = [msd_statistics.data[scheme][dt][1][_][ind[0]][ind[1]] 
                     for _ in range(num_steps)]
      msd_entries_std = [msd_statistics.data[scheme][dt][2][_][ind[0]][ind[1]] 
                         for _ in range(num_steps)]
      pyplot.errorbar(msd_statistics.data[scheme][dt][0],
                      msd_entries,
                      yerr = 2.*np.array(msd_entries_std),
                      fmt = scheme_colors[scheme_num] + dt_styles[dt_num],
                      label = '%s, dt=%s' % (scheme, dt))
      dt_num += 1
    scheme_num += 1
    pyplot.title('MSD(t) for Scheme %s' % scheme)
    pyplot.ylabel('MSD')
    pyplot.xlabel('time')
    pyplot.legend(loc='best', prop={'size': 9})
    pyplot.savefig('./figures/TimeDependentRotationalMSD-%s.pdf' % scheme)


if __name__ == "__main__":
  # Make sure figures folder exists
  if not os.path.isdir(os.path.join(os.getcwd(), 'figures')):
    os.mkdir(os.path.join(os.getcwd(), 'figures'))
  # Open data file.
  data_name = os.path.join('data', sys.argv[1])
  with open(data_name, 'rb') as f:
    msd_statistics = cPickle.load(f)

  plot_time_dependent_msd(msd_statistics, [5, 5])
