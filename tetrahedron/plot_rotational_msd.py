''' 
Plot rotational msd data from a pickle file. 
'''
import os
import sys
sys.path.append('..')
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot
import numpy as np
import cPickle

from tetrahedron_rotational_msd import MSDStatistics

def plot_time_dependent_msd(msd_statistics, ind):
  ''' 
  Plot the <ind> entry of the rotational MSD as 
  a function of time.  This uses the msd_statistics object
  that is saved by tetrahedron_rotational_msd.py.
  
  ind contains the indices of the entry of the MSD matrix to be plotted.
  ind = [row index, column index].
  '''
  # Types of lines for different dts.
  write_data = False
  if write_data:
    np.set_printoptions(threshold=np.nan)
  dt_styles = ['', ':', '--', '-.']
  scheme_colors = ['b','g','r']
  scheme_num = 0
  num_err_bars = 12

  for scheme in msd_statistics.data.keys():
    dt_num = 0
    for dt in msd_statistics.data[scheme].keys():
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
                  scheme_colors[scheme_num] + dt_styles[dt_num],
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
                      fmt = scheme_colors[scheme_num] + '.')
      dt_num += 1
    scheme_num += 1
  pyplot.ylabel('MSD')
  pyplot.xlabel('time')
  pyplot.legend(loc='best', prop={'size': 9})


if __name__ == "__main__":
  # Make sure figures folder exists
  if not os.path.isdir(os.path.join(os.getcwd(), 'figures')):
    os.mkdir(os.path.join(os.getcwd(), 'figures'))
  # Open data file.
  data_name = os.path.join('data', sys.argv[1])
  with open(data_name, 'rb') as f:
    msd_statistics = cPickle.load(f)
  
  ind = [3, 3]
  plot_time_dependent_msd(msd_statistics, ind)
  pyplot.title('MSD(t) for Tetrahedron')
  pyplot.savefig('./figures/TimeDependentRotationalMSD-Component-%s.pdf' % 
                   (ind))

