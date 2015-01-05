''' 
Plot rotational msd data from a pickle file. 
'''
import os
import sys
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot
import numpy as np
import cPickle
from quaternion_integrator.tetrahedron.tetrahedron_rotational_msd import MSDStatistics

def plot_time_dependent_msd(msd_statistics):
  ''' Plot the rotational MSD as a function of time.'''
  # Types of lines for different dts.
  dt_styles = ['', ':', '--']
  scheme_colors = ['b','g','r']
  scheme_num = 0
  for scheme in msd_statistics.data.keys():
    dt_num = 0
    pyplot.figure(scheme_num)
    for dt in msd_statistics.data[scheme].keys():
      pyplot.errorbar(msd_statistics.data[scheme][dt][0], 
                      msd_statistics.data[scheme][dt][1],
                      yerr = 2.*np.array(msd_statistics.data[scheme][dt][2]),
                      fmt = scheme_colors[scheme_num] + dt_styles[dt_num],
                      label = '%s, dt=%s' % (scheme, dt))
      dt_num += 1
    scheme_num += 1
    pyplot.title('MSD(t) for Scheme %s' % scheme)
    pyplot.ylabel('MSD')
    pyplot.xlabel('time')
    pyplot.legend(loc='best', prop={'size': 9})
    pyplot.savefig('./plots/TimeDependentRotationalMSD-%s.pdf' % scheme)


if __name__ == "__main__":
  # Open data file.
  data_name = os.path.join('data', sys.argv[1])
  with open(data_name, 'rb') as f:
    msd_statistics = cPickle.load(f)

  plot_time_dependent_msd(msd_statistics)
