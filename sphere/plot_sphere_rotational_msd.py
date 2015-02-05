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

from tetrahedron.tetrahedron_rotational_msd import MSDStatistics
from tetrahedron.plot_rotational_msd import plot_time_dependent_msd


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
  pyplot.title('MSD(t) for Sphere')
  pyplot.savefig('./figures/TimeDependentRotationalMSD-Component-%s.pdf' % 
                   (ind))
