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

from tetrahedron.plot_rotational_msd import plot_time_dependent_msd
from utils import MSDStatistics


if __name__ == "__main__":
  # Open data file.
  data_name = os.path.join('data', sys.argv[1])
  with open(data_name, 'rb') as f:
    msd_statistics = cPickle.load(f)

  ind = [0, 0]
  plot_time_dependent_msd(msd_statistics, ind)
  pyplot.title('MSD(t) for Sphere')
  pyplot.savefig('./figures/SphereTimeDependentMSD-Component-%s.pdf' % 
                   (ind))
