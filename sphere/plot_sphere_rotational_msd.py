''' 
Plot rotational msd data from a pickle file. 
'''
import os
import sys
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot
import numpy as np
import pickle
import sys
sys.path.append('..')

from general_application_utils import plot_time_dependent_msd
from general_application_utils import MSDStatistics


if __name__ == "__main__":
  # Open data file.
  data_name = os.path.join('data', sys.argv[1])
  with open(data_name, 'rb') as f:
    msd_statistics = pickle.load(f)
    msd_statistics.print_params()

  ind = [0, 0]
  plot_time_dependent_msd(msd_statistics, ind, 1)
  pyplot.title('MSD(t) for Sphere')
  pyplot.savefig('./figures/SphereTimeDependentMSD-Component-%s.pdf' % 
                   (ind))


