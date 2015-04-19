''' Plot the MSD of a boomerang. '''

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import sys
import os

from utils import plot_time_dependent_msd


if __name__ == '__main__':
  
  data_files = sys.argv[1:]
  labels = ['G = 1', 'G = 10', 'G = 20']

  ctr = 0
  for file_name in range(data_files):
    with open(file_name, 'rb') as f:
      msd_statistics = cPickle.load(f)
      msd_statistics.print_params()

    # Add xx and yy to get translational data (D parallel).
    for scheme in msd_statistics.data:
      for dt in msd_statistics.data[scheme]:
        for k in range(len(msd_statistics.data[scheme][dt][1])):
          msd_statistics.data[scheme][dt][1][k][0][0] = (
            msd_statistics.data[scheme][dt][1][k][0][0] +
            msd_statistics.data[scheme][dt][1][k][1][1])
          msd_statistics.data[scheme][dt][2][k][0][0] = np.sqrt(
            msd_statistics.data[scheme][dt][2][k][0][0]**2 +
            msd_statistics.data[scheme][dt][2][k][1][1]**2)
    plot_time_dependent_msd(msd_statistics, [0, 0], 1, num_err_bars=40,
                            label=labels[ctr], symbol=symbols[ctr])
    plot_time_dependent_msd(msd_statistics, [2, 2], 1, num_err_bars=40,
                            label=labels[ctr], symbol=symbols[ctr])
    ctr += 1
  
  plt.savefig(os.path.join('.', 'figures', 'BoomerangMSDPlot.pdf'))
