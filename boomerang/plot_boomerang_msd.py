''' Plot the MSD of a boomerang. '''

import cPickle
import matplotlib
matplotlib.use('Agg')
import numpy as np
from matplotlib import pyplot as plt
import sys
sys.path.append('..')
import os

import boomerang as bm
from config_local import DATA_DIR
from utils import plot_time_dependent_msd


def calculate_boomerang_parallel_mobility(n_samples, sample_file):
  ''' 
  Calculate the boomerang parallel mobility by taking GB samples from
  file and averaging.
  '''
  parallel_mobility = 0.
  with open(sample_file, 'r') as f:
    line = f.readline()
    # Skip parameters. 
    while line != 'Location, Orientation:\n':
      line = f.readline()
    for k in range(n_samples):
      sample = bm.load_equilibrium_sample(f)
      mobility = bm.boomerang_mobility([sample[0]], [sample[1]])
      parallel_mobility += mobility[0, 0] + mobility[1, 1]
    
  parallel_mobility /= (2*n_samples)
  return parallel_mobility


if __name__ == '__main__':
  
  data_files = ['boomerang-msd-dt-0.01-N-500000-end-60.0-scheme-RFD-g-1.0-'
                'runs-8-final-CoH.pkl',
                'boomerang-msd-dt-0.01-N-500000-end-60.0-scheme-RFD-g-10.0-'
                'runs-8-final-CoH.pkl',
                'boomerang-msd-dt-0.01-N-500000-end-60.0-scheme-RFD-g-20.0-'
                'runs-8-final-CoH.pkl']

  labels = [' G = 1 Parallel', ' G = 10 Parallel', ' G = 20 Parallel',
            ' G=1 Perp', ' G = 10 Perp', ' G = 20 Perp']
  symbols = ['d', 'o', 's', '^']
  translation_limit = 20.

  ctr = 0
  for name in data_files:
    file_name = os.path.join('.', 'data', name)
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

    plot_time_dependent_msd(msd_statistics, [0, 0], 1, num_err_bars=120,
                            label=labels[ctr], symbol=symbols[ctr])
    plot_time_dependent_msd(msd_statistics, [2, 2], 1, num_err_bars=120,
                            label=labels[ctr + 3], symbol=symbols[ctr])
    ctr += 1


  # Plot Parallel mobility for different gravities.
  for gfactor in [1., 10., 20.]:
    sample_file = os.path.join(DATA_DIR, 'boomerang',
                               'boomerang-samples-g-%s-old.txt' % gfactor)
    mu_parallel = calculate_boomerang_parallel_mobility(100, sample_file)
    print "mu parallel for g = %s is %s" % (gfactor, mu_parallel)
    plt.figure(1)
    plt.plot([0., translation_limit], [0., 2.*bm.KT*mu_parallel], 'k--')
    
  plt.legend(loc='best', prop={'size': 10})
  plt.xlim([0., translation_limit])
  plt.ylim([0., 20.])
  plt.title('Location MSD for different gravities of Boomerang.')
  plt.savefig(os.path.join('.', 'figures', 'BoomerangMSDPlot.pdf'))
