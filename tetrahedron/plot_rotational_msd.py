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

from utils import MSDStatistics
from utils import plot_time_dependent_msd





if __name__ == "__main__":
  parser = argparse.ArgumentParser(description='Plot results of Rotational MSD '
                                   'Simulations from pkl files created by '
                                   'tetrahedron_rotational_msd.py.')
  parser.add_argument('-dts', dest='dts', type=float, nargs = '+',
                      help='Timesteps to plot')
  parser.add_argument('-N', dest='n_steps', type=int,
                      help='Number of steps taken.')
  parser.add_argument('-files', dest='n_files', type=int, default = None,
                      help='Number of data files at each step to combine. '
                      'This assumes that the data files are named *-1.pkl, '
                      '*-2.pkl, etc.')
  parser.add_argument('-schemes', dest='schemes', type=str, nargs='+',
                      help='Schemes to plot')
  parser.add_argument('-initial', dest='initial', type=bool, default=False,
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

  combined_msd_statistics = None
  # Open data file.
  data_path = os.path.join(os.getcwd(), 'data')
  for dt in args.dts:
    for scheme in args.schemes:
      if args.n_files:
        time = None
        average_msd = None
        std_msd = None
        for k in range(args.n_files):
          data_file = ('rot-msd-initial-%s-location-%s-scheme-%s'
                       '-dt-%s-N-%s-%s-%s.pkl' % (
                         args.initial, args.has_location, scheme, dt, args.n_steps,
                         args.data_name, k+1))
          data_name = os.path.join('data', data_file)
          with open(data_name, 'rb') as f:
            msd_statistics = cPickle.load(f)
            msd_statistics.print_params()
          if time is None:
            time = msd_statistics.data[scheme][dt][0]
            average_msd = msd_statistics.data[scheme][dt][1]
            std_msd = msd_statistics.data[scheme][dt][2]**2
          else:
            average_msd += msd_statistics.data[scheme][dt][1]
            std_msd += msd_statistics.data[scheme][dt][2]**2
        
        average_msd /= float(args.n_files)
        std_msd = np.sqrt(std_msd)/float(args.n_files)
        run_data = [time, average_msd, std_msd]
        if not combined_msd_statistics:
          combined_msd_statistics = MSDStatistics(msd_statistics.params)
        print "adding run for dt = %s, scheme = %s" % (dt, scheme)
        combined_msd_statistics.add_run(scheme, dt, run_data)
      else:
        data_file = ('rot-msd-initial-%s-location-%s-scheme-%s'
                     '-dt-%s-N-%s-%s.pkl' % (
                       args.initial, args.has_location, scheme, dt, args.n_steps,
                       args.data_name))
        data_name = os.path.join('data', data_file)
        with open(data_name, 'rb') as f:
          msd_statistics = cPickle.load(f)
          msd_statistics.print_params()
        if not combined_msd_statistics:
          combined_msd_statistics = msd_statistics
        else:
          combined_msd_statistics.add_run(scheme, dt, msd_statistics.data[scheme][dt])
          
  for l in range(6):
    ind = [l, l]
    plot_time_dependent_msd(combined_msd_statistics, ind, l)
    pyplot.figure(l)
    pyplot.title('MSD(t) for Tetrahedron')
    pyplot.legend(loc='best', prop={'size': 9})
    pyplot.savefig('./figures/TimeDependentRotationalMSD-Component-%s.pdf' % 
                   ([l, l]))

