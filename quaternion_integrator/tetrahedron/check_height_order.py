''' 
Script to check order of accuracy of a scheme by looking
at the error in height distribution.  Use data produced by tetrahedron.py.

'''
import cPickle
import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot

def check_height_order(heights_list, buckets, names, dts, order):
  ''' 
  Plot just the discrepency between each scheme and the equilibrium,
  which is assumed to be the last entry in heights.  
  heights_list[dt][run][scheme][particle] is a histogram of
  the distribution of the height of a particle (0 - 2)
  given by running scheme with timestep dt.  Run indicates
  which run this is, and the multiple runs are used for generating error bars.

  buckets gives a list of the midpionts of buckets, which is used for 
  plotting the histogram.
  
  names gives the names of the schemes (FIXMAN, etc) in the same order as they
  are ordered in heights_list[dt][run] for any dt and run.

  dts gives the timesteps used in the simulations, in the same order as the 
  first index of heights_list.

  order is used to scale errors at smaller timesteps for checking the order
  of accuracy of the schemes.
  '''
  # Just look at the heaviest particle for now.
  particle = 2
  for scheme_idx in range(len(heights_list[0][0]) - 1):
    # Loop through schemes, indexed by scheme_idx
    for dt_idx in range(len(heights_list)):
      # Loop through dts, indexed by dt_idx
      n_runs = len(heights_list[dt_idx])
      # Mean and std of error over all runs:
      error_means = np.mean([heights_list[dt_idx][l][scheme_idx][particle] - 
                             heights_list[dt_idx][l][-1][particle] 
                             for l in range(n_runs)], axis=0)
      error_std = np.std([heights_list[dt_idx][l][scheme_idx][particle] - 
                             heights_list[dt_idx][l][-1][particle] 
                             for l in range(n_runs)], axis=0)/np.sqrt(n_runs)

      # Mean and Std of height over runs.
      height_means = np.mean([heights_list[dt_idx][l][scheme_idx][particle]
                             for l in range(n_runs)], axis=0)
      height_std = np.std([heights_list[dt_idx][l][scheme_idx][particle]
                            for l in range(n_runs)], axis=0)/np.sqrt(n_runs)
      # Figure 1 is just height distribution.
      pyplot.figure(scheme_idx*2)
      pyplot.errorbar(buckets, height_means,
                      yerr = 2.*height_std,
                      label = names[scheme_idx] + ', dt=%s' % dts[dt_idx])
      # Figure 2 is the errors, scaled to check order.
      pyplot.figure(scheme_idx*2 + 1)
      scale_factor = (dts[0]/dts[dt_idx])**order
      pyplot.errorbar(buckets, scale_factor*(error_means),
                      yerr = scale_factor*2.*error_std,
                      label = names[scheme_idx] + ', dt=%s' % dts[dt_idx])

    # Now plot the equilibrium for the distribution plots.
    eq_idx = len(heights_list[0][0]) - 1
    # Mean and Std of height over runs.
    height_means = np.mean([heights_list[dt_idx][l][eq_idx][particle]
                             for l in range(n_runs)], axis=0)
    height_std = np.std([heights_list[dt_idx][l][eq_idx][particle]
                            for l in range(n_runs)], axis=0)/np.sqrt(n_runs)
    # Figure 1 is just height distribution.
    pyplot.figure(scheme_idx*2)
    pyplot.errorbar(buckets, height_means,
                    yerr = 2.*height_std,
                    label = 'Equilibrium')

  # Title and 
  for scheme_idx in range(len(heights_list[0][0]) - 1):
    pyplot.figure(scheme_idx*2)
    pyplot.title('%s Scheme Height Distribution' % names[scheme_idx])
    pyplot.xlabel('Height')
    pyplot.ylabel('PDF')
    pyplot.legend(loc = 'best', prop={'size': 9})
    pyplot.savefig('./plots/HeightRefinement-Scheme-%s-Particle-%s.pdf' %
                   (names[scheme_idx], particle))
    pyplot.figure(scheme_idx*2 + 1)
    pyplot.title('%s scheme, order %s test' % (names[scheme_idx], order))
    pyplot.xlabel('Height')
    pyplot.ylabel('Error in height distribution')
    pyplot.legend(loc = 'best', prop={'size': 9})
    pyplot.savefig('./plots/HeightError-Scheme-%s-Particle-%s.pdf' %
                   (names[scheme_idx], particle))



if __name__  == '__main__':
  #  Grab the data from a few runs with different dts, and
  #  Check their order.
  # List of lists. Each entry should be a list of names of data
  # files for multiple runs with the same timestep and number of steps.
  # data_files = [['tetrahedron-dt-32-N-6000000-run-1-fixed.pkl',
  #                'tetrahedron-dt-32-N-6000000-run-2-fixed.pkl',
  #                'tetrahedron-dt-32-N-6000000-run-3-fixed.pkl',
  #                'tetrahedron-dt-32-N-6000000-run-4-fixed.pkl'],
  #               ['tetrahedron-dt-16-N-6000000-run-1-fixed.pkl',
  #                'tetrahedron-dt-16-N-6000000-run-2-fixed.pkl',
  #                'tetrahedron-dt-16-N-6000000-run-3-fixed.pkl',
  #                'tetrahedron-dt-16-N-6000000-run-4-fixed.pkl'],
  #               ['tetrahedron-dt-8-N-6000000-run-1-fixed.pkl',
  #                'tetrahedron-dt-8-N-6000000-run-2-fixed.pkl',
  #                'tetrahedron-dt-8-N-6000000-run-3-fixed.pkl',
  #                'tetrahedron-dt-8-N-6000000-run-4-fixed.pkl'],
  #               ['tetrahedron-dt-4-N-6000000-run-1-fixed.pkl',
  #                'tetrahedron-dt-4-N-6000000-run-2-fixed.pkl',
  #                'tetrahedron-dt-4-N-6000000-run-3-fixed.pkl',
  #                'tetrahedron-dt-4-N-6000000-run-4-fixed.pkl'],
  #               ['tetrahedron-dt-2-N-6000000-run-1-fixed.pkl',
  #                'tetrahedron-dt-2-N-6000000-run-2-fixed.pkl',
  #                'tetrahedron-dt-2-N-6000000-run-3-fixed.pkl',
  #                'tetrahedron-dt-2-N-6000000-run-4-fixed.pkl',
  #                'tetrahedron-dt-2-N-6000000-run-5-fixed.pkl',
  #                'tetrahedron-dt-2-N-6000000-run-6-fixed.pkl',
  #                'tetrahedron-dt-2-N-6000000-run-7-fixed.pkl',
  #                'tetrahedron-dt-2-N-6000000-run-8-fixed.pkl',
  #                'tetrahedron-dt-2-N-6000000-run-9-fixed.pkl',
  #                'tetrahedron-dt-2-N-6000000-run-10-fixed.pkl',
  #                'tetrahedron-dt-2-N-6000000-run-11-fixed.pkl',
  #                'tetrahedron-dt-2-N-6000000-run-12-fixed.pkl']]

  # dts = [32., 16., 8., 4., 2.]


  # Free tetrahedron
  data_files = [['free-tetrahedron-dt-1-N-1000000-run-1.pkl',
                 'free-tetrahedron-dt-1-N-1000000-run-2.pkl',
                 'free-tetrahedron-dt-1-N-1000000-run-3.pkl',
                 'free-tetrahedron-dt-1-N-1000000-run-4.pkl'],
                ['free-tetrahedron-dt-0.5-N-3000000-run-1.pkl',
                 'free-tetrahedron-dt-0.5-N-3000000-run-2.pkl',
                 'free-tetrahedron-dt-0.5-N-3000000-run-3.pkl',
                 'free-tetrahedron-dt-0.5-N-3000000-run-4.pkl'],
                ['free-tetrahedron-dt-0.25-N-3000000-run-1.pkl',
                 'free-tetrahedron-dt-0.25-N-3000000-run-2.pkl',
                 'free-tetrahedron-dt-0.25-N-3000000-run-3.pkl',
                 'free-tetrahedron-dt-0.25-N-3000000-run-4.pkl']]

  dts = [1.0, 0.5, 0.25]

  heights_list = []
  for parameter_set in data_files:
    heights_list.append([])
    for data_file in parameter_set:
      with open('./data/' + data_file, 'rb') as f:
        heights_data = cPickle.load(f)
        heights_list[-1].append(heights_data['heights'])

  # For now assume all runs have the same scheme order and buckets.
  buckets = heights_data['buckets']
  names = heights_data['names']
  check_height_order(heights_list, buckets, names, dts, 1.0)
      
  
