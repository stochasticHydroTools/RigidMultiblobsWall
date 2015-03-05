''' 
Script to check order of accuracy of a scheme by looking
at the error in height distribution.  Use data produced by tetrahedron.py.
'''
import os
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
  the distribution of the height of a particle (0 - 3)
  given by running scheme with timestep dt.  The 
  5th entry (particle = 4), if it exists, is the PDF of the center of 
  the tetrahedron. Run indicates
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
  if buckets[0] < 0.0:
    # Check for incorrect buckets.  Some old data had this, and is
    # not relevant.
    raise Exception('Expect buckets to start above 0.')
    buckets += 2.0

  symbols = ['*', '.', 's', '^', 'x']
  write_data = True
  error_bars = False  # do we plot error bars?

  # assume smallest dt is last dt.
  small_dt_idx = len(heights_list) - 1
  # Just look at the center of the tetrahedron for now.
  # WARNING: This will not work with the fixed tetrahedron or 
  # with older free tetrahedron runs!  In those cases, particle must
  # be between 0 and 2.
  particle = 2
  if write_data:
    with open('./data/EquilibriumDistributionParticle-%s-data.txt' 
              % particle, 'w') as f:
      f.write('Buckets:\n')
      f.write('%s \n' % buckets)
    with open('./data/ErrorParticle-%s-data.txt' 
              % particle, 'w') as f:
      f.write('Buckets:\n')
      f.write('%s \n' % buckets)

  for scheme_idx in range(len(heights_list[0][0]) - 1):
    # Loop through schemes, indexed by scheme_idx
    error_sums = []
    std_sums = []
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

      error_sums.append(sum(abs(error_means))*(buckets[1] - buckets[0]))
      std_sums.append(sum(error_std)*(buckets[1] - buckets[0]))
      # Mean and Std of height over runs.
      height_means = np.mean([heights_list[dt_idx][l][scheme_idx][particle]
                             for l in range(n_runs)], axis=0)
      height_std = np.std([heights_list[dt_idx][l][scheme_idx][particle]
                            for l in range(n_runs)], axis=0)/np.sqrt(n_runs)
      # Figure 1 is just height distribution.
      pyplot.figure(scheme_idx*3)
      if error_bars:
        pyplot.errorbar(buckets, height_means,
                        yerr = 2.*height_std,
                        label = names[scheme_idx] + ', dt=%s' % dts[dt_idx])
      else:
        pyplot.plot(buckets, height_means, symbols[dt_idx] + '--', 
                    label = names[scheme_idx] + ', dt=%s' % dts[dt_idx])

      if dt_idx == small_dt_idx:
        pyplot.figure(12)  
        if error_bars:
          pyplot.errorbar(buckets, height_means,
                          yerr = 2.*height_std,
                          label = names[scheme_idx])
        else:
          pyplot.plot(buckets, height_means, symbols[scheme_idx] + '--',
                      label=names[scheme_idx])
        if write_data:
          with open('./data/EquilibriumDistributionParticle-%s-data.txt' % 
                    particle, 'a') as f:
            f.write('Scheme: %s\n' % names[scheme_idx])
            f.write('Heights PDF:')
            f.write('%s \n' % height_means)

      # Figure 2 is the errors, scaled to check order.
      pyplot.figure(scheme_idx*3 + 1)
      scale_factor = (dts[0]/dts[dt_idx])**order
      if error_bars:
        pyplot.errorbar(buckets, scale_factor*(error_means), 
                        yerr = scale_factor*2.*error_std,
                        label = names[scheme_idx] + ', dt=%s' % dts[dt_idx])
      else:
        pyplot.plot(buckets, scale_factor*(error_means), 
                    symbols[dt_idx] + '--',
                    label = names[scheme_idx] + ', dt=%s' % dts[dt_idx])
      if write_data:
        with open('./data/ErrorParticle-%s-data.txt' 
                  % particle, 'a') as f:
          f.write('Scheme: %s \n' % names[scheme_idx])
          f.write('dt: %s \n' % dts[dt_idx])
          f.write('Error in Height PDF (not scaled):\n')
          f.write('%s \n' % error_means)

    # Figure 3 is a log-log plot of error v. dt.
    pyplot.figure(scheme_idx*3 + 2)
    pyplot.loglog(dts, error_sums, label='%s' % names[scheme_idx])
    pyplot.loglog(dts, error_sums[0]*np.array(dts)/dts[0], 'k--', label='First Order')

    # Now plot the equilibrium for the distribution plots.
    eq_idx = len(heights_list[0][0]) - 1
    # Mean and Std of height over runs.
    height_means = np.mean([heights_list[dt_idx][l][eq_idx][particle]
                             for l in range(n_runs)], axis=0)
    height_std = np.std([heights_list[dt_idx][l][eq_idx][particle]
                            for l in range(n_runs)], axis=0)/np.sqrt(n_runs)
    # Figure 1 is just height distribution.
    pyplot.figure(scheme_idx*3)
    pyplot.errorbar(buckets, height_means,
                    yerr = 2.*height_std,
                    label = 'Monte Carlo')
  pyplot.figure(12)
  # Uncomment this to include error bars.
#  pyplot.errorbar(buckets, height_means,
#                  yerr = 2.*height_std,
#                  label = 'Monte Carlo')
  pyplot.plot(buckets, height_means, 'k-', linewidth=2.0, label='Monte Carlo')
  if write_data:
    with open('./data/EquilibriumDistributionParticle-%s-data.txt' % particle, 'a') as f:
      f.write('Monte Carlo\n')
      f.write('Heights PDF:')
      f.write('%s \n' % height_means)
    
  

  # Title and labels.
  for scheme_idx in range(len(heights_list[0][0]) - 1):
    pyplot.figure(scheme_idx*3)
    pyplot.title('%s Scheme Height Distribution' % names[scheme_idx])
    pyplot.xlabel('Height')
    pyplot.xlim([0., 5.0])
    pyplot.ylabel('PDF')
    pyplot.legend(loc = 'best', prop={'size': 9})
    if particle == 4:
      pyplot.savefig('./figures/HeightRefinement-Scheme-%s-Center.pdf' %
                     (names[scheme_idx]))
    else:
      pyplot.savefig('./figures/HeightRefinement-Scheme-%s-Particle-%s.pdf' %
                     (names[scheme_idx], particle))
    pyplot.figure(scheme_idx*3 + 1)
    pyplot.title('%s - Error in height distribution'  % (names[scheme_idx]))
    pyplot.xlabel('Height')
    pyplot.xlim([0., 5.0])
    pyplot.ylabel('Error in PDF')
    pyplot.legend(loc = 'best', prop={'size': 9})
    if particle == 4:
      pyplot.savefig('./figures/HeightError-Scheme-%s-Center.pdf' %
                     (names[scheme_idx]))
    else:
      pyplot.savefig('./figures/HeightError-Scheme-%s-Particle-%s.pdf' %
                     (names[scheme_idx], particle))
    pyplot.figure(scheme_idx*3 + 2)
    pyplot.title('LogLog plot dt v. Error, %s' % (names[scheme_idx]))
    pyplot.xlim([0., 8.5])
    pyplot.xlabel('Log(dt)')
    pyplot.ylabel('Log(Error)')
    pyplot.legend(loc = 'best', prop={'size': 9})
    if particle == 4:
      pyplot.savefig('./figures/LogLogError-Scheme-%s-Center.pdf' %
                     (names[scheme_idx]))
    else:
      pyplot.savefig('./figures/LogLogError-Scheme-%s-Particle-%s.pdf' %
                     (names[scheme_idx], particle))


  pyplot.figure(12)
  pyplot.title('Equilibrium Distribution for Particle %s, dt=%s' % 
               (particle, dts[small_dt_idx]))
  pyplot.xlabel('Height')
  pyplot.ylabel('PDF')
  pyplot.legend(loc = 'best', prop={'size': 9})
  pyplot.savefig('./figures/EquilibriumDistributionParticle-%s.pdf' %
                 particle)


if __name__  == '__main__':
  # Grab the data from a few runs with different dts, and
  # Check their order.
  # List of lists. Each entry should be a list of names of data
  # files for multiple runs with the same timestep and number of steps.
  data_files = [['tetrahedron-dt-32-N-6000000-run-1-fixed.pkl',
                 'tetrahedron-dt-32-N-6000000-run-2-fixed.pkl',
                 'tetrahedron-dt-32-N-6000000-run-3-fixed.pkl',
                 'tetrahedron-dt-32-N-6000000-run-4-fixed.pkl'],
                ['tetrahedron-dt-16-N-6000000-run-1-fixed.pkl',
                 'tetrahedron-dt-16-N-6000000-run-2-fixed.pkl',
                 'tetrahedron-dt-16-N-6000000-run-3-fixed.pkl',
                 'tetrahedron-dt-16-N-6000000-run-4-fixed.pkl'],
                ['tetrahedron-dt-8-N-6000000-run-1-fixed.pkl',
                 'tetrahedron-dt-8-N-6000000-run-2-fixed.pkl',
                 'tetrahedron-dt-8-N-6000000-run-3-fixed.pkl',
                 'tetrahedron-dt-8-N-6000000-run-4-fixed.pkl'],
                ['tetrahedron-dt-4-N-6000000-run-1-fixed.pkl',
                 'tetrahedron-dt-4-N-6000000-run-2-fixed.pkl',
                 'tetrahedron-dt-4-N-6000000-run-3-fixed.pkl',
                 'tetrahedron-dt-4-N-6000000-run-4-fixed.pkl'],
                ['tetrahedron-dt-2-N-6000000-run-1-fixed.pkl',
                 'tetrahedron-dt-2-N-6000000-run-2-fixed.pkl',
                 'tetrahedron-dt-2-N-6000000-run-3-fixed.pkl',
                 'tetrahedron-dt-2-N-6000000-run-4-fixed.pkl',
                 'tetrahedron-dt-2-N-6000000-run-5-fixed.pkl',
                 'tetrahedron-dt-2-N-6000000-run-6-fixed.pkl',
                 'tetrahedron-dt-2-N-6000000-run-7-fixed.pkl',
                 'tetrahedron-dt-2-N-6000000-run-8-fixed.pkl',
                 'tetrahedron-dt-2-N-6000000-run-9-fixed.pkl',
                 'tetrahedron-dt-2-N-6000000-run-10-fixed.pkl',
                 'tetrahedron-dt-2-N-6000000-run-11-fixed.pkl',
                 'tetrahedron-dt-2-N-6000000-run-12-fixed.pkl',
                 'tetrahedron-dt-2-N-6000000-run-13.pkl',
                 'tetrahedron-dt-2-N-6000000-run-14.pkl',
                 'tetrahedron-dt-2-N-6000000-run-15.pkl',
                 'tetrahedron-dt-2-N-6000000-run-16.pkl',
                 'tetrahedron-dt-2-N-6000000-run-17.pkl',
                 'tetrahedron-dt-2-N-6000000-run-18.pkl',
                 'tetrahedron-dt-2-N-6000000-run-19.pkl',
                 'tetrahedron-dt-2-N-6000000-run-20.pkl',
                 'tetrahedron-dt-2-N-6000000-run-21.pkl',
                 'tetrahedron-dt-2-N-6000000-run-22.pkl',
                 'tetrahedron-dt-2-N-6000000-run-23.pkl',
                 'tetrahedron-dt-2-N-6000000-run-24.pkl',
                 'tetrahedron-dt-2-N-6000000-run-25.pkl',
                 'tetrahedron-dt-2-N-6000000-run-26.pkl',
                 'tetrahedron-dt-2-N-6000000-run-27.pkl',
                 'tetrahedron-dt-2-N-6000000-run-28.pkl',
                 'tetrahedron-dt-2-N-6000000-run-29.pkl',
                 'tetrahedron-dt-2-N-6000000-run-30.pkl',
                 'tetrahedron-dt-2-N-6000000-run-31.pkl',
                 'tetrahedron-dt-2-N-6000000-run-32.pkl']]
  
  dts = [32., 16., 8., 4., 2.]  #This is assumed to be in decreasing order.

  # Free tetrahedron.
  # data_files = [['free-tetrahedron-dt-1.6-N-2000000-four-blobs-1.pkl',
  #                'free-tetrahedron-dt-1.6-N-2000000-four-blobs-2.pkl',],
  #               ['free-tetrahedron-dt-0.8-N-4000000-four-blobs-1.pkl',
  #                'free-tetrahedron-dt-0.8-N-4000000-four-blobs-2.pkl',]]

                
  # dts = [1.6, 0.8] This is assumed to be in decreasing order.
  
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
  check_height_order(heights_list, buckets, names, dts, 0.0)
      
  
