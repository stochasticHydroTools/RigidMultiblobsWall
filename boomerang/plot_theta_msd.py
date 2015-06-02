''' Script to read theta MSD from a pkl file and plot the results. '''

import cPickle
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from matplotlib import rc
rc('text', usetex=True)
import os

import boomerang as bm
from config_local import DATA_DIR

def estimate_theta_diffusion(n_samples, f):
  ''' Use MC to estimate the last diagonal of the mobility.'''
  avg_zz_theta_mob = 0.
  with open(sample_file, 'r') as f:
    line = f.readline()
    # Skip parameters. 
    while line != 'Location, Orientation:\n':
      line = f.readline()
    for k in range(n_samples):
      sample = bm.load_equilibrium_sample(f)
      mobility = bm.boomerang_mobility([sample[0]], [sample[1]])
      avg_zz_theta_mob += mobility[5, 5]

  avg_zz_theta_mob /= n_samples
  return avg_zz_theta_mob


if __name__ == '__main__':
  gfactor = 20.
  translation_end = 250.
  colors = ['r', 'g', 'b', 'c', 'k', 'm']

  styles=['-', '--', '-.']
  symbols = ['s', 'd', 'o']

  data_out_file = os.path.join('.', 'data', 'BoomerangThetaMSD-data.txt')
  
  with open(data_out_file, 'w') as f:
    f.write('Data for MSD of theta, the angle that the projection of '
            'the boomerang bisector onto the x-y plane makes with the '
            'z axis:\n')
    f.write('\n')
    f.write('dt: %s \n' % 0.01)
    f.write('N: %s\n' % 500000)
    f.write('scheme: %s\n' % 'RFD')
    f.write('8 Runs\n')
    f.write('\n\n')

  ctr = 0
  with open(data_out_file, 'a') as out:
    for gfactor in [1., 10., 20.]:

      data_file = ('boomerang-theta-msd-dt-0.01-N-500000-end-250.0-'
                   'scheme-RFD-g-%s-runs-8-final.pkl' % gfactor)
      data_file = os.path.join('.', 'data', data_file)
      with open(data_file, 'rb') as f:
        msd_data = cPickle.load(f)
      
      if ctr == 0:
        out.write('Time:\n')
        out.write('%s \n\n' % msd_data[0])
      out.write('Gravity: %s\n' % gfactor)
      out.write('\n')
      out.write('Theta MSD Component:\n')
      out.write('%s \n' % msd_data[1])
      out.write('Theta MSD STD:\n')
      out.write('%s \n' % msd_data[2])
      
      plt.errorbar(msd_data[0], msd_data[1], 
                   yerr = 2.*msd_data[2], 
                   label='rotational MSD, g = %s' % gfactor,
                   c=colors[ctr], lw=1, marker=symbols[ctr],
                   linestyle='--')
      D_theta = (msd_data[1][1] - msd_data[1][0])/(
        msd_data[0][1] - msd_data[0][0])
      print "D_theta is ", D_theta, " for g = ", gfactor
      out.write('D_theta (based on first 2 points):\n')
      out.write('%s\n' % D_theta)
      out.write('\n\n')
  
      sample_file = os.path.join(DATA_DIR, 'boomerang',
                                 'boomerang-samples-g-%s-old.txt' % gfactor)
  
      zz_theta_mob = estimate_theta_diffusion(10, sample_file)
      print "zz_theta_mob is ", zz_theta_mob, " for g = ", gfactor
      plt.plot([0., translation_end], [0., D_theta*translation_end],
               label=r'constant $D_\theta = %.2f$' % D_theta, 
               c=colors[ctr+3], lw=2,
               linestyle=styles[ctr])
      long_D_theta = (msd_data[1][-1] - msd_data[1][-2])/(
        msd_data[0][1] - msd_data[0][0])
      # if ctr == 0:
      #   plt.plot([0., translation_end], [0., long_D_theta*translation_end],
      #            'k:',
      #            label=r'long-time slope', lw=2)
      # else:
      #   plt.plot([0., translation_end], [0., long_D_theta*translation_end],
      #          'k:', lw=2)
      ctr += 1

  plt.ylabel(r'Rotational MSD $(\Delta \theta)^2$')
  plt.xlabel('Time (s)')
  plt.legend(loc='best', prop={'size': 10})
  plt.ylim([0., 400])
  plt.xlim([0., translation_end])
  plt.savefig(os.path.join('.', 'figures', 'BoomerangThetaMSD.pdf'))  
