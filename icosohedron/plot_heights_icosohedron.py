''' Plot heights data for Icosohedron (and Theta if it exists from
a nonuniform Icosohedron run). '''

import cPickle
import numpy as np
from matplotlib import pyplot
import os
import sys

import icosohedron as ic
import icosohedron_nonuniform as icn


def generate_equilibrium_heights(buckets):
  ''' Generate the equilibrium height distribution'''
  distribution = []
  for x in buckets:
    if x < ic.A:
      distribution.append(0.)
      continue
    # WARNING: This assumes uniform and nonuniform icosohedrons
    # have the *same* total mass!!  This must be changed for uniform.
    potential = x*sum(icn.M) + (ic.REPULSION_STRENGTH*np.exp(-1.*(x - ic.A)/
                                                              ic.DEBYE_LENGTH)/
                                (x - ic.A))
    distribution.append(np.exp(-1.*potential/ic.KT))

  distribution = np.array(distribution)
  distribution /= sum(distribution)*(buckets[1] - buckets[0])
  return distribution

def generate_equilibrium_thetas(theta_buckets):
  ''' 
  Generate the equilibrium theta distribution for nonuniform 
  Icosohedron distribution.
  '''
  distribution = []
  for theta in theta_buckets:
    gibbs = np.exp((icn.M[11])*np.cos(theta)*ic.A/ic.KT)
    distribution.append(gibbs*np.sin(theta))

  distribution = np.array(distribution)
  distribution /= sum(distribution)*(theta_buckets[1] - theta_buckets[0])
  return distribution
  

def plot_heights_and_theta(heights_data):
  ''' Plot height histogram and also theta histogram if the data exists.
  heights_data is a list of height_data dictionaries produced by 
  the icosohedron.py or icosohedron_nonuniform.py scripts.
  It is assumed that the same schemes exist in each of the 
  runs, and the same buckets are used.
  '''
  write_data = True
  colors = ['b', 'g', 'r']
  lines = ['--', ':', '-.']
  symbols = ['o', 's', '^']
  # Get buckets and names. We assume these are the same for all runs.
  # TODO: Allow different buckets.
  buckets = heights_data[0]['buckets']
  error_indices = range(0, len(buckets), len(buckets)/40)
  names = heights_data[0]['names']
  avg_heights = np.zeros(len(buckets))
  std_heights = 0.0
  all_heights_data = []
  all_theta_data = []
  for k in range(len(heights_data)):
    all_heights_data.append(heights_data[k]['heights'])
    if 'thetas' in heights_data[k]:
      all_theta_data.append(heights_data[k]['thetas'])

  average_heights = np.mean(all_heights_data, axis=0)
  std_heights = np.std(all_heights_data, axis=0)/np.sqrt(len(heights_data))
  if 'thetas' in heights_data[0]:
    theta_buckets = heights_data[0]['theta_buckets']
    theta_error_indices = range(0, len(theta_buckets), 
                                len(theta_buckets)/15)
    average_theta = np.mean(all_theta_data, axis=0)
    std_theta = np.std(all_theta_data, axis=0)

  pyplot.figure(1)
  # HACK: DON'T PLOT EM FOR NOW.
  for k in range(len(average_heights) - 1):
    pyplot.plot(buckets, average_heights[k], colors[k] + lines[k], 
                label=names[k])
    pyplot.errorbar(buckets[error_indices], average_heights[k][error_indices],
                    fmt=(colors[k] + symbols[k]),
                    yerr=2.*std_heights[k][error_indices])

  equilibrium_heights = generate_equilibrium_heights(buckets)
  pyplot.plot(buckets, equilibrium_heights, 'k-', linewidth=2, 
              label='Gibbs Boltzmann')
  pyplot.legend(loc='best', prop={'size': 13})
  pyplot.title('PDF of Height distribution of Icosahedron')
  pyplot.xlabel('Height')
  pyplot.ylabel('PDF')
  pyplot.xlim([0., 5.])    
  pyplot.savefig('./figures/IcosahedronHeightDistribution.pdf')

  pyplot.figure(3)
  # Plot Error:
  for k in range(len(average_heights) - 1):
    pyplot.plot(buckets, average_heights[k] - equilibrium_heights, colors[k] + lines[k], 
                label=names[k])
    pyplot.errorbar(buckets[error_indices], 
                    average_heights[k][error_indices] - equilibrium_heights[error_indices],
                    fmt=(colors[k] + symbols[k]),
                    yerr=2.*std_heights[k][error_indices])
  pyplot.legend(loc='best', prop={'size': 13})
  pyplot.title('Error Height distribution of Icosahedron')
  pyplot.xlabel('Height')
  pyplot.ylabel('Error in PDF')
  pyplot.xlim([0., 5.])    
  pyplot.savefig('./figures/IcosahedronHeightError.pdf')
    
  if write_data:
    with open('./data/IcosohedronHeightDistribution-data.txt', 'w') as f:
      f.write('Icosohedron Height PDF data\n')
      f.write('Height Buckets:\n')
      f.write('%s \n' % buckets)
      for k in range(len(average_heights)):
        f.write('Scheme %s PDF:\n' % names[k])
        f.write('%s \n' % average_heights[k])
        f.write('Standard Deviation:')
        f.write('%s \n' % std_heights[k])
        f.write('  \n')

      f.write('Equilibrium PDF\n')
      f.write('%s \n' % equilibrium_heights)
      
  if 'thetas' in heights_data[0]:
    # Plot theta as well.
    pyplot.figure(2)
    for k in range(len(average_theta) - 1):
      pyplot.plot(theta_buckets, average_theta[k], 
                  colors[k] + lines[k], label=names[k])
      pyplot.errorbar(theta_buckets[theta_error_indices], 
                      average_theta[k][theta_error_indices],
                      fmt=(colors[k] + symbols[k]), 
                      yerr=2.*std_theta[k][theta_error_indices])
    
    equilibrium_thetas = generate_equilibrium_thetas(theta_buckets)
    # HACK, accidentally bucketed negative theta.
    pyplot.plot(theta_buckets, equilibrium_thetas, 'k-', label='Gibbs Boltzmann')
    pyplot.legend(loc='best', prop={'size': 13})
    pyplot.title('PDF of Theta Distribution of Icosahedron.')
    pyplot.xlabel('Theta')
    pyplot.ylabel('PDF')
    pyplot.savefig('./figures/IcosahedronThetaDistribution.pdf')

    pyplot.figure(4)
    for k in range(len(average_theta)):
      pyplot.plot(theta_buckets, average_theta[k] - equilibrium_thetas, 
                  colors[k] + lines[k], label=names[k])
      pyplot.errorbar(theta_buckets[theta_error_indices], 
                      average_theta[k][theta_error_indices] - equilibrium_thetas[theta_error_indices],
                      fmt=(colors[k] + symbols[k]), 
                      yerr=2.*std_theta[k][theta_error_indices])
    
    # HACK, accidentally bucketed negative theta.
    pyplot.legend(loc='best', prop={'size': 14})
    pyplot.title('PDF of Theta PDF Error of Icosahedron.')
    pyplot.xlabel('Theta')
    pyplot.ylabel('Error in PDF')
    pyplot.savefig('./figures/IcosahedronThetaError.pdf')


  if write_data:
    with open('./data/IcosohedronThetaDistribution-data.txt', 'w') as f:
      f.write('Icosohedron Theta PDF data\n')
      f.write('Theta Buckets:\n')
      #HACK, negated to fix bad bucketing.
      f.write('%s \n' % theta_buckets)
      for k in range(len(average_theta)):
        f.write('Scheme %s Theta PDF:\n' % names[k])
        f.write('%s \n' % average_theta[k])
        f.write('Standard Deviation: \n')
        f.write('%s \n' % std_theta[k])
        f.write('  \n')

      f.write('Equilibrium Theta PDF\n')
      f.write('%s \n' % equilibrium_thetas)

    
if __name__ == '__main__':


  data_names = ['nonuniform-icosohedron-dt-0.5-N-400000-fixed-heavy-1.pkl',
                'nonuniform-icosohedron-dt-0.5-N-400000-fixed-heavy-2.pkl',
                'nonuniform-icosohedron-dt-0.5-N-400000-fixed-heavy-3.pkl',
                'nonuniform-icosohedron-dt-0.5-N-400000-fixed-heavy-4.pkl',
                'nonuniform-icosohedron-dt-0.5-N-400000-fixed-heavy-5.pkl',
                'nonuniform-icosohedron-dt-0.5-N-400000-fixed-heavy-6.pkl']
  
  heights_data = []
  for file_name in data_names:
    with open(os.path.join('.', 'data', file_name), 'rb') as data:
      heights_data.append(cPickle.load(data))

  plot_heights_and_theta(heights_data)


  
