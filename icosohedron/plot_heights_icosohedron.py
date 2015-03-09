''' Plot heights data for Icosohedron (and Theta if it exists from
a nonuniform Icosohedron run). '''

import cPickle
import numpy as np
from matplotlib import pyplot
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
  colors = ['g', 'b', 'r']
  lines = ['--', ':', '-.']
  symbols = ['o', 's', '^']
  # Get buckets and names. We assume these are the same for all runs.
  # TODO: Allow different buckets.
  buckets = heights_data[0]['buckets']
  error_indices = range(0, len(buckets), len(buckets)/15)
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
  std_heights = np.std(all_heights_data, axis=0)
  if 'thetas' in heights_data[0]:
    theta_buckets = heights_data[0]['theta_buckets']
    theta_error_indices = range(0, len(theta_buckets), 
                                len(theta_buckets)/15)
    average_theta = np.mean(all_theta_data, axis=0)
    std_theta = np.std(all_theta_data, axis=0)

  pyplot.figure(1)
  # HACK: DON'T PLOT EM FOR NOW.
  for k in range(len(average_heights)):
    pyplot.plot(buckets, average_heights[k], colors[k] + lines[k], 
                label=names[k])
    pyplot.errorbar(buckets[error_indices], average_heights[k][error_indices],
                    fmt=(colors[k] + symbols[k]),
                    yerr=2.*std_heights[k][error_indices])

  equilibrium_heights = generate_equilibrium_heights(buckets)
  pyplot.plot(buckets, equilibrium_heights, 'k-', linewidth=2, 
              label='Gibbs Boltzmann')
  pyplot.legend(loc='best', prop={'size': 9})
  pyplot.title('PDF of Height distribution of Icosahedron')
  pyplot.xlabel('Height')
  pyplot.ylabel('PDF')
  pyplot.xlim([0., 5.])    
  pyplot.savefig('./figures/IcosahedronHeightDistribution.pdf')

  if 'thetas' in heights_data[0]:
    # Plot theta as well.
    pyplot.figure(2)
    for k in range(len(average_theta)):
      pyplot.plot(theta_buckets, average_theta[k], 
                  colors[k] + lines[k], label=names[k])
      pyplot.errorbar(theta_buckets[theta_error_indices], 
                      average_theta[k][theta_error_indices],
                      fmt=(colors[k] + symbols[k]), 
                      yerr=2.*std_theta[k][theta_error_indices])
    
    equilibrium_thetas = generate_equilibrium_thetas(theta_buckets)
    # HACK, accidentally bucketed negative theta.
    pyplot.plot(theta_buckets, equilibrium_thetas, 'k-', label='Gibbs Boltzmann')
    pyplot.legend(loc='best', prop={'size': 9})
    pyplot.title('PDF of Theta Distribution of Icosahedron.')
    pyplot.xlabel('Theta')
    pyplot.ylabel('PDF')
    pyplot.savefig('./figures/IcosahedronThetaDistribution.pdf')

    
if __name__ == '__main__':
  
  heights_data = []
  data_name = './data/%s' % sys.argv[1]
  with open(data_name, 'rb') as data:
    heights_data.append(cPickle.load(data))

  print "params are: ", heights_data[0]['params']
  plot_heights_and_theta(heights_data)

