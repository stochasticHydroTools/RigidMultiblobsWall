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
    # have the *same* total mass!!
    potential = x*sum(ic.M) + (ic.REPULSION_STRENGTH*np.exp(-1.*(x - ic.A)/
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
    #HACK, shouldn't be negative theta for new data.
    gibbs = np.exp(-1.*icn.M[11]*np.cos(theta)*ic.A/ic.KT)
    distribution.append(gibbs*np.sin(theta))

  distribution = np.array(distribution)
  distribution /= sum(distribution)*(theta_buckets[1] - theta_buckets[0])
  return distribution
  

def plot_heights_and_theta(heights_data):
  ''' Plot height histogram and also theta histogram if the data exists.'''
  buckets = heights_data['buckets']
  heights = heights_data['heights']
  names = heights_data['names']
  
  pyplot.figure(1)
  for k in range(len(heights)):
    pyplot.plot(buckets, heights[k], label=names[k])
    
  equilibrium_heights = generate_equilibrium_heights(buckets)
  pyplot.plot(buckets, equilibrium_heights, 'k-', label='Gibbs Boltzmann')
  pyplot.legend(loc='best', prop={'size': 9})
  pyplot.title('PDF of Height distribution of Icosohedron')
  pyplot.xlabel('Height')
  pyplot.ylabel('PDF')
  pyplot.savefig('./figures/IcosohedronHeightDistribution.pdf')

  if 'thetas' in heights_data:
    # Plot theta as well.
    thetas = heights_data['thetas']
    theta_buckets = heights_data['theta_buckets']
    pyplot.figure(2)
    for k in range(len(thetas)):
      pyplot.plot(theta_buckets, thetas[k], label=names[k])
    
    equilibrium_thetas = generate_equilibrium_thetas(theta_buckets)
    # HACK, accidentally bucketed negative theta.
    pyplot.plot(-1.*theta_buckets, equilibrium_thetas, 'k-', label='Gibbs Boltzmann')
    pyplot.legend(loc='best', prop={'size': 9})
    pyplot.title('PDF of Theta Distribution of Icosohedron.')
    pyplot.xlabel('Theta')
    pyplot.ylabel('PDF')
    pyplot.savefig('./figures/IcosohedronThetaDistribution.pdf')

    
if __name__ == '__main__':

  data_name = './data/%s' % sys.argv[1]
  with open(data_name, 'rb') as data:
    heights_data = cPickle.load(data)

  print "params are: ", heights_data['params']
  plot_heights_and_theta(heights_data)

