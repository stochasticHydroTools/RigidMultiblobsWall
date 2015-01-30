import os
import sys
sys.path.append('..')
import numpy as np
import tetrahedron as tdn
from matplotlib import pyplot
from quaternion_integrator.quaternion import Quaternion
import cPickle

def distribution_height_particle(heights, buckets, names):
  ''' 
  Given histograms of heights for schemes, plot the distributions 
  of height for each particle.
  '''
  if len(names) != len(heights):
    raise Exception('Heights and names must have the same length.')

  for particle in range(3):
    fig = pyplot.figure()
    ax = fig.add_subplot(1, 1, 1)
    for k in range(len(heights)):
      pyplot.plot(buckets, heights[k][particle],  label=names[k])

    pyplot.legend(loc='best', prop={'size': 9})
    pyplot.title('Location of particle %d' % particle)
    pyplot.ylabel('Probability Density')
    pyplot.xlabel('Height')
    # ax.set_yscale('log')
    pyplot.savefig('./figures/Height%d_Distribution.pdf' % particle)


def check_first_order_height_distribution(heights, buckets, names):
  ''' 
  Plot just the discrepency between each scheme and the equilibrium, which is
  assumed to be the last entry in heights.
  '''
  # TODO: Buckets shouldl be determined in the script, not at plot time.
  for particle in range(3):
    fig = pyplot.figure()
    for k in range(len(heights) - 1):
      pyplot.plot(buckets, heights[k][particle] - heights[-1][particle], 
                  label = names[k])
  

if __name__ == '__main__':
  # Make sure figures folder exists
  if not os.path.isdir(os.path.join(os.getcwd(), 'figures')):
    os.mkdir(os.path.join(os.getcwd(), 'figures'))
  # Load data and plot.
  data_name = './data/%s' % sys.argv[1]
  with open(data_name, 'rb') as data:
    height_data = cPickle.load(data)

  distribution_height_particle(height_data['heights'],
                               height_data['buckets'],
                               height_data['names'])
