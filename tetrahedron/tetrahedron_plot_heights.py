import os
import sys
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

  # Test if this is free tetrahedron data by looking at number of 
  # particles we binned.  NOTE: Older runs of the free tetrahedron
  # only consider 3 particles (not the top vertex), and will be 
  # plotted as if they're fixed runs.
  if len(heights[0]) == 5:
    location = True
  else:
    location = False


  for particle in range(3 + 1*location):
    fig = pyplot.figure()
    ax = fig.add_subplot(1, 1, 1)
    for k in range(len(heights)):
      pyplot.plot(buckets, heights[k][particle],  label=names[k])
    
    # #HACK for floren's data
    # if particle == 0:
    #   # Mass 0.005
    #   data_file = './data/hBlob.geometricCenter.mass.0.005.dat'
    # elif particle == 1:
    #   data_file = './data/hBlob.geometricCenter.mass.0.015.dat'
    # elif particle == 2:
    #   data_file = './data/hBlob.geometricCenter.mass.0.01.dat'

    # x = []
    # num = []
    # with open(data_file, 'r') as f:
    #   for line in f:
    #     dat = line.split(' ')
    #     if len(dat) > 1:
    #       x.append(float(dat[0]))
    #       num.append(float(dat[1]))

    # pyplot.plot(x, num, label='IBAMR')
      
    pyplot.legend(loc='best', prop={'size': 9})
    pyplot.title('Location of particle %d' % particle)
    pyplot.ylabel('Probability Density')
    pyplot.xlabel('Height')
    # ax.set_yscale('log')
    pyplot.savefig('./figures/Height%d_Distribution.pdf' % particle)

  # Plot center of tetrahedron.  Only do this for the Free tetrahedron.
  if location:
    fig = pyplot.figure()
    for k in range(len(heights)):
      #HACk, fix buckets from mistake where I subtracted 2.0
      pyplot.plot(buckets, heights[k][4], label=names[k])
    pyplot.legend(loc='best', prop={'size': 9})
    pyplot.title('PDF for Location of center of Tetrahedron')
    pyplot.xlim([0.0, 8.5])
    pyplot.ylabel('Probability Density')
    pyplot.xlabel('Height')
    # ax.set_yscale('log')
    pyplot.savefig('./figures/TetrahedronCenterHeightDistribution.pdf')
    


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
  # Load data and plot.
  data_name = './data/%s' % sys.argv[1]
  with open(data_name, 'rb') as data:
    height_data = cPickle.load(data)

  distribution_height_particle(height_data['heights'],
                               height_data['buckets'],
                               height_data['names'])
