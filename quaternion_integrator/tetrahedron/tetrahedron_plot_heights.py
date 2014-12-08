import sys
sys.path.append('..')
import numpy as np
import tetrahedron as tdn
from matplotlib import pyplot
from quaternion import Quaternion
import cPickle

def distribution_height_particle(heights, bin_width, names):
  ''' 
  Given histograms of heights for Fixman, RFD, EM, and equilibrium (in that
  order), plot the distributions for each height.
  '''
  if len(names) != len(heights):
    raise Exception('Heights and names must have the same length.')

  buckets = tdn.H + bin_width*np.linspace(-2./bin_width, 2./bin_width, len(heights[0][0]))  
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
    pyplot.savefig('./plots/Height%d_Distribution.pdf' % particle)
  
  

if __name__ == '__main__':
  # TODO: keep more data in the pkl file, so that nothing here needs to be specified.
#  names = ['Fixman', 'Gibbs-Boltzmann']
  data_name = './data/%s' % sys.argv[1]
#  bin_width = float(sys.argv[2])  # This should match the bin_width in tetrahedron.py
  with open(data_name, 'rb') as data:
    height_data = cPickle.load(data)

  distribution_height_particle(height_data['heights'],
                               height_data['bin_width'],
                               height_data['names'])
