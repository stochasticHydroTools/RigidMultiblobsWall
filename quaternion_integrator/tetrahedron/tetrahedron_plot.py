import sys
sys.path.append('..')
import numpy as np
import tetrahedron as tdn
from matplotlib import pyplot
from quaternion import Quaternion
import cPickle

def distribution_height_particle(particle, paths, names):
  ''' 
  Given paths of a quaternion, make a historgram of the 
  height of particle <particle> and compare to equilibrium. 
  names are used for labeling the plot, and should have the same 
  length as paths. 
  '''
  if len(names) != len(paths):
    raise Exception('Paths and names must have the same length.')
    
  fig = pyplot.figure()
  ax = fig.add_subplot(1, 1, 1)
  hist_bins = np.linspace(-1.9, 1.9, 60) + tdn.H
  for k in range(len(paths)):
    path = paths[k]
    heights = []
    for pos in path:
      # TODO: do this a faster way perhaps with a special function.
      r_vectors = tdn.get_r_vectors(pos[0])
      heights.append(r_vectors[particle][2])

    height_hist = np.histogram(heights, density=True, bins=hist_bins)
    buckets = (height_hist[1][:-1] + height_hist[1][1:])/2.
    pyplot.plot(buckets, height_hist[0],  label=names[k])

  pyplot.legend(loc='best', prop={'size': 9})
  pyplot.title('Location of particle %d' % particle)
  pyplot.ylabel('Probability Density')
  pyplot.xlabel('Height')
#  ax.set_yscale('log')
  pyplot.savefig('./plots/Height%d_Distribution.pdf' % particle)


if __name__ == '__main__':
  names = ['Fixman', 'RFD', 'E-M', 'Gibbs-Boltzmannn']
  data_name = './data/%s' % sys.argv[1]
  with open(data_name, 'rb') as data:
    paths = cPickle.load(data)

  distribution_height_particle(0, paths, names)
  distribution_height_particle(1, paths, names)
  distribution_height_particle(2, paths, names)
