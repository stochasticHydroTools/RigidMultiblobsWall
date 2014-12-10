''' 
Script to check order of accuracy of a scheme by looking
at the error in height distribution.
'''
import cPickle
from matplotlib import pyplot

def check_height_order(heights_list, buckets, names, dts, order):
  ''' 
  Plot just the discrepency between each scheme and the equilibrium, which is
  assumed to be the last entry in heights.
  '''
  # Just look at the heaviest particle for now.
  particle = 2
  for j in range(len(heights_list[0]) - 1):
    # loop through schemes, indexed by j
    fig = pyplot.figure()
    for k in range(len(heights_list)):
      #loop through dts, indexed by k
      scale_factor = (dts[0]/dts[k])**order
      pyplot.plot(buckets, scale_factor*(heights_list[k][j][particle] -
                                         heights_list[k][-1][particle]), 
                  label = names[j] + ', dt=%s' % dts[k])
    pyplot.title('%s scheme, order %s test' % (names[j], order))
    pyplot.xlabel('Height')
    pyplot.ylabel('Error in height distribution')
    pyplot.legend(loc = 'best', prop={'size': 9})
    pyplot.savefig('./plots/HeightError-Scheme-%s-Particle-%s.pdf' %
                   (names[j], particle))


if __name__  == '__main__':
  #  Grab the data from a few runs with different dts, and
  #  Check their order.
  data_files = ['tetrahedron-dt-4-N-80000.pkl', 'tetrahedron-dt-2-N-80000.pkl',
                'tetrahedron-dt-1-N-80000.pkl']
  dts = [4., 2., 1.]

  heights_list = []
  for data_file in data_files:
    with open('./data/' + data_file, 'rb') as f:
      heights_data = cPickle.load(f)
      heights_list.append(heights_data['heights'])
      
  buckets = heights_data['buckets']
  names = heights_data['names']
  print 'length of heights list is ', len(heights_list)
  check_height_order(heights_list, buckets, names, dts, 1.)
      
  
