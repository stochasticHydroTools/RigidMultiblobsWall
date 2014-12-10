''' 
Script to check order of accuracy of a scheme by looking
at the error in height distribution.
'''


def check_height_order(heights_list, buckets, names, dts, order):
  ''' 
  Plot just the discrepency between each scheme and the equilibrium, which is
  assumed to be the last entry in heights.
  '''
  for particle in range(3):
    for j in range(len(heights_list[0]) - 1):
      fig = pyplot.figure()
      for k in range(len(heights)):
        scale_factor = (dts[0]/dts[k])**order
        pyplot.plot(buckets, scale_factor*(heights_list[k][j][particle] -
                                           heights[k][-1][particle]), 
                    label = names[k] + '%s' % dts[k])

      pyplot.savefig('./plots/HeightError-Scheme-%s-Particle-%s' % (names[k], particle))


if __name__  == '__main__':
  
