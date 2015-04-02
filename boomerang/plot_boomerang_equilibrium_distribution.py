''' 
Script to generate (using MC) the boomerang equilibrium
distribution and plot it.  This is mostly useful for 
choosing parameters.
'''


import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import numpy as np
import sys
sys.path.append('..')

import boomerang as bm
from utils import static_var


@static_var('max_index', 0)
def bin_cross_height(location, heights, bucket_width):
  ''' 
  Bin the height of the cross point given location 
  (which is the cross point).
  Assumes uniform buckets starting at 0
  '''
  idx = int(location[2]/bucket_width)
  if idx < len(heights):
    heights[idx] += 1
  else:
    if idx > bin_cross_height.max_index:
      bin_cross_height.max_index = idx
      print "New maximum Index  %d is beyond histogram length " % idx


@static_var('max_index', 0)  
def bin_arm_tip_difference(location, orientation, heights, bucket_width):
  ''' 
  Bin the difference in height of the blob at the end of one of the arms
  compared to the cross point blob.  This way we can examine how
  flat the boomerang stays as it diffuses.
  '''
  r_vectors = bm.get_boomerang_r_vectors(location, orientation)
  diff = r_vectors[6][2] - location[2]
  idx = int((diff + 1.575)/bucket_width)
  if idx < len(heights):
    heights[idx] += 1
  else:
    if idx > bin_arm_tip_difference.max_index:
      bin_arm_tip_difference.max_index = idx
      print 'New max index for arm tip difference, %s ' % idx
  
  
if __name__ == '__main__':

  n_steps = int(sys.argv[1])

  bin_width = 0.1
  cross_heights = np.zeros(int(10/bin_width))
  diff_heights = np.zeros(int(3.15/bin_width))
  
  for k in range(n_steps):
    sample = bm.generate_boomerang_equilibrium_sample()
    bin_cross_height(sample[0], cross_heights, bin_width)
    bin_arm_tip_difference(sample[0], sample[1], diff_heights, bin_width)

  cross_heights = cross_heights/n_steps/bin_width
  diff_heights = diff_heights/n_steps/bin_width

  buckets_cross = np.arange(0, len(cross_heights))*bin_width
  buckets_diff = np.arange(0, len(diff_heights))*bin_width - 1.575
  
  plt.figure(1)
  plt.plot(buckets_cross, cross_heights)
  plt.savefig('./figures/BoomerangCrossHeightDistribution.pdf')

  plt.figure(2)
  plt.plot(buckets_diff, diff_heights)
  plt.savefig('./figures/BoomerangTipDifferenceDistribution.pdf')

    
    
  
  
  
