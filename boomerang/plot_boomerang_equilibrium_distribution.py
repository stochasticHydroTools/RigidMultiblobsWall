''' 
Script to generate (using MC) the boomerang equilibrium
distribution and plot it.  This is mostly useful for 
choosing parameters.
'''

import numpy as np

import boomerang as bm
from utils import static_var


def bin_cross_height(location, heights, bucket_width):
  ''' 
  Bin the height of the cross point given location 
  (which is the cross point).
  Assumes uniform buckets starting at 0
  '''
  idx = int(location[2]/bucket_width)
  if idx > 
  heights[idx] += 1
  
  

