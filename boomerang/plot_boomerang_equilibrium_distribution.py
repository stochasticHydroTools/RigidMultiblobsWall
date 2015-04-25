''' 
Script to generate (using MC) the boomerang equilibrium
distribution and plot it.  This is mostly useful for 
choosing parameters.
'''

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import numpy as np
import os
import sys
sys.path.append('..')
import time

import boomerang as bm
from config_local import DATA_DIR
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
  r_vectors = bm.get_boomerang_r_vectors_15(location, orientation)
  diff = r_vectors[0][2] - location[2]
  idx = int((diff + 1.575)/bucket_width)
  if idx < len(heights):
    heights[idx] += 1
  else:
    if idx > bin_arm_tip_difference.max_index:
      bin_arm_tip_difference.max_index = idx
      print 'New max index for arm tip difference, %s ' % idx


@static_var('max_index', 0)  
def bin_arm_tip(location, orientation, heights, bucket_width):
  ''' 
  Bin the height of the blob at the end of one of the arms.
  We would like this to stay under 2 for our highest gravity.
  '''
  r_vectors = bm.get_boomerang_r_vectors_15(location, orientation)
  idx = int((r_vectors[0][2])/bucket_width)
  if idx < len(heights):
    heights[idx] += 1
  else:
    if idx > bin_arm_tip.max_index:
      bin_arm_tip.max_index = idx
      print 'New max index for arm tip: %s ' % idx

@static_var('max_index', 0)  
def bin_max_blob(location, orientation, heights, bucket_width):
  ''' 
  Bin the height of the highest blob.
  '''
  r_vectors = bm.get_boomerang_r_vectors_15(location, orientation)
  idx = int(max([r_vectors[k][2] for k in range(len(r_vectors))])/bucket_width)
  if idx < len(heights):
    heights[idx] += 1
  else:
    if idx > bin_max_blob.max_index:
      bin_max_blob.max_index = idx
      print 'New max index for max_blob: %s ' % idx


@static_var('max_index', 0)  
def bin_min_blob(location, orientation, heights, bucket_width):
  ''' 
  Bin the height of the lowest blob.
  '''
  r_vectors = bm.get_boomerang_r_vectors_15(location, orientation)
  idx = int(min([r_vectors[k][2] for k in range(len(r_vectors))])/bucket_width)
  if idx < len(heights):
    heights[idx] += 1
  else:
    if idx > bin_min_blob.max_index:
      bin_min_blob.max_index = idx
      print 'New max index for min_blob: %s ' % idx


@static_var('max_index', 0)  
def bin_max_blob_diff(location, orientation, heights, bucket_width):
  ''' 
  Bin the difference in height between the highest and lowest blob.
  '''
  r_vectors = bm.get_boomerang_r_vectors_15(location, orientation)
  diff = (max([r_vectors[k][2] for k in range(len(r_vectors))]) - 
          min([r_vectors[k][2] for k in range(len(r_vectors))]))
  idx = int(diff/bucket_width)
  if idx < len(heights):
    heights[idx] += 1
  else:
    if idx > bin_max_blob_diff.max_index:
      bin_max_blob_diff.max_index = idx
      print 'New max index for max_blob_diff: %s ' % idx
  
  
if __name__ == '__main__':

  n_steps = int(sys.argv[1])
  
  # Plot different multiples of earth's gravity.
  factor_list = [1., 10., 20.]
  names_list = ['old', 'old', 'old']

  bin_width = 0.05
  cross_heights = np.zeros(int(11./bin_width))
  max_heights = np.zeros(int(11./bin_width))
  min_heights = np.zeros(int(11./bin_width))
  diff_heights = np.zeros(int(3.0/bin_width))

  # This is used for tip, max, and min as well.
  buckets_cross = np.arange(0, len(cross_heights))*bin_width + 0.5*bin_width
  # Buckets for max_diff in blobs.
  buckets_diff = np.arange(0, len(diff_heights))*bin_width + 0.5*bin_width


  data_name = 'BoomerangDistributionsData.txt'
  data_name = os.path.join('.', 'data', data_name)
  start_time = time.time()
  with open(data_name, 'w') as f_out:
    for k in range(len(factor_list)):
      factor = factor_list[k]
      data_name = names_list[k]
      f_out.write('Gravity = %s Earths Gravity\n' % factor)
      avg_height = 0.
      file_name = 'boomerang-samples-g-%s-%s.txt' % (factor, data_name)
      file_name = os.path.join(DATA_DIR, 'boomerang', file_name)
      with open(file_name, 'r') as f:
        line = f.readline()
        # Skip parameters. 
        while line != 'Location, Orientation:\n':
          line = f.readline()
        for k in range(n_steps):
          sample = bm.load_equilibrium_sample(f)
          bin_cross_height(sample[0], cross_heights, bin_width)
          bin_max_blob(sample[0], sample[1], max_heights, bin_width)
          bin_min_blob(sample[0], sample[1], min_heights, bin_width)
          bin_max_blob_diff(sample[0], sample[1], diff_heights, bin_width)
          avg_height += sample[0][2]
    
      avg_height /= float(n_steps)
      print 'average height: %s for factor: %s ' % (avg_height, factor)
      cross_heights = cross_heights/n_steps/bin_width
      max_heights = max_heights/n_steps/bin_width
      min_heights = min_heights/n_steps/bin_width
      diff_heights = diff_heights/n_steps/bin_width
    
      plt.figure(1)
      plt.plot(buckets_cross, cross_heights, label='Earth Mass * %s, %s' % (
          factor, data_name))
      f_out.write('Buckets: \n')
      f_out.write('%s \n' % buckets_cross)
      f_out.write('Cross Point Heights PDF:\n')
      f_out.write('%s \n' % cross_heights)    
    
      plt.figure(2)
      plt.plot(buckets_cross, max_heights,  label='Earth Mass * %s, %s' % (
          factor, data_name))
      f_out.write('Buckets: \n')
      f_out.write('%s \n' % buckets_cross)
      f_out.write('Max Blob Heights PDF:\n')
      f_out.write('%s \n' % max_heights)    
    
      plt.figure(3)
      plt.plot(buckets_cross, min_heights,  label='Earth Mass * %s, %s' % (
          factor, data_name))
      f_out.write('Buckets: \n')
      f_out.write('%s \n' % buckets_cross)
      f_out.write('Min Blob Heights PDF:\n')
      f_out.write('%s \n' % min_heights)    
    
      plt.figure(4)
      plt.plot(buckets_diff, diff_heights,  label='Earth Mass * %s, %s' % (
          factor, data_name))
      f_out.write('Buckets: \n')
      f_out.write('%s \n' % buckets_diff)
      f_out.write('Max Difference in Blob Heights PDF:\n')
      f_out.write('%s \n' % diff_heights)    
    
  print 'time elapsed is ', time.time() - start_time

  plt.figure(1)
  plt.title('PDF of Height of Cross Point of Boomerang')
  plt.legend(loc='best', prop={'size': 9})
  plt.xlim([0.,6.])
  plt.savefig('./figures/BoomerangCrossHeightDistribution.pdf')

  plt.figure(2)
  plt.title('PDF of Max Blob of Boomerang')
  plt.legend(loc='best', prop={'size': 9})
  plt.xlim([0.,6.])
  plt.savefig('./figures/BoomerangMaxBlobDistribution.pdf')
    
  plt.figure(3)
  plt.title('PDF of Min Blob of Boomerang')
  plt.legend(loc='best', prop={'size': 9})
  plt.xlim([0.,6.])
  plt.savefig('./figures/BoomerangMinBlobDistribution.pdf')
  
  plt.figure(4)
  plt.title('PDF of Diff btw Max and Min Blob')
  plt.legend(loc='best', prop={'size': 9})
  plt.savefig('./figures/BoomerangBlobDiffDistribution.pdf')
  
  
  
