''' 
Evaluate how the CoH changes with the GB distribution in
3D.
'''

import numpy as np


import boomerang as bm
from quaternion_integrator.quaternion import Quaternion

def boomerang_coh_mobility(locations, orientations):
  ''' 
  Calculate the force and torque mobility for the boomerang.  Here
  the mobility is calculated using the CoH as the tracking point.
  '''
  r_vectors = bm.get_boomerang_r_vectors(locations[0], orientations[0])
  dist = 1.16 # From the PDF plots.
  dist = 0.7127 # From numerical calculation
  coh = (locations[0] + 
         np.cos(np.pi/4.)*(dist/1.575)*(r_vectors[0] - locations[0]) +
         np.sin(np.pi/4.)*(dist/1.575)*(r_vectors[6] - locations[0]))
  return bm.force_and_torque_boomerang_mobility(r_vectors, coh)


def find_boomerang_coh():
  '''
  Script to test different lengths for the boomerang and find the CoH
  numerically (in ~bulk). 
  This is just used to find the CoH, and should not be
  used in any of the actual calculations or scripts.
  
  Running this gives CoH = 0.70707 from tracking point along
  45 degree line.
  '''
  location = [0., 0., 10000.]
  orientation = Quaternion([1., 0., 0., 0.])
  min_norm = 9999999.
  min_dist = 0.
  for dist in np.linspace(0.0, 2.0, 1000):
    r_vectors = bm.get_boomerang_r_vectors(location, orientation)
    tracking_point = location + np.array([np.cos(np.pi/4.)*dist,
                                          np.sin(np.pi/4.)*dist,
                                          0.])
    mobility = bm.force_and_torque_boomerang_mobility(r_vectors, tracking_point)
    off_diag_norm = np.linalg.norm(mobility[0:3, 3:6])
    if (off_diag_norm < min_norm):
      min_norm = off_diag_norm
      min_dist = dist
      
  return min_dist



if __name__ == '__main__':
  # First, find CoH
  coh = find_boomerang_coh()
  print 'CoH distance from cross point is ', coh

  n_samples = 20000

  cross_norm = 0.
  coh_norm = 0.
  for k in range(n_samples):
    sample = bm.generate_boomerang_equilibrium_sample()
    mobility_cross = bm.boomerang_mobility([sample[0]], [sample[1]])
    cross_norm += np.linalg.norm(mobility_cross[0:3, 3:6])
    mobility_coh = boomerang_coh_mobility([sample[0]], [sample[1]])
    coh_norm += np.linalg.norm(mobility_coh[0:3, 3:6])
    
  coh_norm /= float(n_samples)
  cross_norm /= float(n_samples)

  print 'Ratio of CoH norm to cross norm is ', (coh_norm/cross_norm)

  
