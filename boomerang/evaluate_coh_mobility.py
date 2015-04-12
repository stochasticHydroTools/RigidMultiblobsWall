''' 
Evaluate how the CoH changes with the GB distribution in
3D.
'''
import matplotlib
matplotlib.use('Agg')
import numpy as np
from matplotlib import pyplot as plt
import os

import boomerang as bm
from quaternion_integrator.quaternion import Quaternion

def boomerang_coh_mobility(locations, orientations):
  ''' 
  Calculate the force and torque mobility for the boomerang.  Here
  the mobility is calculated using the CoH as the tracking point.
  '''
  r_vectors = bm.get_boomerang_r_vectors(locations[0], orientations[0])
  dist = 1.16 # From the PDF plots, but maybe this is to the Corner?
  dist = 0.80943
  coh = (locations[0] + 
         np.cos(np.pi/4.)*(dist/1.575)*(r_vectors[0] - locations[0]) +
         np.sin(np.pi/4.)*(dist/1.575)*(r_vectors[6] - locations[0]))
  return bm.force_and_torque_boomerang_mobility(r_vectors, coh)


def newtons_method(f, x):
  ''' 
  minimize function f by newtons method with
  initial guess x.
  '''  
  tol = 1e-6
  delta = 1e-7
  magnitude = 1
  while magnitude > tol:
    df_dx = (f(x + delta) - f(x-delta))/(2.*delta)
    df_2 = (f(x + delta) + f(x-delta) - 2*f(x))/(delta**2)
    x_new = x - df_dx/df_2
    magnitude = abs(x - x_new)
    x = x_new

  print "norm at x is ", f(x)
  print 'cross norm', f(0.)
  return x
  

def find_boomerang_coh():
  '''
  Script to test different lengths for the boomerang and find the CoH
  numerically (in ~bulk). 
  This is just used to find the CoH, and should not be
  used in any of the actual calculations or scripts.
  
  Running this gives CoH = 0.809431 from tracking point along
  45 degree line.
  '''
  location = [0., 0., 90000000.]
  orientation = Quaternion([1., 0., 0., 0.])
  def coupling_function(dist):
    ''' 
    Calculate norm of coupling at distance dist 
    from cross point.
    '''
    location = [0., 0., 90000000.]
    orientation = Quaternion([1., 0., 0., 0.])
    r_vectors = bm.get_boomerang_r_vectors(location, orientation)
    tracking_point = location + np.array([np.cos(np.pi/4.)*dist,
                                          np.sin(np.pi/4.)*dist,
                                          0.])
    mobility = bm.force_and_torque_boomerang_mobility(r_vectors, tracking_point)
    coupling_norm = np.linalg.norm(mobility[0:2, 5:6])**2
    return coupling_norm

  coh_dist = newtons_method(coupling_function, 0.0)

  print "coh_dist is ", coh_dist
  # Compare to theory from Bernal and De La Torre 'Transport Properties and Hydrodynamic Centers
  # of Rigid Macromolecules with Arbitrary Shapes'
  r_vectors = bm.get_boomerang_r_vectors(location, orientation)
  # tracking_point = location + np.array([np.cos(np.pi/4.)*coh_dist,
  #                                       np.sin(np.pi/4.)*coh_dist,
  #                                       0.])
  mobility = bm.force_and_torque_boomerang_mobility(r_vectors, location)
  resistance = np.linalg.inv(mobility)

  # 2D system. Want coupling between Force_x, Force_y and Torque_z to be 0.
  # Solve the relevant part of hte system in equation 2 of that paper.
  rhs = np.array([resistance[5, 0], resistance[5, 1]])
  coupling_matrix = np.array([[-1.*resistance[1, 0], resistance[0, 0]],
                              [-1.*resistance[1, 1], resistance[0, 1]]])
  r_xy = np.dot(np.linalg.inv(coupling_matrix),
                rhs)
  print 'r_xy is ', r_xy
  print 'norm of r_xy is ', np.linalg.norm(r_xy)

  return coh_dist


def calculate_coupling_norm(distance, n_samples, gfactor):
  '''
  Calculate the norm of the coupling tensor averaged over the GB
  distribution at gfactor times "earth gravity."
  '''
  file_name = 'boomerang-samples-g-%s.txt' % gfactor
  file_name = os.path.join('.', 'data', file_name)
  print "file name is ", file_name
  with open(file_name, 'r') as f:
    line = f.readline()
    # Print parameters and skip past them.
    while line != 'Location, Orientation:\n':
      print line
      line = f.readline()

    coupling_norm = 0.
    for k in range(n_samples):
      sample = bm.load_equilibrium_sample(f)
      rotation_matrix = sample[1].rotation_matrix()
      tracking_point = np.dot(rotation_matrix,
                              np.array([distance/np.sqrt(2.),
                                        distance/np.sqrt(2), 0.]))
      mobility = bm.boomerang_mobility_at_arbitrary_point(
        [sample[0]], [sample[1]], tracking_point)
      coupling_norm += np.linalg.norm(mobility[0:2, 5:6])
      
  coupling_norm /= float(n_samples)
  return coupling_norm


if __name__ == '__main__':
  # First, find CoH
  coh = find_boomerang_coh()
  print 'CoH distance from cross point is ', coh

  n_samples = 400

  cross_norm = 0.
  coh_norm = 0.
  for k in range(n_samples):
    sample = bm.generate_boomerang_equilibrium_sample()
    mobility_cross = bm.boomerang_mobility([sample[0]], [sample[1]])
    cross_norm += np.linalg.norm(mobility_cross[0:2, 5:6])
    mobility_coh = boomerang_coh_mobility([sample[0]], [sample[1]])
    coh_norm += np.linalg.norm(mobility_coh[0:2, 5:6])
    
  coh_norm /= float(n_samples)
  cross_norm /= float(n_samples)

  print 'Ratio of CoH norm to cross norm is ', (coh_norm/cross_norm)

  # plot distance v. coupling norm for various gravities.
  gfactors = [1., 5., 10.]
  distances = np.linspace(0., 2., 30)
  cross_point_norm = calculate_coupling_norm(0., 500, 1.0)
  for g in gfactors:
    coupling_norms = []
    for distance in distances:
      coupling_norms.append(calculate_coupling_norm(distance, n_samples, g))
      

    plt.plot(distances, coupling_norms/cross_point_norm, label='gravity=%s' % g)
    
  plt.title('XY velocity to Z torque Coupling norms')
  plt.xlabel('Distance from cross point')
  plt.ylabel('Norm of coupling')
  plt.legend(loc='best', prop={'size': 9})
  plt.savefig(os.path.join('.', 'figures', 'BoomerangCouplingNorm.pdf'))
