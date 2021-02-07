''' Mobilities and other useful functions for sphere. '''
import numpy as np
import os
import sys
sys.path.append('..')

from mobility import mobility as mb
from general_application_utils import static_var

import selfMobilityHuang as Huang
import selfMobilityGoldman as Goldman
import selfMobilityFaucheux as Faucheux
import splines

# Make sure figures folder exists
if not os.path.isdir(os.path.join(os.getcwd(), 'figures')):
  os.mkdir(os.path.join(os.getcwd(), 'figures'))
# Make sure data folder exists
if not os.path.isdir(os.path.join(os.getcwd(), 'data')):
  os.mkdir(os.path.join(os.getcwd(), 'data'))
# Make sure logs folder exists
if not os.path.isdir(os.path.join(os.getcwd(), 'logs')):
  os.mkdir(os.path.join(os.getcwd(), 'logs'))


#Parameters
ETA = 1.0  # Viscosity.
A   = 0.5  # Radius of sphere.
M   = 0.5  # Mass*g of sphere.
H   = 3.5  # Initial Distance from Wall.
KT  = 0.2  # Temperature.
# Parameters for Yukawa potential
REPULSION_STRENGTH = 2.0
DEBYE_LENGTH = 0.5

def sphere_check_function(location, orientation):
  ''' Check that sphere is not overlapping the wall. '''
  if location[0][2] < A:
    return False
  else:
    return True
  

def null_torque_calculator(location, orientation):
  return [0., 0., 0.]


def sphere_force_calculator(location, orientation):
  gravity = -1*M
  h = location[0][2]
  repulsion = (REPULSION_STRENGTH*((h - A)/DEBYE_LENGTH + 1)*
               np.exp(-1.*(h - A)/DEBYE_LENGTH)/((h - A)**2))
  return [0., 0., gravity + repulsion]




def sphere_mobility(location, orientation):
  location = location[0]

  # Select sphere mobility close to a wall
  # Mobility based on the paper Swan and Brady, Physics of fluids 2007.
  fluid_mobility = mb.single_wall_self_mobility_with_rotation(location, ETA, A)

  # Mobility based on several theories and a cubic spline fit to
  # the mobility computed from a higer resolution method.
  #fluid_mobility = sphere_best_mobility_known(location, ETA, A)
  return fluid_mobility



@static_var('mX', {})
@static_var('mRotationRotationParallel', {})
@static_var('mRotationRotationPerpendicular', {})
@static_var('mRotationTranslationCoupling', {})
@static_var('mRotationRotationParallel_y2', {})
@static_var('mRotationRotationPerpendicular_y2', {})
@static_var('mRotationTranslationCoupling_y2', {})
def sphere_best_mobility_known(location, ETA, A):
  '''Best mobility known for a single sphere close to a wall. This function uses the mobilities
  translational-perpendicular: P. Huang and K. S. Breuer PRE 76, 046307 (2007).

  translational-parallel: for distances very close to the wall 
                          A. J. Goldman, R. G. Cox and H. Brenner, Chemical engineering science, 22, 637 (1967).
                          For other distances
                          L. P. Fauxcheux and A. J. Libchaber PRE 49, 5158 (1994).

  rotational-perpendicular:      Cubic spline fit to the mobility of a sphere discretize with 162 markers.
  rotational-parallel:           Cubic spline fit to the mobility of a sphere discretize with 162 markers.
  rotation-translation-coupling: Cubic spline fit to the mobility of a sphere discretize with 162 markers.
  '''
  
  # Init the function the first time
  try:
    # Check if the spline function has been called
    sphere_best_mobility_known.init += 1e-01
  except:
    # Exception to call the spline function only once per mobility component
    sphere_best_mobility_known.init = 1e-01 

    # Create variables to store mobilities:
    # distance to the wall
    sphere_best_mobility_known.mX = []

    # Mobilities
    sphere_best_mobility_known.mRotationRotationParallel = []
    sphere_best_mobility_known.mRotationRotationPerpendicular = []
    sphere_best_mobility_known.mRotationTranslationCoupling = []

    # Read 162-blobs sphere mobility.
    # rotational-roational mobility is normalize by (8*pi*ETA*A**3)
    # rotational-translation mobility is normalize by (6*pi*ETA*A**2)
    f = open('mobility.162-blob.dat', 'r')
    next(f)
    for line in f:
      data = line.split()

      # distance to the wall
      sphere_best_mobility_known.mX.append(float(data[0]))
      
      # Mobilities
      sphere_best_mobility_known.mRotationRotationParallel.append(float(data[3]))
      sphere_best_mobility_known.mRotationRotationPerpendicular.append(float(data[4]))
      sphere_best_mobility_known.mRotationTranslationCoupling.append(float(data[5]))
      
      
      
    # Create second derivative of mobility finctions 
    # Call spline function 
    sphere_best_mobility_known.mRotationRotationParallel_y2 = splines.spline(sphere_best_mobility_known.mX, 
                                                                             sphere_best_mobility_known.mRotationRotationParallel, 
                                                                             len(sphere_best_mobility_known.mX), 
                                                                             1e+30, 
                                                                             1e+30) 
    sphere_best_mobility_known.mRotationRotationPerpendicular_y2 = splines.spline(sphere_best_mobility_known.mX, 
                                                                                  sphere_best_mobility_known.mRotationRotationPerpendicular, 
                                                                                  len(sphere_best_mobility_known.mX), 
                                                                                  1e+30, 
                                                                                  1e+30) 
    sphere_best_mobility_known.mRotationTranslationCoupling_y2 = splines.spline(sphere_best_mobility_known.mX, 
                                                                                sphere_best_mobility_known.mRotationTranslationCoupling, 
                                                                                len(sphere_best_mobility_known.mX), 
                                                                                1e+30, 
                                                                                1e+30) 
    


  # Compute mobilities 
  # define threshold, at this distance the parallel mobilities
  # of Goldman and Feucheux cross
  threshold = 1.02979 * A
  
  # distance to the wall
  h = location[2]

  # dimensional factors
  factor_tt = 1.0 / (6.0*np.pi*ETA*A)
  factor_rr = 1.0 / (8.0*np.pi*ETA*A**3)
  factor_tr = 1.0 / (6.0*np.pi*ETA*A**2)

  fluid_mobility = np.zeros( [6, 6] )
  
  # translation-translation perpendicular to the wall
  fluid_mobility[2,2] = factor_tt * Huang.selfMobilityHuang(A,h)[1]

  # translation-translation parallel to the wall
  if(h < threshold):
    mobility = factor_tt * Goldman.selfMobilityGoldman(A, h)[0,0]
    fluid_mobility[0,0] = mobility
    fluid_mobility[1,1] = mobility
  else:
    mobility = factor_tt * Faucheux.selfMobilityFaucheux(A, h)
    fluid_mobility[0,0] = mobility
    fluid_mobility[1,1] = mobility

  # Rescale distance to the wall, splines are for a sphere of radius 1
  h = h / A

  # rotation-rotation parallel to the wall
  mobility = factor_rr * splines.splint(sphere_best_mobility_known.mX, 
                                        sphere_best_mobility_known.mRotationRotationParallel, 
                                        sphere_best_mobility_known.mRotationRotationParallel_y2, 
                                        len(sphere_best_mobility_known.mX), 
                                        h)
  fluid_mobility[3,3] = mobility
  fluid_mobility[4,4] = mobility

  # rotation-rotation perpendicular to the wall
  mobility = factor_rr * splines.splint(sphere_best_mobility_known.mX, 
                                        sphere_best_mobility_known.mRotationRotationPerpendicular, 
                                        sphere_best_mobility_known.mRotationRotationPerpendicular_y2, 
                                        len(sphere_best_mobility_known.mX), 
                                        h)
  fluid_mobility[5,5] = mobility

  # rotation-translation coupling
  mobility = factor_tr * splines.splint(sphere_best_mobility_known.mX, 
                                        sphere_best_mobility_known.mRotationTranslationCoupling, 
                                        sphere_best_mobility_known.mRotationTranslationCoupling_y2, 
                                        len(sphere_best_mobility_known.mX), 
                                        h)
  fluid_mobility[0,4] = mobility
  fluid_mobility[1,3] = mobility
  fluid_mobility[3,1] = mobility
  fluid_mobility[4,0] = mobility


  
  return fluid_mobility

