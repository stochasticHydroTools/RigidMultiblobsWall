''' Script to test a tetrahedron near a wall '''

import numpy as np
from quaternion import Quaternion
from quaternion_integrator import QuaternionIntegrator

ETA = 1.0   # Fluid viscosity.
A = 0.1     # Particle Radius.

def TetrahedronMobility(self, position):
  '''
  Calculate the mobility, torque -> angular velocity, at position 
  In this case, position is length 1, as there is just 1 quaternion.
  The mobility is equal to R M^-1 R^t where R is 3N x 3 (9 x 3)
  Rx = r cross x
  r is the distance from the fixed vertex of the tetrahedron to
  each other vertex (a length 3N vector).
  M (3N x 3N) is the singular image stokeslet for a point force near a wall, but
  we've replaced the diagonal piece by 1/(6 pi eta a).
  '''
  


def ImageSingularStokeslet(quaternion):
  ''' Calculate the image system for the singular stokeslet (M above).'''
  diag_entry = 1./(6*np.pi*ETA*A)
  diag_block = np.array([[diag_entry, 0., 0.], 
                         [0., diag_entry, 0.],
                         [0., 0., diag_entry]])

  
def CalculateR(quaternion):
  ''' Calculate R, 3N by N matrix of cross products for r_i. '''
  

def GetRVectors(quaternion):
  ''' Calculate r_i from a given quaternion. 
  The initial configuration is hard coded here but can be changed by
  considering an initial quaternion not equal to the identity rotation.
  initial configuration:
                         O r_1 = (0, 2/sqrt(3), -(2 sqrt(2))/3)
                        / \
                       /   \
                      /     \
                     /   O(0, 0, 0)
                    /          \
                   /            \
               -> O--------------O  r_3 = (1, -1/sqrt(3),-(2 sqrt(2))/3)
             /
           r_2 = (-1, -1/sqrt(3),-(2 sqrt(2))/3)

  Each side of the tetrahedron has length 2.
  '''
  initial_r1 = np.array([0., 2./np.sqrt(3.), -2.*np.sqrt(2.)/np.sqrt(3.)])
  initial_r2 = np.array([-1., -1./np.sqrt(3.), -2.*np.sqrt(2.)/np.sqrt(3.)])
  initial_r3 = np.array([1., -1./np.sqrt(3.), -2.*np.sqrt(2.)/np.sqrt(3.)])
  
  rotation_matrix = quaternion.RotationMatrix()

  r1 = np.dot(rotation_matrix, initial_r1)
  r2 = np.dot(rotation_matrix, initial_r2)
  r3 = np.dot(rotation_matrix, initial_r3)

  
  
  

  
  
  
  


  

