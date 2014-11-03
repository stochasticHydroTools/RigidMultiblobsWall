''' Script to test a tetrahedron near a wall '''

import numpy as np
from quaternion import Quaternion
from quaternion_integrator import QuaternionIntegrator

ETA = 1.0   # Fluid viscosity.
A = 0.1     # Particle Radius.
H = 10.     # Distance to wall.

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
  


def ImageSingularStokeslet(quaternion, r_vectors):
  ''' Calculate the image system for the singular stokeslet (M above).'''
  mobility = np.array([zeros(9) for _ in range(9)])
  # Loop through particle interactions
  for j in range(3):  
    for k in range(3):
      if j != k:  #  do particle interaction
        r_particles = r_vectors[j] - r_vectors[k]
        r_norm = np.linalg.norm(r_particles)
        r_reflect = r_vectors[j] - (r_vectors[k] - 2.*np.array([0., 0., H])
                                       - 2.*r_vectors[k][2])
        r_ref_norm = np.linalg.norm(r_reflect)
        # Loop through components.
        for l in range(3):
          for m in range(3):
            # Two stokeslets, one with negative force at image.
            mobility[i*3 + l][j*3 + m] = (
              (l == m)*1./r_norm + r_particles[l]*r_particles[m]/(r_norm**3) -
              (l == m)*1./r_ref_norm + r_reflect[l]*r_reflect[m]/(r_ref_norm**3))
            # Add Doublet.
            
            
              
def StokesDoublet(direction, strength, r):
  ''' Calculate stokes doublet from direction, strength, and r. '''
  doublet = np.dot(direction, strength)*r + np.dot(strength, r)*direction

  
def CalculateR(quaternion, r_vectors):
  ''' Calculate R, 9 by 3 matrix of cross products for r_i. '''
  
  # Create the 9 x 3 matrix.  Each 3x3 block is the matrix for a cross
  # product with one of the r_vectors.
  return np.array([
      #Block 1, r_1 cross
      [0.0, -1.*r_vectors[0][2], r_vectors[0][1]],
      [r_vectors[0][2], 0.0, -1.*r_vectors[0][0]],
      [-1.*r_vectors[0][1], *r_vectors[0][0],0.0],
      # Block 2, r_2 cross
      [0.0, -1.*r_vectors[1][2], r_vectors[1][1]],
      [r_vectors[1][2], 0.0, -1.*r_vectors[1][0]],
      [-1.*r_vectors[1][1], *r_vectors[1][0],0.0],
      # Block 3, r_3 cross
      [0.0, -1.*r_vectors[2][2], r_vectors[2][1]],
      [r_vectors[2][2], 0.0, -1.*r_vectors[2][0]],
      [-1.*r_vectors[2][1], *r_vectors[2][0],0.0],
      ])


def GetRVectors(quaternion):
  ''' Calculate r_i from a given quaternion. 
  The initial configuration is hard coded here but can be changed by
  considering an initial quaternion not equal to the identity rotation.
  initial configuration (top down view, the top vertex is fixed at the origin):

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
  
  return [r1, r2, r3]

  
  
  

  
  
  
  


  

