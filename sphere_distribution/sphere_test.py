'''
Small test to verify that a uniformly distributed rotation
corresponds to a quaternion uniformly distributed on the surface
of the 3-Sphere.
'''

import numpy as np

def GenerateRandomRotationMatrix():
  ''' 
  Generate a rotation matrix that represents a uniformly
  random distributed rotation.
  '''
  A = np.matrix([np.random.normal(0., 1., 3) for _ in range(3)])
  P = np.linalg.cholesky(A.T*A)
  R = A*np.linalg.inv(P)

  return R

def MatrixToQuaterion(R):
  ''' Convert rotation matrix to a quaternion. '''
  
  
if __name__ == "__main__":
  
  

  
