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

def MatrixToQuaterion():
  ''' Convert rotation matrix to a quaternion '''
  



if __name__ == "__main__":
  
  

  
