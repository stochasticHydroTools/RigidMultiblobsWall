'''
Class to compute the stochastic forcing (sqrt(2*k_B*T*dt)*M^{1/2}*z) with several algorithms.
'''

import numpy as np

import sys
sys.path.append('..')
import mobility as mob
from boomerang import boomerang as bm
from quaternion_integrator import quaternion as q


def stochastic_forcing_sdv(mobility, factor):
  '''
  Compute the stochastic forcing (factor * M^{1/2} * z) using
  SVD decomposition. This functions is more expensive that
  using a Cholesky factorization but it works for non-positive
  definite matrix (e.g. mobility of a 1D rod).
  
  Input:
  Mobility = the mobility matrix. You can pass it like a list of lists or
             a list of numpy arrays.
  factor = the prefactor, in general something like sqrt(2*k_B*T*dt)
  
  Output:
  stochastic_forcing = (factor * M^{1/2} * z)
  '''
  
  # Compute eigenvalues and eigenvectors 
  eig_values, eig_vectors = np.linalg.eigh(mobility)

  # Compute the square root of positive eigenvalues set to zero otherwise
  eig_values_sqrt_noise = np.array([np.sqrt(x) if x > 0 else 0 for x in eig_values])

  # Multiply by random vector with zero mean and unit variance
  eig_values_sqrt_noise *= np.random.normal(0.0, 1.0, len(mobility))
 
  # Compute stochastic forcing
  stochastic_forcing = factor * np.dot( eig_vectors, eig_values_sqrt_noise)
      
  return stochastic_forcing



def stochastic_forcing_cholesky(mobility, factor):
  '''
  Compute the stochastic forcing (factor * M^{1/2} * z) using
  Cholesky decomposition. 
  
  Input:
  Mobility = the positive-definite mobility matrix. You can 
             pass it like a list of lists or a list of numpy arrays.
  factor = the prefactor, in general something like sqrt(2*k_B*T*dt)
  
  Output:
  stochastic_forcing = (factor * M^{1/2} * z)
  '''

  # Compute Cholesky factorization
  mobility_half = np.linalg.cholesky(mobility)

  # Compute random vectore with mean zero and variance 1
  noise = np.random.normal(0.0, 1.0, len(mobility))

  # Compute stochastic forcing
  stochastic_forcing = factor * np.dot(mobility_half, noise)
      
  return stochastic_forcing


  return


if __name__ == '__main__':

  # Define objects
  location = [0., 0., 100000.]
  theta = np.random.normal(0., 1., 4)
  #orientation = Quaternion(theta/np.linalg.norm(theta))
  orientation = q.Quaternion([1., 0., 0., 0.])
  r_vectors = bm.get_boomerang_r_vectors_15(location, orientation)
  mobility = bm.force_and_torque_boomerang_mobility(r_vectors, location)
  
  mobility_np = []
  for i in range(len(mobility)):
    m = np.zeros( len(mobility) )
    for j in range(len(mobility)):
      m[j] = mobility[i][j]
    mobility_np.append(m)


    
  # Compute noise
  noise = stochastic_forcing_sdv(mobility_np, 1.0)
  print noise, '\n'

  noise = stochastic_forcing_cholesky(mobility_np, 1.0)
  print noise, '\n'



  print '# End'
