'''
Module to compute the stochastic forcing (sqrt(2*k_B*T*dt)*M^{1/2}*z) with several algorithms.
'''
import numpy as np

def stochastic_forcing_sdv(mobility, factor):
  '''
  Compute the stochastic forcing (factor * M^{1/2} * z) using
  SVD decomposition. This functions is more expensive that
  using a Cholesky factorization but it works for non-positive
  definite matrix (e.g. mobility of a 1D rod).
  
  Input:
  mobility = the mobility matrix. You can pass it like a list of lists or
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
  mobility = the positive-definite mobility matrix. You can 
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


def stochastic_forcing_lanczos(mobility_mult, factor, r):
  '''
  Compute the stochastic forcing (factor * M^{1/2} * z) using
  the Lanczos algorithm, see Krylov subspace methods for 
  computing hydrodynamic interactions in Brownian dynamics simulations, 
  T. Ando et al. The Journal of Chemical Physics 137, 064106 (2012) 
  doi: 10.1063/1.4742347 and subsequent papers by Yousef Saad.
  
  Input:
  mobility_mult = function that computes a matrix vector product 
                  with the mobility matrix.
  factor = the prefactor, in general something like sqrt(2*k_B*T*dt)
  
  Output:
  stochastic_forcing = (factor * M^{1/2} * z)
  '''


  return 0


