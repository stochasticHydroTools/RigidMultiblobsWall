'''
Module to compute the stochastic forcing (sqrt(2*k_B*T*dt)*M^{1/2}*z) with several algorithms.
'''
import numpy as np

def stochastic_forcing_sdv(mobility, factor, z = None):
  '''
  Compute the stochastic forcing (factor * M^{1/2} * z) using
  SVD decomposition. This functions is more expensive that
  using a Cholesky factorization but it works for non-positive
  definite matrix (e.g. mobility of a 1D rod).
  
  Input:
  mobility = the mobility matrix. You can pass it like a list of lists or
             a list of numpy arrays.
  factor = the prefactor, in general something like sqrt(2*k_B*T*dt)
  z = (Optional) the random vector z
  
  Output:
  stochastic_forcing = (factor * M^{1/2} * z)
  '''
  
  # Compute eigenvalues and eigenvectors 
  eig_values, eig_vectors = np.linalg.eigh(mobility)

  # Compute the square root of positive eigenvalues and set to zero otherwise
  eig_values_sqrt = np.array([np.sqrt(x) if x > 0 else 0 for x in eig_values])

  # Multiply by random vector with zero mean and unit variance
  if z is None:
    eig_values_sqrt *= np.random.normal(0.0, 1.0, len(mobility))
  else:
     eig_values_sqrt *= z

  # Compute stochastic forcing
  stochastic_forcing = factor * np.dot(eig_vectors, eig_values_sqrt)
      
  return stochastic_forcing


def stochastic_forcing_cholesky(mobility, factor, z = None):
  '''
  Compute the stochastic forcing (factor * M^{1/2} * z) using
  Cholesky decomposition. 
  
  Input:
  mobility = the positive-definite mobility matrix. You can 
             pass it like a list of lists or a list of numpy arrays.
  factor = the prefactor, in general something like sqrt(2*k_B*T*dt)
  z = (Optional) the random vector z
  
  Output:
  stochastic_forcing = (factor * M^{1/2} * z)
  '''

  # Compute Cholesky factorization
  mobility_half = np.linalg.cholesky(mobility)

  # Compute random vectore with mean zero and variance 1
  if z is None:
    z = np.random.normal(0.0, 1.0, len(mobility))

  # Compute stochastic forcing
  stochastic_forcing = factor * np.dot(mobility_half, z)
      
  return stochastic_forcing


def stochastic_forcing_lanczos(factor = 1.0, 
                               tolerance = 1e-06, 
                               max_iter = 1000, 
                               name = '',
                               r_vectors = None, 
                               eta = None, 
                               radius = None, 
                               mobility = None, 
                               mobility_mult = None,
                               z = None):
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
  r_vectors = 
  eta = 
  a =
  max_iter = 
  
  Output:
  stochastic_forcing = (factor * M^{1/2} * z)
  '''

  # Define array dimension (3 * number of blobs)
  if mobility is None:
    dim = len(r_vectors) * 3
  else:
    dim = len(mobility)

  # Create matrix v (initial column is random)
  if z is None:
    v = np.random.randn(1, dim)
  else:
    v = np.reshape(z, (1, dim))
  # print 'v', v[0]

  # Normalize v
  v_norm = np.linalg.norm(v[0])
  v[0] /= v_norm 

  print v
  print v[0]

  # Create list for the data of the symmetric tridiagonal matrix h 
  h_sup = []
  h_diag = []

  # Create vectors noise
  noise = np.zeros(dim) 
  noise_old = np.zeros(dim) 

  # Iterate until convergence or max_iter
  for i in range(max_iter):
    print '\n\n', i

    # w = mobility * v[i]
    if mobility is None:
      w = mobility_mult(r_vectors, np.reshape(v[i], (dim/3, 3)), eta, radius)
    else:
      w = np.dot(mobility, v[i])

    # w = w - h[i-1, i] * v[i-1]
    # print '\n\n\n', 'v', i, v[i]
    # print i, w
    if i > 0:
      w = w - h_sup[i-1] * v[i-1]
      # print i, w
      # print i, h[i-1]

    # h(i, i) = <w, v[i]> 
    h_diag.append( np.dot(w, v[i]) ) 
    # print i, h_diag

    # w = w - h(i, i)*v(i)
    w = w - h_diag[i] * v[i]
    # print 'w', w

    # h(i+1, i) = h(i, i+1) = < w, w>
    h_sup.append( np.linalg.norm(w) )

    # w = w/normw;
    if h_sup[i] > 0:
      w /= h_sup[i]
    else:
      w[0] = 1.0;
    # print 'w', w
    
    # Build tridiagonal matrix h
    h = h_diag * np.eye(len(h_diag)) + h_sup * np.eye(len(h_sup), k=-1) + (h_sup * np.eye(len(h_sup), k=-1)).T
    # print 'h', h

    # Compute eigenvalues and eigenvectors of h
    # IMPORTANT: this is NOT optimized for tridiagonal matrices
    eig_values, eig_vectors = np.linalg.eigh(h)
    # eig_values = eig_values.T
    # eig_vectors = eig_vectors.T
    # print 'eig_values', eig_values
    # print 'eig_vectors', eig_vectors
   
    # Compute the square root of positive eigenvalues set to zero otherwise
    eig_values_sqrt = np.array([np.sqrt(x) if x > 0 else 0 for x in eig_values])
    # print 'eig_values_sqrt', eig_values_sqrt
    
    h_half = np.linalg.cholesky(h)

    # Create vector e_1
    e_1 = np.zeros(len(eig_values))
    e_1[0] = 1.0
    # print 'e_1', e_1
    # print 'V*e_1', np.dot(e_1, v)
    
    # Compute noise approximation
    if 0:
      noise = factor * eig_values_sqrt * e_1
      noise = np.dot(eig_vectors, noise)
      noise = np.dot(noise, v)
    elif 1:
      noise = np.dot(eig_vectors.T, e_1)
      noise = v_norm * factor * eig_values_sqrt * noise
      noise = np.dot(eig_vectors, noise)
      noise = np.dot(v.T, noise)
    else:
      noise = factor * np.dot(v.T, np.dot(h_half, e_1))

    print 'noise', noise
    # print 'v', (v.T).shape
    # print 'w', w

    # v(i+1) = w
    # v.append( w )
    v = np.concatenate([v, [w]])
    # print 'v', v

    if i > 0:
      noise_old_norm = np.linalg.norm(noise_old)
      diff_norm = np.linalg.norm(noise - noise_old)
    
      # (Optional) Save residual
      if i == 1 and name != '':
        with open(name, 'w') as f:
          data = '1\n' + str(diff_norm / noise_old_norm) + '\n'
          f.write(data)
      elif i > 1 and name != '':
        with open(name, 'a') as f:
          data = str(diff_norm / noise_old_norm) + '\n'
          f.write(data)

      # Check convergence and return if error < tolerance
      if diff_norm / noise_old_norm < tolerance:
        return noise
          
    # Save noise to check convergence in the next iteration
    noise_old = np.copy(noise)

  # Return un-converged noise
  return noise





def return_eta(get_eta, eta):

  return get_eta(eta)
  return 0


