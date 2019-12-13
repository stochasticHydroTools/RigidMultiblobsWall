from __future__ import division, print_function
import numpy as np

# Try to import numba
try:
  from numba import njit, prange
  import numba as nb
except ImportError:
  print('numba not found')


@njit(parallel=True, fastmath=True)
def blob_blob_force_numba(r_vectors, L, eps, b, a):
  '''
  This function compute the force between two blobs
  with vector between blob centers r.

  In this example the force is derived from the potential
  
  U(r) = U0 + U0 * (2*a-r)/b   if z<2*a
  U(r) = U0 * exp(-(r-2*a)/b)  iz z>=2*a
  
  with
  eps = potential strength
  r_norm = distance between blobs
  b = Debye length
  a = blob_radius
  '''

  N = r_vectors.size // 3
  r_vectors = r_vectors.reshape((N, 3))
  force = np.zeros((N, 3))

  for i in prange(N):
    for j in range(N):
      if i == j:
        continue

      dr = np.zeros(3)
      for k in range(3):
        dr[k] = r_vectors[j,k] - r_vectors[i,k]
        if L[k] > 0:
          dr[k] -= int(dr[k] / L[k] + 0.5 * (int(dr[k]>0) - int(dr[k]<0))) * L[k]

      # Compute force
      r_norm = np.sqrt(dr[0]*dr[0] + dr[1]*dr[1] + dr[2]*dr[2])
      if r_norm > 2*a:
        f0 = -((eps / b) * np.exp(-(r_norm - 2.0*a) / b) / r_norm)
      else:
        f0 = -((eps / b) / np.maximum(r_norm, 1e-25))

      for k in range(3):
        force[i,k] += f0*dr[k]

  return force


def calc_blob_blob_forces_numba(r_vectors, *args, **kwargs):
  '''
  This function computes the blob-blob forces and returns
  an array with shape (Nblobs, 3).
  '''
    
  # Get parameters from arguments
  L = kwargs.get('periodic_length')
  eps = kwargs.get('repulsion_strength')
  b = kwargs.get('debye_length')
  a = kwargs.get('blob_radius')

  force_blobs = blob_blob_force_numba(r_vectors, L, eps, b, a)

  return force_blobs
