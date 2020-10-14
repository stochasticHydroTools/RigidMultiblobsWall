
import numpy as np
import scipy.sparse

# Try to import numba
try:
  from numba import njit, prange
except ImportError:
  print('numba not found')


@njit(parallel=True)
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
    for j in range(i+1, N):
      rx = r_vectors[j,0] - r_vectors[i,0]
      ry = r_vectors[j,1] - r_vectors[i,1]
      rz = r_vectors[j,2] - r_vectors[i,2]

      if L[0] > 0:
        rx = rx - int(rx / L[0] + 0.5 * (int(rx>0) - int(rx<0))) * L[0]
      if L[1] > 0:
        ry = ry - int(ry / L[1] + 0.5 * (int(ry>0) - int(ry<0))) * L[1]
      if L[2] > 0:
        rz = rz - int(rz / L[2] + 0.5 * (int(rz>0) - int(rz<0))) * L[2]

      # Compute force
      r_norm = np.sqrt(rx**2 + ry**2 + rz**2)
      f0 = -((eps / b) + (eps / r_norm)) * np.exp(-r_norm / b) / r_norm**2  
      force[i, 0] += f0 * rx
      force[i, 1] += f0 * ry
      force[i, 2] += f0 * rz

    for j in range(0, i):
      rx = r_vectors[j,0] - r_vectors[i,0]
      ry = r_vectors[j,1] - r_vectors[i,1]
      rz = r_vectors[j,2] - r_vectors[i,2]

      if L[0] > 0:
        rx = rx - int(rx / L[0] + 0.5 * (int(rx>0) - int(rx<0))) * L[0]
      if L[1] > 0:
        ry = ry - int(ry / L[1] + 0.5 * (int(ry>0) - int(ry<0))) * L[1]
      if L[2] > 0:
        rz = rz - int(rz / L[2] + 0.5 * (int(rz>0) - int(rz<0))) * L[2]

      # Compute force
      r_norm = np.sqrt(rx**2 + ry**2 + rz**2)
      f0 = -((eps / b) + (eps / r_norm)) * np.exp(-r_norm / b) / r_norm**2   
      force[i, 0] += f0 * rx
      force[i, 1] += f0 * ry
      force[i, 2] += f0 * rz

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
