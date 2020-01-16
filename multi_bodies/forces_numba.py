from __future__ import division, print_function
import numpy as np
import scipy.spatial as scsp

# Try to import numba
try:
  from numba import njit, prange
except ImportError:
  print('numba not found')
import general_application_utils as utils

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


@njit(parallel=True, fastmath=True)
def blob_blob_force_tree_numba(r_vectors, L, eps, b, a, list_of_neighbors, offsets):
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

  # Copy arrays
  rx_vec = np.copy(r_vectors[:,0])
  ry_vec = np.copy(r_vectors[:,1])
  rz_vec = np.copy(r_vectors[:,2])

  for i in prange(N):
    for k in range(offsets[i+1] - offsets[i]):
      j = list_of_neighbors[offsets[i] + k]
      if i == j:
        continue
      rx = rx_vec[j] - rx_vec[i]
      ry = ry_vec[j] - ry_vec[i]
      rz = rz_vec[j] - rz_vec[i]

      # Compute force
      r_norm = np.sqrt(rx*rx + ry*ry + rz*rz)
      if r_norm > 2*a:
        f0 = -((eps / b) * np.exp(-(r_norm - 2.0*a) / b) / r_norm)
      else:
        f0 = -((eps / b) / np.maximum(r_norm, 1e-25))
      force[i, 0] += f0 * rx
      force[i, 1] += f0 * ry
      force[i, 2] += f0 * rz
  return force


@utils.static_var('r_vectors_old', [])
@utils.static_var('list_of_neighbors', [])
@utils.static_var('offsets', [])
def calc_blob_blob_forces_tree_numba(r_vectors, *args, **kwargs):
  '''
  This function computes the blob-blob forces and returns
  an array with shape (Nblobs, 3).
  '''
    
  # Get parameters from arguments
  L = kwargs.get('periodic_length')
  eps = kwargs.get('repulsion_strength')
  b = kwargs.get('debye_length')
  a = kwargs.get('blob_radius')
  d_max = 2 * a + 30 * b

  # Build tree and find neighbors
  build_tree = True
  if len(calc_blob_blob_forces_tree_numba.list_of_neighbors) > 0:
    if np.array_equal(calc_blob_blob_forces_tree_numba.r_vectors_old, r_vectors):
      build_tree = False
      list_of_neighbors = calc_blob_blob_forces_tree_numba.list_of_neighbors
      offsets = calc_blob_blob_forces_tree_numba.offsets
  if build_tree:  
    tree = scsp.cKDTree(r_vectors)
    pairs = tree.query_ball_tree(tree, d_max)
    offsets = np.zeros(len(pairs)+1, dtype=int)
    for i in range(len(pairs)):
      offsets[i+1] = offsets[i] + len(pairs[i])
    list_of_neighbors = np.concatenate(pairs).ravel()
    calc_blob_blob_forces_tree_numba.offsets = np.copy(offsets)
    calc_blob_blob_forces_tree_numba.list_of_neighbors = np.copy(list_of_neighbors)
    calc_blob_blob_forces_tree_numba.r_vectors_old = np.copy(r_vectors)
  
  # Compute forces
  force_blobs = blob_blob_force_tree_numba(r_vectors, L, eps, b, a, list_of_neighbors, offsets)
  return force_blobs
