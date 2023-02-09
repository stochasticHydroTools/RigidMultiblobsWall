
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
def blob_blob_force_radii_numba(r_vectors, radius_blobs, L, eps, b, a):
  '''
  This function compute the force between two blobs
  with vector between blob centers r.

  The effective radius, a, for interaction blobs i and j with radius a_i and a_j is
  a = (a_i + a_j) / 2

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
  radius_blobs = radius_blobs.reshape(N)
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
      a = (radius_blobs[i] + radius_blobs[j]) * 0.5
      r_norm = np.sqrt(dr[0]*dr[0] + dr[1]*dr[1] + dr[2]*dr[2])
      if r_norm > 2*a:
        f0 = -((eps / b) * np.exp(-(r_norm - 2.0*a) / b) / r_norm)
      else:
        f0 = -((eps / b) / np.maximum(r_norm, 1e-25))

      for k in range(3):
        force[i,k] += f0*dr[k]

  return force


def calc_blob_blob_forces_radii_numba(r_vectors, radius_blobs, *args, **kwargs):
  '''
  This function computes the blob-blob forces and returns
  an array with shape (Nblobs, 3).
  '''
    
  # Get parameters from arguments
  L = kwargs.get('periodic_length')
  eps = kwargs.get('repulsion_strength')
  b = kwargs.get('debye_length')
  a = kwargs.get('blob_radius')

  force_blobs = blob_blob_force_radii_numba(r_vectors, radius_blobs, L, eps, b, a)
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
  Lx = L[0]
  Ly = L[1]
  Lz = L[2]

  for i in prange(N):
    for k in range(offsets[i+1] - offsets[i]):
      j = list_of_neighbors[offsets[i] + k]
      if i == j:
        continue
      rx = rx_vec[j] - rx_vec[i]
      ry = ry_vec[j] - ry_vec[i]
      rz = rz_vec[j] - rz_vec[i]

      # Use distance with PBC
      if Lx > 0:
        rx = rx - int(rx / Lx + 0.5 * (int(rx>0) - int(rx<0))) * Lx
      if Ly > 0:
        ry = ry - int(ry / Ly + 0.5 * (int(ry>0) - int(ry<0))) * Ly
      if Lz > 0:
        rz = rz - int(rz / Lz + 0.5 * (int(rz>0) - int(rz<0))) * Lz

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

  def project_to_periodic_image(r, L):
    '''
    Project a vector r to the minimal image representation
    of size L=(Lx, Ly, Lz) and with a corner at (0,0,0). If 
    any dimension of L is equal or smaller than zero the 
    box is assumed to be infinite in that direction.
    
    If one dimension is not periodic shift all coordinates by min(r[:,i]) value.
    '''
    if L is not None:
      for i in range(3):
        if(L[i] > 0):
          r[:,i] = r[:,i] - (r[:,i] // L[i]) * L[i]
        else:
          ri_min =  np.min(r[:,i])
          if ri_min < 0:
            r[:,i] -= ri_min
    return r
    
  # Get parameters from arguments
  L = kwargs.get('periodic_length')
  eps = kwargs.get('repulsion_strength')
  b = kwargs.get('debye_length')
  a = kwargs.get('blob_radius')
  d_max = 2 * a + 30 * b

  # Project to PBC, this is necessary here to build the Kd-tree with scipy.
  # Copy is necessary because we don't want to modify the original vector here
  r_vectors = project_to_periodic_image(np.copy(r_vectors), L)

  # Build tree and find neighbors
  build_tree = True
  if len(calc_blob_blob_forces_tree_numba.list_of_neighbors) > 0:
    if np.array_equal(calc_blob_blob_forces_tree_numba.r_vectors_old, r_vectors):
      build_tree = False
      list_of_neighbors = calc_blob_blob_forces_tree_numba.list_of_neighbors
      offsets = calc_blob_blob_forces_tree_numba.offsets
  if build_tree:
    # Set box dimensions for PBC
    if L[0] > 0 or L[1] > 0 or L[2] > 0:
      boxsize = np.zeros(3)
      for i in range(3):
        if L[i] > 0:
          boxsize[i] = L[i]
        else:
          boxsize[i] = (np.max(r_vectors[:,i]) - np.min(r_vectors[:,i])) + d_max * 10
    else:
      boxsize = None   

    # Build tree
    tree = scsp.cKDTree(r_vectors, boxsize=boxsize)
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
