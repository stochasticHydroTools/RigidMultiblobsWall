from functools import partial
import numpy as np
import sys
sys.path.append('../')

try:
  from mpi4py import MPI
except ImportError:
  print('It didn\'t find mpi4py!')

# CHANGE 2: Add path for PySTKFMM
sys.path.append('/home/fbalboa/sfw/FMM2/STKFMM-lib-gnu/lib64/python/')
PySTKFMM_found = False

# CHANGE 3: Load STKFMM
try:
  import PySTKFMM
  PySTKFMM_found = True
except ImportError:
  print('STKFMM library not found')

import mobility as mob


# CHANGE 5: Add this function to multi_bodies.py
def set_double_layer_kernels(implementation, mult_order, pbc_string, max_pts, L=np.zeros(3), blob_radius=0, *args, **kwargs):
  if pbc_string == 'None':
    pbc = PySTKFMM.PAXIS.NONE
  elif pbc_string == 'PX':
    pbc = PySTKFMM.PAXIS.PX
  elif pbc_string == 'PXY':
    pbc = PySTKFMM.PAXIS.PXY
  elif pbc_string == 'PXYZ':
    pbc = PySTKFMM.PAXIS.PXYZ
  else:
    print('Error while setting pbc for stkfmm!')

  comm = kwargs.get('comm')
  comm.Barrier()

  # Get kernel
  if implementation == 'PVel':
    kernel = PySTKFMM.KERNEL.PVel
  elif implementation == 'PVelLaplacian':
    kernel = PySTKFMM.KERNEL.PVelLaplacian
  else:
    print('Wrong kernel for double layer STKFMM')
    sys.exit()

  # Setup FMM
  PVel = PySTKFMM.Stk3DFMM(mult_order, max_pts, pbc, kernel)
  function_partial = partial(mob.double_layer_stkfmm,
                             PVel=PVel, 
                             L=L,
                             kernel=kernel,
                             blob_radius=blob_radius,
                             comm=kwargs.get('comm'))  
  return function_partial
  

if __name__ == '__main__':
  print('# Start')
  # MPI
  if PySTKFMM_found:
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
  else:
    comm = None

  # Set parameters
  N_src = 10
  N = N_src 
  eta = 1
  a = 1
  z_min = 2
  dx = 1e-06
  blob_radius = 0.2
  stkfmm_mult_order = 10
  stkfmm_pbc = 'None'
  stkfmm_max_points = 512
  mobility_vector_prod_implementation = 'numba_no_wall'

  # Create vectors
  # np.random.seed(0)
  r_src = np.random.rand(N_src, 3)
  r_src[:,2] += z_min
  radius_source = np.zeros(N_src)
  weights = np.ones(N_src) * a  
  vector_src = np.random.normal(0, 1, N_src * 3).reshape((N_src, 3))
  normals_src = np.random.normal(0, 1, N_src * 3).reshape((N_src, 3))
  normals_src /= np.linalg.norm(normals_src, axis=1)[:,None]
  print('r_src       = ', r_src)
  print('normals_src = ', normals_src)
  print('vector_src  = ', vector_src)
  print(' ')

  # CHANGE 7: Inside the main replace the call to set_mobility_vector_prod with these lines
  no_wall_double_layer_PVelLaplacian = set_double_layer_kernels('PVelLaplacian',
                                                                stkfmm_mult_order,
                                                                stkfmm_pbc,
                                                                stkfmm_max_points,
                                                                L=np.zeros(3),
                                                                blob_radius=blob_radius,
                                                                comm=comm)  
  
  # Test symmetry
  velocity_numba = mob.no_wall_double_layer_source_target_numba(r_src, r_src, normals_src, vector_src, weights, blob_radius)
  velocity_PVelLaplacian = no_wall_double_layer_PVelLaplacian(r_src, normals_src, vector_src, weights).flatten()

  # # Compute difference
  vel_diff = velocity_PVelLaplacian - velocity_numba
  
  # print('No wall test')
  print('vel_numba         = ', velocity_numba)
  print('vel_PVelLaplacian = ', velocity_PVelLaplacian)
  print(' ')
  print('|vel_diff|     = ', np.linalg.norm(vel_diff))
  print('|vel_diff|_rel =', np.linalg.norm(vel_diff) / np.linalg.norm(velocity_numba))
  print(' ')
  
  
  
