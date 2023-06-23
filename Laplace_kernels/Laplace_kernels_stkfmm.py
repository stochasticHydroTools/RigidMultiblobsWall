import numpy as np
from functools import partial
import general_application_utils as utils

# Try to import numba
try:
  from numba import njit, prange
except ImportError:
  print('numba not found')

# CHANGE 7: Add these lines at the top of mobility.py
import general_application_utils as utils
# Try to import stkfmm library
try:
  import PySTKFMM
except ImportError:
  print('PySTKFMM not found')
  pass


def set_Laplace_kernels(mult_order, pbc_string, max_pts, L=np.zeros(3), *args, **kwargs):
  print('pbc_string = ', pbc_string)

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

  # u, lapu kernel (4->6)
  kernel = PySTKFMM.KERNEL.LapPGrad

  # Setup FMM
  LapPGrad = PySTKFMM.Stk3DFMM(mult_order, max_pts, pbc, kernel)
  no_wall_Laplace_kernels_stkfmm_partial = partial(Laplace_kernels_stkfmm,
                                                   LapPGrad=LapPGrad, 
                                                   L=L,
                                                   comm=kwargs.get('comm'))
  return no_wall_Laplace_kernels_stkfmm_partial


@utils.static_var('r_vectors_old', [])
def Laplace_kernels_stkfmm(r, field_SL, field_DL, weights, LapPGrad, L=np.zeros(3), *args, **kwargs):
  '''
  WARNING: compared with the kernels implemented in Laplace_kernels_numba.py this function computes
  Single_layer(inputs) = c,  grad_c
  Double_layer(inputs) = c, -grad_c
  
  Note the minus sign. It should be taken into account when calling this function.
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
  
  # Prepare coordinates
  N = r.size // 3
  r_vectors = np.copy(r)
  r_vectors = project_to_periodic_image(r_vectors, L)
 
  # Set tree if necessary
  build_tree = True
  if len(Laplace_kernels_stkfmm.r_vectors_old) > 0:
    if np.array_equal(Laplace_kernels_stkfmm.r_vectors_old, r_vectors):
      Laplace_kernels_stkfmm.r_vectors_old = np.copy(r_vectors)
      build_tree = True
  if build_tree:
    # Build tree in STKFMM
    if L[0] > 0:
      x_min = 0
      Lx_pvfmm = L[0]
      Lx_cKDTree = L[0]
    else:
      x_min = np.min(r_vectors[:,0])
      Lx_pvfmm = (np.max(r_vectors[:,0]) * 1.01 - x_min)
      Lx_cKDTree = (np.max(r_vectors[:,0]) * 1.01 - x_min) * 10
    if L[1] > 0:
      y_min = 0
      Ly_pvfmm = L[1]
      Ly_cKDTree = L[1]
    else:
      y_min = np.min(r_vectors[:,1])
      Ly_pvfmm = (np.max(r_vectors[:,1]) * 1.01 - y_min)
      Ly_cKDTree = (np.max(r_vectors[:,1]) * 1.01 - y_min) * 10
    if L[2] > 0:
      z_min = 0
      Lz_pvfmm = L[2]
      Lz_cKDTree = L[2]
    else:
      z_min = np.min(r_vectors[:,2])
      z_min = 0
      Lz_pvfmm = (np.max(r_vectors[:,2]) * 1.01 - z_min)
      Lz_cKDTree = (np.max(r_vectors[:,2]) * 1.01 - z_min) * 10

    # Set box size for pvfmm
    if L[0] > 0 or L[1] > 0 or L[2] > 0:
      L_box = np.max(L)
    else:
      L_box = np.max([Lx_pvfmm, Ly_pvfmm, 2 * Lz_pvfmm])
    
    # Set box size for pvfmm
    if L[0] > 0 or L[1] > 0 or L[2] > 0:
      L_box = np.max(L)
    else:
      L_box = np.max([Lx_pvfmm, Ly_pvfmm, 2 * Lz_pvfmm])

    # Buid FMM tree
    LapPGrad.set_box(np.array([x_min, y_min, z_min]), L_box)
    LapPGrad.set_points(r_vectors, r_vectors, r_vectors)
    LapPGrad.setup_tree(PySTKFMM.KERNEL.LapPGrad)

  # Set single and double layer
  trg_value = np.zeros((N, 4))
  src_SL_value = field_SL * weights 
  src_DL_value = field_DL * weights[:,None] 

  # Evaluate fmm; format c = trg_value[:,0], grad_c = trg_value[:,1:4]
  LapPGrad.clear_fmm(PySTKFMM.KERNEL.LapPGrad)
  LapPGrad.evaluate_fmm(PySTKFMM.KERNEL.LapPGrad, src_SL_value, trg_value, src_DL_value)
  comm = kwargs.get('comm')
  comm.Barrier()

  # Return concentration and gradient
  c = trg_value[:,0]
  grad_c = trg_value[:,1:4]
  return c, -1 * grad_c

