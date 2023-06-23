# CHANGE 1: Add to the top of multi_bodies.py these lines with the right path to STKFMM
try:
  from mpi4py import MPI
except ImportError:
  print('It didn\'t find mpi4py!')

# CHANGE 2: Add path for PySTKFMM
sys.path.append('/home/fbalboa/sfw/FMM2/STKFMM-lib-gnu/lib/python/')
PySTKFMM_found = False

# CHANGE 3: Load STKFMM
try:
  import PySTKFMM
  PySTKFMM_found = True
except ImportError:
  print('STKFMM library not found')


# CHANGE 4: Replace this function in multi_bodies.py
def set_mobility_vector_prod(implementation, *args, **kwargs):
  '''
  New function with STKFMM call.
  ''' 
  # Implementations without wall
  if implementation == 'python_no_wall':
    return mb.no_wall_fluid_mobility_product
  elif implementation == 'pycuda_no_wall':
    return mb.no_wall_mobility_trans_times_force_pycuda
  elif implementation == 'numba_no_wall':
    return mb.no_wall_mobility_trans_times_force_numba
  elif implementation == 'stkfmm_no_wall':
    # STKFMM parameters
    mult_order = kwargs.get('stkfmm_mult_order')
    pbc_string = kwargs.get('stkfmm_pbc')
    max_pts = kwargs.get('stkfmm_max_points')
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
    kernel = PySTKFMM.KERNEL.RPY

    # Setup FMM
    rpy_fmm = PySTKFMM.Stk3DFMM(mult_order, max_pts, pbc, kernel)
    no_wall_mobility_trans_times_force_stkfmm_partial = partial(mb.mobility_trans_times_force_stkfmm, 
                                                                rpy_fmm=rpy_fmm, 
                                                                L=kwargs.get('L'),
                                                                wall=False,
                                                                comm=kwargs.get('comm'))
    return no_wall_mobility_trans_times_force_stkfmm_partial

  
  # Implementations with wall
  elif implementation == 'python':
    return mb.single_wall_fluid_mobility_product
  elif implementation == 'C++':
    return mb.single_wall_mobility_trans_times_force_cpp
  elif implementation == 'pycuda':
    return mb.single_wall_mobility_trans_times_force_pycuda
  elif implementation == 'numba':
    return mb.single_wall_mobility_trans_times_force_numba
  elif implementation == 'stkfmm_single_wall':
    # STKFMM parameters
    mult_order = kwargs.get('stkfmm_mult_order')
    pbc_string = kwargs.get('stkfmm_pbc')
    max_pts = kwargs.get('stkfmm_max_points')
    if pbc_string == 'None':
      pbc = PySTKFMM.PAXIS.NONE
    elif pbc_string == 'PX':
      pbc = PySTKFMM.PAXIS.PX
    elif pbc_string == 'PXY':
      pbc = PySTKFMM.PAXIS.PXY
    elif pbc_string == 'PXYZ':
      pbc = PySTKFMM.PAXIS.PXYZ

    # u, lapu kernel (4->6)
    kernel = PySTKFMM.KERNEL.RPY

    # Setup FMM
    rpy_fmm = PySTKFMM.StkWallFMM(mult_order, max_pts, pbc, kernel)
    no_wall_mobility_trans_times_force_stkfmm_partial = partial(mb.mobility_trans_times_force_stkfmm, 
                                                                rpy_fmm=rpy_fmm, 
                                                                L=kwargs.get('L'),
                                                                wall=True,
                                                                comm=kwargs.get('comm'))
    return no_wall_mobility_trans_times_force_stkfmm_partial
  
  # Implementations free surface
  elif implementation == 'pycuda_free_surface':
    return mb.free_surface_mobility_trans_times_force_pycuda
  # Implementations different radii
  elif implementation.find('radii') > -1:
    # Get right function
    if implementation == 'radii_numba':
      function = mb.single_wall_mobility_trans_times_force_source_target_numba
    elif implementation == 'radii_numba_no_wall':
      function = mb.no_wall_mobility_trans_times_force_source_target_numba
    elif implementation == 'radii_pycuda':
      function = mb.single_wall_mobility_trans_times_force_source_target_pycuda
    elif implementation == 'radii':
      function = mb.mobility_vector_product_source_target_one_wall
    elif implementation == 'radii_no_wall':
      function = mb.mobility_vector_product_source_target_unbounded
    # Get blobs radii
    bodies = kwargs.get('bodies')
    radius_blobs = []
    for k, b in enumerate(bodies):
      radius_blobs.append(b.blobs_radius)
    radius_blobs = np.concatenate(radius_blobs, axis=0)    
    return partial(mb.mobility_radii_trans_times_force, radius_blobs=radius_blobs, function=function)


# CHANGE 5: Add this function to multi_bodies.py
def set_double_layer_kernels(mult_order, pbc_string, max_pts, L=np.zeros(3), *args, **kwargs):
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

  comm = kwargs.get('comm')
  comm.Barrier()

  # u, lapu kernel (4->6)
  kernel = PySTKFMM.KERNEL.PVel

  # Setup FMM
  PVel = PySTKFMM.Stk3DFMM(mult_order, max_pts, pbc, kernel)
  no_wall_double_layer_stkfmm_partial = partial(mb.double_layer_stkfmm,
                                                PVel=PVel, 
                                                L=L,
                                                comm=kwargs.get('comm'))
  return no_wall_double_layer_stkfmm_partial



# CHANGE 6: Add these lines at the very top of __main__ in multi_bodies.py
  # MPI
  if PySTKFMM_found:
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
  else:
    comm = None


# CHANGE 7: Inside the main replace the call to set_mobility_vector_prod with these lines
  mobility_vector_prod = set_mobility_vector_prod(read.mobility_vector_prod_implementation, 
                                                  stkfmm_mult_order=read.stkfmm_mult_order, 
                                                  stkfmm_pbc=read.stkfmm_pbc,
                                                  stkfmm_max_points=read.stkfmm_max_points,
                                                  L=read.periodic_length,
                                                  comm=comm)

# CHANGE 8: add to read_input.py
    # Info for STKFMM
    self.stkfmm_mult_order = int(self.options.get('stkfmm_mult_order') or 8)
    self.stkfmm_max_points = int(self.options.get('stkfmm_max_points') or 512)
    self.stkfmm_pbc = str(self.options.get('stkfmm_pbc') or 'None')  



# CHANGE 9: Add these lines at the top of mobility.py
import general_application_utils as utils
# Try to import stkfmm library
try:
  import PySTKFMM
except ImportError:
  pass



# CHANGE 10, LAST CHANGE: Add the rest of the file to mobility.py
import scipy.spatial as scsp
from numba import njit, prange

@njit(parallel=True, fastmath=True)
def no_wall_mobility_trans_times_force_overlap_correction_numba(r_vectors, force, eta, a, list_of_neighbors, offsets, L=np.array([0., 0., 0.])):
  ''' 
  Returns the blob-blob overlap correction for unbound fluids using the
  RPY mobility. It subtract the uncorrected value for r<2*a and it adds
  the corrected value.

  This function uses numba.
  '''
  # Variables
  N = r_vectors.size // 3
  r_vectors = r_vectors.reshape(N, 3)
  force = force.reshape(N, 3)
  u = np.zeros((N, 3))
  fourOverThree = 4.0 / 3.0
  inva = 1.0 / a
  norm_fact_f = 1.0 / (8.0 * np.pi * eta * a)

  Lx = L[0]
  Ly = L[1]
  Lz = L[2]
  
  rx_vec = np.copy(r_vectors[:,0])
  ry_vec = np.copy(r_vectors[:,1])
  rz_vec = np.copy(r_vectors[:,2])
  fx_vec = np.copy(force[:,0])
  fy_vec = np.copy(force[:,1])
  fz_vec = np.copy(force[:,2])
  
  # Loop over image boxes and then over particles
  for i in prange(N):
    rxi = rx_vec[i]
    ryi = ry_vec[i]
    rzi = rz_vec[i]
    ux = 0
    uy = 0
    uz = 0
    for k in range(offsets[i+1] - offsets[i]):
      j = list_of_neighbors[offsets[i] + k]
      if i == j:
        continue
      # Compute vector between particles i and j
      rx = rxi - rx_vec[j]
      ry = ryi - ry_vec[j]
      rz = rzi - rz_vec[j]

      # PBC
      if Lx > 0:
        rx = rx - int(rx / Lx + 0.5 * (int(rx>0) - int(rx<0))) * Lx
      if Ly > 0:
        ry = ry - int(ry / Ly + 0.5 * (int(ry>0) - int(ry<0))) * Ly
      if Lz > 0:
        rz = rz - int(rz / Lz + 0.5 * (int(rz>0) - int(rz<0))) * Lz
      
      # Normalize distance with hydrodynamic radius
      rx = rx * inva 
      ry = ry * inva
      rz = rz * inva
      r2 = rx*rx + ry*ry + rz*rz
      r = np.sqrt(r2)
        
      # TODO: We should not divide by zero 
      invr = 1.0 / r
      invr2 = invr * invr
        
      if r > 2:
        Mxx = 0
        Mxy = 0
        Mxz = 0
        Myy = 0
        Myz = 0
        Mzz = 0
      else:
        c1 = fourOverThree * (1.0 - 0.28125 * r) # 9/32 = 0.28125
        c2 = fourOverThree * 0.09375 * invr      # 3/32 = 0.09375
        Mxx = c1 + c2 * rx*rx 
        Mxy =      c2 * rx*ry 
        Mxz =      c2 * rx*rz 
        Myy = c1 + c2 * ry*ry 
        Myz =      c2 * ry*rz 
        Mzz = c1 + c2 * rz*rz 
        c1 = 1.0 + 2.0 / (3.0 * r2)
        c2 = (1.0 - 2.0 * invr2) * invr2
        Mxx -= (c1 + c2*rx*rx) * invr
        Mxy -= (     c2*rx*ry) * invr
        Mxz -= (     c2*rx*rz) * invr
        Myy -= (c1 + c2*ry*ry) * invr
        Myz -= (     c2*ry*rz) * invr
        Mzz -= (c1 + c2*rz*rz) * invr                     
      Myx = Mxy
      Mzx = Mxz
      Mzy = Myz
	  
      # 2. Compute product M_ij * F_j           
      ux += (Mxx * fx_vec[j] + Mxy * fy_vec[j] + Mxz * fz_vec[j]) 
      uy += (Myx * fx_vec[j] + Myy * fy_vec[j] + Myz * fz_vec[j]) 
      uz += (Mzx * fx_vec[j] + Mzy * fy_vec[j] + Mzz * fz_vec[j]) 
    u[i,0] = ux * norm_fact_f
    u[i,1] = uy * norm_fact_f
    u[i,2] = uz * norm_fact_f          
  return u.flatten()


@utils.static_var('r_vectors_old', [])
@utils.static_var('list_of_neighbors', [])
@utils.static_var('offsets', [])
def mobility_trans_times_force_stkfmm(r, force, eta, a, rpy_fmm=None, L=np.array([0.,0.,0.]), wall=False, *args, **kwargs):
  ''' 
  Returns the product of the mobility at the blob level to the force 
  on the blobs. Mobility for particles in an unbounded, semiperiodic or
  periodic domain. It uses the standard RPY tensor.
  
  This function uses the stkfmm library.
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

  if wall:
    # Compute damping matrix B
    B_damp, overlap = damping_matrix_B(r_vectors, a, *args, **kwargs)
    # Get effective height
    r_vectors = shift_heights(r_vectors, a)

    if overlap is True:
      force = B_damp.dot(force.flatten())

  # Set tree if necessary
  build_tree = True
  if len(mobility_trans_times_force_stkfmm.list_of_neighbors) > 0:
    if np.array_equal(mobility_trans_times_force_stkfmm.r_vectors_old, r_vectors):
      build_tree = False
      list_of_neighbors = mobility_trans_times_force_stkfmm.list_of_neighbors
      offsets = mobility_trans_times_force_stkfmm.offsets
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

    # Buid FMM tree
    rpy_fmm.set_box(np.array([x_min, y_min, z_min]), L_box)
    rpy_fmm.set_points(r_vectors, r_vectors, np.zeros(0))
    rpy_fmm.setup_tree(PySTKFMM.KERNEL.RPY)

    # Build tree in python and neighbors lists
    d_max = 2 * a
    if L[0] > 0 or L[1] > 0 or L[2] > 0:
      boxsize = np.array([Lx_cKDTree, Ly_cKDTree, Lz_cKDTree])      
    else:
      boxsize = None
    tree = scsp.cKDTree(r_vectors, boxsize = boxsize)
    pairs = tree.query_ball_tree(tree, d_max)
    offsets = np.zeros(len(pairs)+1, dtype=int)
    for i in range(len(pairs)):
      offsets[i+1] = offsets[i] + len(pairs[i])
    list_of_neighbors = np.concatenate(pairs).ravel()
    mobility_trans_times_force_stkfmm.offsets = np.copy(offsets)
    mobility_trans_times_force_stkfmm.list_of_neighbors = np.copy(list_of_neighbors)
    mobility_trans_times_force_stkfmm.r_vectors_old = np.copy(r_vectors)

  # Set force with right format (single layer potential)
  trg_value = np.zeros((N, 6))
  src_SL_value = np.zeros((N, 4))
  src_SL_value[:,0:3] = np.copy(force.reshape((N, 3)))
  src_SL_value[:,3] = a
    
  # Evaluate fmm; format p = trg_value[:,0], v = trg_value[:,1:4], Lap = trg_value[:,4:]
  rpy_fmm.clear_fmm(PySTKFMM.KERNEL.RPY)
  rpy_fmm.evaluate_fmm(PySTKFMM.KERNEL.RPY, src_SL_value, trg_value, np.zeros(0))
  comm = kwargs.get('comm')
  comm.Barrier()

  # Compute RPY mobility 
  # 1. Self mobility 
  vel = (1.0 / (6.0 * np.pi * eta * a)) * force.reshape((N,3)) 
  # 2. Stokeslet 
  vel += trg_value[:,0:3] / (eta) 
  # 3. Laplacian 
  vel += (a**2 / (6.0 * eta)) * trg_value[:,3:] 
  # 4. Double Laplacian 
  #    it is zero with PBC 
  # 5. Add blob-blob overlap correction 
  v_overlap = no_wall_mobility_trans_times_force_overlap_correction_numba(r_vectors, force, eta, a, list_of_neighbors, offsets, L=L) 
  vel += v_overlap.reshape((N, 3)) 
  
  if wall:
    if overlap is True:
      vel = B_damp.dot(vel.flatten())
  
  return vel.flatten()


@utils.static_var('r_source_old', [])
@utils.static_var('r_target_old', [])
@utils.static_var('list_of_neighbors', [])
@utils.static_var('offsets', [])
def fluid_velocity_stkfmm(r_source, r_target, force, eta, a, rpy_fmm=None, L=np.array([0.,0.,0.]), wall=False, *args, **kwargs):
  ''' 
  Velocity of the fluid computed as

  v(x) = (1 + a**2/6 * Laplacian_y) * G(x,y) * F(y)   if r > a.
  v(x) = F(y) / (6 * pi * eta * a) if r <= a.

  with r = |x - y|
  
  This function uses the stkfmm library.
  '''
  def project_to_periodic_image(r1, r2, L):
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
          r1[:,i] = r1[:,i] - (r1[:,i] // L[i]) * L[i]
          r2[:,i] = r2[:,i] - (r2[:,i] // L[i]) * L[i]
        else:
          ri_min =  min(np.min(r1[:,i]), np.min(r2[:,i]))
          if ri_min < 0:
            r1[:,i] -= ri_min
            r2[:,i] -= ri_min
    return r1, r2

  # Prepare coordinates
  N_source = r_source.size // 3
  N_target = r_target.size // 3
  r_source, r_target = project_to_periodic_image(np.copy(r_source), np.copy(r_target), L)

  if wall:
    # Compute damping matrix B
    B_damp, overlap = damping_matrix_B(r_vectors, a, *args, **kwargs)
    # Get effective height
    r_vectors = shift_heights(r_vectors, a)

    if overlap is True:
      force = B_damp.dot(force.flatten())

  # Set tree if necessary
  build_tree = True
  if len(fluid_velocity_stkfmm.list_of_neighbors) > 0:
    if np.array_equal(fluid_velocity_stkfmm.r_source_old, r_source) and np.array_equal(fluid_velocity_stkfmm.r_target_old, r_target):
      build_tree = False
      list_of_neighbors = fluid_velocity_stkfmm.list_of_neighbors
      offsets = fluid_velocity_stkfmm.offsets
  if build_tree:
    # Build tree in STKFMM
    if L[0] > 0:
      x_min = 0
      Lx_pvfmm = L[0]
      Lx_cKDTree = L[0]
    else:
      x_min = min(np.min(r_source[:,0]), np.min(r_target[:,0]))
      Lx_pvfmm = max(np.max(r_source[:,0]), np.max(r_target[:,0])) * 1.01 - x_min
      Lx_cKDTree = (max(np.max(r_source[:,0]), np.max(r_target[:,0])) * 1.01 - x_min) * 10
    if L[1] > 0:
      y_min = 0
      Ly_pvfmm = L[1]
      Ly_cKDTree = L[1]
    else:
      y_min = min(np.min(r_source[:,1]), np.min(r_target[:,1]))
      Ly_pvfmm = max(np.max(r_source[:,1]), np.max(r_target[:,1])) * 1.01 - y_min            
      Ly_cKDTree = (max(np.max(r_source[:,1]), np.max(r_target[:,1])) * 1.01 - y_min) * 10
    if L[2] > 0:
      z_min = 0
      Lz_pvfmm = L[2]
      Lz_cKDTree = L[2]
    else:
      z_min = 0
      Lz_pvfmm = max(np.max(r_source[:,2]), np.max(r_target[:,2])) * 1.01 - z_min
      Lz_cKDTree = (max(np.max(r_source[:,2]), np.max(r_target[:,2])) * 1.01 - z_min) * 10

    # Set box size for pvfmm
    if L[0] > 0 or L[1] > 0 or L[2] > 0:
      L_box = np.max(L)
    else:
      L_box = np.max([Lx_pvfmm, Ly_pvfmm, Lz_pvfmm])

    # Buid FMM tree
    rpy_fmm.set_box(np.array([x_min, y_min, z_min]), L_box)
    rpy_fmm.set_points(r_source, r_target, np.zeros(0))
    rpy_fmm.setup_tree(PySTKFMM.KERNEL.RPY)

    # Build tree in python and neighbors lists
    d_max = a
    if L[0] > 0 or L[1] > 0 or L[2] > 0:
      boxsize = np.array([Lx_cKDTree, Ly_cKDTree, Lz_cKDTree])      
    else:
      boxsize = None
    tree_source = scsp.cKDTree(r_source, boxsize = boxsize)
    tree_target = scsp.cKDTree(r_target, boxsize = boxsize)
    # pairs = tree_source.query_ball_tree(tree_target, d_max)
    pairs = tree_target.query_ball_tree(tree_source, d_max)
    offsets = np.zeros(len(pairs)+1, dtype=int)
    for i in range(len(pairs)):
      offsets[i+1] = offsets[i] + len(pairs[i])
    list_of_neighbors = np.concatenate(pairs).ravel().astype(int)
    fluid_velocity_stkfmm.offsets = np.copy(offsets)
    fluid_velocity_stkfmm.list_of_neighbors = np.copy(list_of_neighbors)
    fluid_velocity_stkfmm.r_source_old = np.copy(r_source)
    fluid_velocity_stkfmm.r_target_old = np.copy(r_target)

  # Set force with right format (single layer potential)
  trg_value = np.zeros((N_target, 6))
  src_SL_value = np.zeros((N_source, 4))
  src_SL_value[:,0:3] = np.copy(force.reshape((N_source, 3)))
  src_SL_value[:,3] = a
    
  # Evaluate fmm; format p = trg_value[:,0], v = trg_value[:,1:4], Lap = trg_value[:,4:]
  rpy_fmm.clear_fmm(PySTKFMM.KERNEL.RPY)
  rpy_fmm.evaluate_fmm(PySTKFMM.KERNEL.RPY, src_SL_value, trg_value, np.zeros(0))
  comm = kwargs.get('comm')
  comm.Barrier()
 
  # Compute RPY mobility 
  # 1. Stokeslet 
  vel = trg_value[:,0:3] / (eta)
  if list_of_neighbors.size > 0:
    v_overlap = fluid_velocity_overlap_correction_numba(r_source, r_target, force, eta, a, list_of_neighbors, offsets, L=L) 
    vel += v_overlap.reshape((N_target, 3))    
  return vel.flatten()


@njit(parallel=True, fastmath=True)
def fluid_velocity_overlap_correction_numba(r_source, r_target, force, eta, a, list_of_neighbors, offsets, L=np.array([0., 0., 0.])):
  ''' 
  Returns the blob-blob overlap correction for unbound fluids using the
  RPY mobility. It subtract the uncorrected value for r<2*a and it adds
  the corrected value.

  This function uses numba.
  '''
  # Variables
  N_source = r_source.size // 3
  r_source = r_source.reshape(N_source, 3)
  N_target = r_target.size // 3
  r_taget = r_target.reshape(N_target, 3) 
  force = force.reshape(N_source, 3)
  u = np.zeros((N_target, 3))
  fourOverThree = 4.0 / 3.0
  inva = 1.0 / a
  norm_fact_f = 1.0 / (8.0 * np.pi * eta * a)

  Lx = L[0]
  Ly = L[1]
  Lz = L[2]
  
  rx_src = np.copy(r_source[:,0])
  ry_src = np.copy(r_source[:,1])
  rz_src = np.copy(r_source[:,2])
  rx_trg = np.copy(r_target[:,0])
  ry_trg = np.copy(r_target[:,1])
  rz_trg = np.copy(r_target[:,2])
  fx_vec = np.copy(force[:,0])
  fy_vec = np.copy(force[:,1])
  fz_vec = np.copy(force[:,2])
  
  # Loop over image boxes and then over particles
  for i in prange(N_target):
    rxi = rx_trg[i]
    ryi = ry_trg[i]
    rzi = rz_trg[i]
    ux = 0
    uy = 0
    uz = 0
    for k in range(offsets[i+1] - offsets[i]):
      j = list_of_neighbors[offsets[i] + k]
      # Compute vector between particles i and j
      rx = rxi - rx_src[j]
      ry = ryi - ry_src[j]
      rz = rzi - rz_src[j]

      # PBC
      if Lx > 0:
        rx = rx - int(rx / Lx + 0.5 * (int(rx>0) - int(rx<0))) * Lx
      if Ly > 0:
        ry = ry - int(ry / Ly + 0.5 * (int(ry>0) - int(ry<0))) * Ly
      if Lz > 0:
        rz = rz - int(rz / Lz + 0.5 * (int(rz>0) - int(rz<0))) * Lz
      
      # Normalize distance with hydrodynamic radius
      rx = rx * inva 
      ry = ry * inva
      rz = rz * inva
      r2 = rx*rx + ry*ry + rz*rz
      r = np.sqrt(r2)
        
      # TODO: We should not divide by zero 
      invr = 1.0 / r
      invr2 = invr * invr
        
      if r > 1:
        Mxx = 0
        Mxy = 0
        Mxz = 0
        Myy = 0
        Myz = 0
        Mzz = 0
      if r == 0:
        pass
      else:
        Mxx = fourOverThree 
        Myy = fourOverThree 
        Mzz = fourOverThree 
        c1 = 1.0 + 1.0 / (3.0 * r2)
        c2 = (1.0 - 1.0 * invr2) * invr2
        Mxx -=  (c1 + c2*rx*rx) * invr
        Mxy  = -(     c2*rx*ry) * invr
        Mxz  = -(     c2*rx*rz) * invr
        Myy -=  (c1 + c2*ry*ry) * invr
        Myz  = -(     c2*ry*rz) * invr
        Mzz -=  (c1 + c2*rz*rz) * invr
      Myx = Mxy
      Mzx = Mxz
      Mzy = Myz
	  
      # 2. Compute product M_ij * F_j           
      ux += (Mxx * fx_vec[j] + Mxy * fy_vec[j] + Mxz * fz_vec[j]) 
      uy += (Myx * fx_vec[j] + Myy * fy_vec[j] + Myz * fz_vec[j]) 
      uz += (Mzx * fx_vec[j] + Mzy * fy_vec[j] + Mzz * fz_vec[j]) 
    u[i,0] = ux * norm_fact_f
    u[i,1] = uy * norm_fact_f
    u[i,2] = uz * norm_fact_f          
  return u.flatten()


@utils.static_var('r_vectors_old', [])
def double_layer_stkfmm(r, normals, field, weights, PVel, L=np.zeros(3), *args, **kwargs):
  '''
  Stokes double layer.
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
  if len(double_layer_stkfmm.r_vectors_old) > 0:
    if np.array_equal(double_layer_stkfmm.r_vectors_old, r_vectors):
      double_layer_stkfmm.r_vectors_old = np.copy(r_vectors)
      build_tree = False
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
    PVel.set_box(np.array([x_min, y_min, z_min]), L_box)
    # PVel.set_points(r_vectors, r_vectors, r_vectors)
    PVel.set_points(np.zeros(0), r_vectors, r_vectors)
    PVel.setup_tree(PySTKFMM.KERNEL.PVel)
    
  # Set double layer
  trg_value = np.zeros((N, 4))
  src_DL_value = np.einsum('bi,bj,b->bij', normals, field, weights).reshape((N, 9))
  src_SL_value = np.zeros((N, 4))  
  src_SL_value[:,3] = src_DL_value[:,0] + src_DL_value[:,4] + src_DL_value[:,8]

  # Evaluate fmm; format c = trg_value[:,0], grad_c = trg_value[:,1:4]
  PVel.clear_fmm(PySTKFMM.KERNEL.PVel)
  # PVel.evaluate_fmm(PySTKFMM.KERNEL.PVel, np.zeros((N,4)), trg_value, src_DL_value)
  PVel.evaluate_fmm(PySTKFMM.KERNEL.PVel, np.zeros(0), trg_value, src_DL_value)  
  comm = kwargs.get('comm')
  comm.Barrier()

  # Return velocity
  u = 2 * trg_value[:,1:4]
  return u
