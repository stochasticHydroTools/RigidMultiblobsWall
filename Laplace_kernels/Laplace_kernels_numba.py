
import numpy as np
import scipy.sparse

# Try to import numba
try:
  from numba import njit, prange
except ImportError:
  print('numba not found')


@njit(parallel=True, fastmath=True)
def Laplace_single_layer_operator_numba(r_vectors, field, weights, wall=0):
  ''' 
  Returns the product of the Laplace single layer operator to the concentration field on the particle's surface. Kernel in an unbounded 
  or half space domain.
  '''
  # Variables
  N = r_vectors.size // 3
  r_vectors = r_vectors.reshape(N, 3)
  res = np.zeros(N)
  norm_fact = 1.0 / (4.0 * np.pi)

  rx_vec = np.copy(r_vectors[:,0])
  ry_vec = np.copy(r_vectors[:,1])
  rz_vec = np.copy(r_vectors[:,2])

  for i in prange(N):
    rxi = rx_vec[i]
    ryi = ry_vec[i]
    rzi = rz_vec[i]

    for j in range(N):
      # Multiply field by quadrature weight   
      c = field[j] * weights[j]
         
      # Compute vector between particles i and j
      rx = rxi - r_vectors[j,0]
      ry = ryi - r_vectors[j,1]
      rz = rzi - r_vectors[j,2]
                
      # 1. Compute single layer kernel for pair i-j, if i==j kernel = 0
      if i == j:
        S = 0.0          
   
      else:
        # Compute distance
        r = np.sqrt(rx*rx + ry*ry + rz*rz)
              
        # TODO: We should not divide by zero 
        invr = 1.0 / r

        # Compute single layer kernel
        S = invr

      if wall:
        # Add wall contribution
        rz = rzi + r_vectors[j,2]
        r = np.sqrt(rx*rx + ry*ry + rz*rz)
        S += 1.0 / r
           
      # 2. Compute product S * c           
      res[i] += S * c

  return norm_fact * res


@njit(parallel=True, fastmath=True)
def Laplace_double_layer_operator_numba(r_vectors, field, weights, normals, wall=0):
  ''' 
  Returns the product of the Laplace double layer operator to the concentration field on the particle's surface. Kernel in an unbounded 
  of half space domain.
  '''
  # Variables
  N = r_vectors.size // 3
  r_vectors = r_vectors.reshape(N, 3)
  normals = normals.reshape(N, 3)
  res = np.zeros(N)
  norm_fact = -1.0 / (4.0 * np.pi)

  rx_vec = np.copy(r_vectors[:,0])
  ry_vec = np.copy(r_vectors[:,1])
  rz_vec = np.copy(r_vectors[:,2])
  nx_vec = np.copy(normals[:,0])
  ny_vec = np.copy(normals[:,1])
  nz_vec = np.copy(normals[:,2])

  for i in prange(N):
    rxi = rx_vec[i]
    ryi = ry_vec[i]
    rzi = rz_vec[i]

    for j in range(N):
      # Multiply field by quadrature weight   
      c = field[j] * weights[j]
         
      # Compute vector between particles i and j
      rx = rxi - r_vectors[j,0]
      ry = ryi - r_vectors[j,1]
      rz = rzi - r_vectors[j,2]

      nx = nx_vec[j]
      ny = ny_vec[j]
      nz = nz_vec[j]
               
      # 1. Compute double layer kernel for pair i-j, if i==j kernel = 0
      if i == j:
        T = 0.0          
 
      else:
        # Normalize distance with hydrodynamic radius
        r = np.sqrt(rx*rx + ry*ry + rz*rz)
              
        # TODO: We should not divide by zero 
        invr3 = 1.0 / r**3

        # Compute double  layer kernel T_i * n_i
        T =  invr3 * ( rx*nx + ry*ny + rz*nz )

      if wall:
        # Normalize distance with hydrodynamic radius
        rz = rzi + r_vectors[j,2]        
        r = np.sqrt(rx*rx + ry*ry + rz*rz)
              
        # TODO: We should not divide by zero 
        invr3 = 1.0 / r**3

        # Compute double  layer kernel T_i * n_i
        T +=  invr3 * ( rx*nx + ry*ny + rz*nz )

           
      # 2. Compute product T_i * n_i * c           
      res[i] += T * c

  return norm_fact * res


@njit(parallel=False, fastmath=False)
def Laplace_deriv_double_layer_operator_numba(r_vectors, field, weights, normals, wall=0):
  ''' 
  Returns the product of the derivative of the Laplace double layer operator to the concentration field on the particle's surface. Kernel in an unbounded 
  or half space domain.
  '''
  # Variables
  N = r_vectors.size // 3
  r_vectors = r_vectors.reshape(N, 3)
  normals = normals.reshape(N, 3)
  res = np.zeros((N,3))
  norm_fact = -1.0 / (4.0 * np.pi)

  rx_vec = np.copy(r_vectors[:,0])
  ry_vec = np.copy(r_vectors[:,1])
  rz_vec = np.copy(r_vectors[:,2])
  nx_vec = np.copy(normals[:,0])
  ny_vec = np.copy(normals[:,1])
  nz_vec = np.copy(normals[:,2])
  resx_vec = np.zeros(N)
  resy_vec = np.zeros(N)
  resz_vec = np.zeros(N)

  for i in prange(N):
    rxi = rx_vec[i]
    ryi = ry_vec[i]
    rzi = rz_vec[i]

    for j in range(N):
      # Multiply field by quadrature weight   
      c = field[j] * weights[j]
         
      nx = nx_vec[j]
      ny = ny_vec[j]
      nz = nz_vec[j]
             
      # 1. Compute kernel for pair i-j, if i==j kernel = 0
      if i == j:
        Lxx = 0.0
        Lxy = 0.0
        Lxz = 0.0
        Lyy = 0.0
        Lyz = 0.0
   
      else:
        # Compute vector between particles i and j
        rx = rxi - r_vectors[j,0]
        ry = ryi - r_vectors[j,1]
        rz = rzi - r_vectors[j,2]

        # Normalize distance with hydrodynamic radius
        r = np.sqrt(rx*rx + ry*ry + rz*rz)
              
        # TODO: We should not divide by zero 
        invr = 1.0 / r
        invr2 = invr * invr
        invr3 = invr2 * invr

        # Compute  kernel
        factor_off_diagonal = -3.0 * invr2
        Lxx = (1.0 + factor_off_diagonal * rx*rx) * invr3
        Lxy = (      factor_off_diagonal * rx*ry) * invr3
        Lxz = (      factor_off_diagonal * rx*rz) * invr3
        Lyy = (1.0 + factor_off_diagonal * ry*ry) * invr3
        Lyz = (      factor_off_diagonal * ry*rz) * invr3

      if wall:
        # Compute vector between particles i and j
        rz = rzi + r_vectors[j,2]

        # Normalize distance with hydrodynamic radius
        r = np.sqrt(rx*rx + ry*ry + rz*rz)
              
        # TODO: We should not divide by zero 
        invr = 1.0 / r
        invr2 = invr * invr
        invr3 = invr2 * invr

        # Compute  kernel
        factor_off_diagonal = -3.0 * invr2
        Lxx += (1.0 + factor_off_diagonal * rx*rx) * invr3
        Lxy += (      factor_off_diagonal * rx*ry) * invr3
        Lxz += (      factor_off_diagonal * rx*rz) * invr3
        Lyy += (1.0 + factor_off_diagonal * ry*ry) * invr3
        Lyz += (      factor_off_diagonal * ry*rz) * invr3

      # Uses symmetries of the kernel
      Lyx = Lxy
      Lzx = Lxz
      Lzy = Lyz
      Lzz = - Lxx - Lyy
      
      # 2. Compute product L_ij * n_j * c           
      resx_vec[i] += (Lxx * nx + Lxy * ny + Lxz * nz) * c 
      resy_vec[i] += (Lyx * nx + Lyy * ny + Lyz * nz) * c 
      resz_vec[i] += (Lzx * nx + Lzy * ny + Lzz * nz) * c 

    res[i,0] = resx_vec[i]
    res[i,1] = resy_vec[i]
    res[i,2] = resz_vec[i]

  return norm_fact * res.flatten()


@njit(parallel=True, fastmath=True)
def Laplace_dipole_operator_numba(r_vectors, field, weights, wall=0):
  ''' 
  Returns the product of the Laplace dipole operator to the concentration field on the particle's surface. Kernel in an unbounded 
  or half space domain.
  '''
  # Variables
  N = r_vectors.size // 3
  r_vectors = r_vectors.reshape(N, 3)
  res = np.zeros((N,3))
  norm_fact = -1.0 / (4.0 * np.pi)

  rx_vec = np.copy(r_vectors[:,0])
  ry_vec = np.copy(r_vectors[:,1])
  rz_vec = np.copy(r_vectors[:,2])
  resx_vec = np.zeros(N)
  resy_vec = np.zeros(N)
  resz_vec = np.zeros(N)

  for i in prange(N):
    rxi = rx_vec[i]
    ryi = ry_vec[i]
    rzi = rz_vec[i]

    for j in range(N):
      # Multiply field by quadrature weight   
      c = field[j] * weights[j]

      # Compute vector between particles i and j
      rx = rxi - r_vectors[j,0]
      ry = ryi - r_vectors[j,1]
      rz = rzi - r_vectors[j,2]

      # 1. Compute dipole  kernel for pair i-j, if i==j kernel = 0
      if i == j:
        Tx = 0.0          
        Ty = 0.0          
        Tz = 0.0          
      else:
        # Normalize distance with hydrodynamic radius
        r = np.sqrt(rx*rx + ry*ry + rz*rz)
              
        # TODO: We should not divide by zero 
        invr3 = 1.0 / r**3
        
        # Compute kernel
        Tx = rx * invr3
        Ty = ry * invr3
        Tz = rz * invr3

      if wall:
        # Normalize distance with hydrodynamic radius
        rz = rzi + r_vectors[j,2]
        r = np.sqrt(rx*rx + ry*ry + rz*rz)
              
        # TODO: We should not divide by zero 
        invr3 = 1.0 / r**3
        
        # Compute kernel
        Tx += rx * invr3
        Ty += ry * invr3
        Tz += rz * invr3
        
         
      # 2. Compute product T_i * c           
      resx_vec[i] += Tx * c  
      resy_vec[i] += Ty * c 
      resz_vec[i] += Tz * c 

    res[i,0] = resx_vec[i]
    res[i,1] = resy_vec[i]
    res[i,2] = resz_vec[i]

  return norm_fact * res.flatten()


@njit(parallel=True, fastmath=True)
def Laplace_single_layer_operator_source_target_numba(source, target, field, weights_source, wall=0):
  ''' 
  Returns the product of the Laplace single layer operator applied to source points on a particle surface and evaluated at target points in space.
  Kernel in an unbounded or half space domain.
  '''
  # Variables
  num_targets = target.size // 3 
  num_sources = source.size // 3 
  source = source.reshape(num_sources, 3)
  target = target.reshape(num_targets, 3)
  res = np.zeros(num_targets)
  norm_fact = 1.0 / (4.0 * np.pi)

  # Copy to one dimensional vectors
  rx_src = np.copy(source[:,0])
  ry_src = np.copy(source[:,1])
  rz_src = np.copy(source[:,2])
  rx_trg = np.copy(target[:,0])
  ry_trg = np.copy(target[:,1])
  rz_trg = np.copy(target[:,2])

  for i in prange(num_targets):
    rxi = rx_trg[i]
    ryi = ry_trg[i]
    rzi = rz_trg[i]

    for j in range(num_sources):
      # Multiply field by quadrature weight   
      c = field[j] * weights_source[j]
         
      # Compute vector between particles i and j
      rx = rxi - rx_src[j]
      ry = ryi - ry_src[j]
      rz = rzi - rz_src[j]

      # Compute distance
      r = np.sqrt(rx*rx + ry*ry + rz*rz)
                
      # 1. Compute single layer kernel for pair i-j, if i==j kernel = 0
      if r < 1e-12:
        S = 0       
   
      else:
              
        # TODO: We should not divide by zero 
        invr = 1.0 / r

        # Compute single layer kernel
        S = invr

      if wall:
        # Compute distance
        rz = rzi + rz_src[j]
        r = np.sqrt(rx*rx + ry*ry + rz*rz)
              
        # TODO: We should not divide by zero 
        invr = 1.0 / r

        # Compute single layer kernel
        S += invr

        
      # 2. Compute product S * c           
      res[i] += S * c

  return norm_fact * res


@njit(parallel=True, fastmath=True)
def Laplace_double_layer_operator_source_target_numba(source, target, field, weights_source, normals_source, wall=0):
  ''' 
  Returns the product of the Laplace double layer operator applied to source points on a particle surface and evaluated at target points in space.
  Kernel in an unbounded domain.
  '''
  # Variables
  num_targets = target.size // 3 
  num_sources = source.size // 3 
  source = source.reshape(num_sources, 3)
  target = target.reshape(num_targets, 3)
  normals_source = normals_source.reshape(num_sources, 3)
  res = np.zeros(num_targets)
  norm_fact = -1.0 / (4.0 * np.pi)

  # Copy to one dimensional vectors
  rx_src = np.copy(source[:,0])
  ry_src = np.copy(source[:,1])
  rz_src = np.copy(source[:,2])
  nx_src = np.copy(normals_source[:,0])
  ny_src = np.copy(normals_source[:,1])
  nz_src = np.copy(normals_source[:,2])
  rx_trg = np.copy(target[:,0])
  ry_trg = np.copy(target[:,1])
  rz_trg = np.copy(target[:,2])

  for i in prange(num_targets):
    rxi = rx_trg[i]
    ryi = ry_trg[i]
    rzi = rz_trg[i]

    for j in range(num_sources):
      # Multiply field by quadrature weight   
      c = field[j] * weights_source[j]
         
      # Compute vector between particles i and j
      rx = rxi - rx_src[j]
      ry = ryi - ry_src[j]
      rz = rzi - rz_src[j]
                
      nx = nx_src[j]
      ny = ny_src[j]
      nz = nz_src[j]

      # Compute distance
      r = np.sqrt(rx*rx + ry*ry + rz*rz)
               
      # 1. Compute double layer kernel for pair i-j, if i==j kernel = 0
      if r < 1e-12:
        T = 0       
 
      else:
             
        # TODO: We should not divide by zero 
        invr3 = 1.0 / r**3

        # Compute double  layer kernel T_i * n_i
        T =  invr3 * ( rx*nx + ry*ny + rz*nz )

      if wall:
        rz = rzi + rz_src[j]
        r = np.sqrt(rx*rx + ry*ry + rz*rz)
      
        # TODO: We should not divide by zero 
        invr3 = 1.0 / r**3

        # Compute double  layer kernel T_i * n_i
        T +=  invr3 * ( rx*nx + ry*ny + rz*nz )
        
           
      # 2. Compute product T_i * n_i * c           
      res[i] += T * c

  return norm_fact * res

