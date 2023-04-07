
import numpy as np
import scipy.sparse

# Try to import numba
try:
  from numba import njit, prange
except ImportError:
  print('numba not found')


@njit(parallel=True, fastmath=True)
def no_wall_laplace_single_layer_operator_numba(r_vectors, field, weights):
  ''' 
  Returns the product of the Laplace single layer operator to the concentration field on the particle's surface. Kernel in an unbounded domain.
  
  This function uses numba.
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
        # Normalize distance with hydrodynamic radius
        r2 = rx*rx + ry*ry + rz*rz
        r = np.sqrt(r2)
              
        # TODO: We should not divide by zero 
        invr = 1.0 / r

        # Compute single layer kernel
        S = invr
           
      # 2. Compute product S * c           
      res[i] += invr * c

  return norm_fact * res

@njit(parallel=True, fastmath=True)
def no_wall_laplace_double_layer_operator_numba(r_vectors, field, weights, normals):
  ''' 
  Returns the product of the Laplace double layer operator to the concentration field on the particle's surface. Kernel in an unbounded domain.
  
  This function uses numba.
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
        r2 = rx*rx + ry*ry + rz*rz
        r = np.sqrt(r2)
              
        # TODO: We should not divide by zero 
        invr3 = 1.0 / r**3

        # Compute double  layer kernel T_i * n_i
        T =  invr3 * ( rx*nx + ry*ny + rz*nz )
           
      # 2. Compute product T_i * n_i * c           
      res[i] += T * c

  return norm_fact * res


@njit(parallel=True, fastmath=True)
def no_wall_laplace_deriv_double_layer_operator_numba(r_vectors, field, weights, normals):
  ''' 
  Returns the product of the derivative of the Laplace double layer operator to the concentration field on the particle's surface. Kernel in an unbounded domain.
  
  This function uses numba.
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
         
      # Compute vector between particles i and j
      rx = rxi - r_vectors[j,0]
      ry = ryi - r_vectors[j,1]
      rz = rzi - r_vectors[j,2]

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
        # Normalize distance with hydrodynamic radius
        r2 = rx*rx + ry*ry + rz*rz
        r = np.sqrt(r2)
              
        # TODO: We should not divide by zero 
        invr = 1.0 / r
        invr2 = invr * invr
        invr3 = invr2 * invr

        # Compute  kernel
        c = -3.0 / invr2
        Lxx = (1.0 + c*rx*rx) * invr3
        Lxy = (      c*rx*ry) * invr3
        Lxz = (      c*rx*rz) * invr3
        Lyy = (1.0 + c*ry*ry) * invr3
        Lyz = (      c*ry*rz) * invr3

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
def no_wall_laplace_dipole_operator_numba(r_vectors, field, weights):
  ''' 
  Returns the product of the Laplace dipole operator without normals to the concentration field on the particle's surface. Kernel in an unbounded domain.
  
  This function uses numba.
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
        r2 = rx*rx + ry*ry + rz*rz
        r = np.sqrt(r2)
              
        # TODO: We should not divide by zero 
        invr3 = 1.0 / r**3
        
        # Compute kernel
        Tx = rx * invr3
        Ty = ry * invr3
        Tz = rz * invr3
         
      # 2. Compute product T_i * c           
      resx_vec[i] += Tx * c  
      resy_vec[i] += Ty * c 
      resz_vec[i] += Tz * c 

    res[i,0] = resx_vec[i]
    res[i,1] = resy_vec[i]
    res[i,2] = resz_vec[i]

  return norm_fact * res.flatten()

