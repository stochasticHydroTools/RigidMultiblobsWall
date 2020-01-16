from __future__ import division, print_function
import numpy as np
import scipy.sparse

# Try to import numba
try:
  from numba import njit, prange
except ImportError:
  print('numba not found')


@njit(parallel=True)
def no_wall_mobility_trans_times_force_numba(r_vectors, force, eta, a, L):
  ''' 
  Returns the product of the mobility at the blob level to the force 
  on the blobs. Mobility for particles in an unbounded domain, it uses
  the standard RPY tensor.  
  
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

  # Determine if the space is pseudo-periodic in any dimension
  # We use a extended unit cell of length L=3*(Lx, Ly, Lz)
  periodic_x = 0
  periodic_y = 0 
  periodic_z = 0
  if L[0] > 0:
    periodic_x = 1
  if L[1] > 0:
    periodic_y = 1
  if L[2] > 0:
    periodic_z = 1
  
  # Loop over image boxes and then over particles
  for i in prange(N):
    for boxX in range(-periodic_x, periodic_x+1):
      for boxY in range(-periodic_y, periodic_y+1):
        for boxZ in range(-periodic_z, periodic_z+1):
          for j in range(N):
	  
            # Compute vector between particles i and j
            rx = r_vectors[i,0] - r_vectors[j,0]
            ry = r_vectors[i,1] - r_vectors[j,1]
            rz = r_vectors[i,2] - r_vectors[j,2]

            # Project a vector r to the extended unit cell
            # centered around (0,0,0) and of size L=3*(Lx, Ly, Lz). If 
            # any dimension of L is equal or smaller than zero the 
            # box is assumed to be infinite in that direction.
            if L[0] > 0:
              rx = rx - int(rx / L[0] + 0.5 * (int(rx>0) - int(rx<0))) * L[0]
              rx = rx + boxX * L[0]
            if L[1] > 0:
              ry = ry - int(ry / L[1] + 0.5 * (int(ry>0) - int(ry<0))) * L[1]
              ry = ry + boxY * L[1]              
            if L[2] > 0:
              rz = rz - int(rz / L[2] + 0.5 * (int(rz>0) - int(rz<0))) * L[2]
              rz = rz + boxZ * L[2]            
               
            # 1. Compute mobility for pair i-j, if i==j use self-interation
            j_image = j
            if boxX != 0 or boxY != 0 or boxZ != 0:
              j_image = -1         

            if i == j_image:
              Mxx = fourOverThree
              Mxy = 0
              Mxz = 0
              Myy = Mxx
              Myz = 0
              Mzz = Mxx           
            else:
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
                c1 = 1.0 + 2.0 / (3.0 * r2)
                c2 = (1.0 - 2.0 * invr2) * invr2
                Mxx = (c1 + c2*rx*rx) * invr
                Mxy = (     c2*rx*ry) * invr
                Mxz = (     c2*rx*rz) * invr
                Myy = (c1 + c2*ry*ry) * invr
                Myz = (     c2*ry*rz) * invr
                Mzz = (c1 + c2*rz*rz) * invr 
              else:
                c1 = fourOverThree * (1.0 - 0.28125 * r) # 9/32 = 0.28125
                c2 = fourOverThree * 0.09375 * invr      # 3/32 = 0.09375
                Mxx = c1 + c2 * rx*rx 
                Mxy =      c2 * rx*ry 
                Mxz =      c2 * rx*rz 
                Myy = c1 + c2 * ry*ry 
                Myz =      c2 * ry*rz 
                Mzz = c1 + c2 * rz*rz 
                
            Myx = Mxy
            Mzx = Mxz
            Mzy = Myz
	  
            # 2. Compute product M_ij * F_j           
            u[i,0] += (Mxx * force[j,0] + Mxy * force[j,1] + Mxz * force[j,2]) * norm_fact_f
            u[i,1] += (Myx * force[j,0] + Myy * force[j,1] + Myz * force[j,2]) * norm_fact_f
            u[i,2] += (Mzx * force[j,0] + Mzy * force[j,1] + Mzz * force[j,2]) * norm_fact_f

  return u.flatten()


@njit(parallel=True, fastmath=True)
def single_wall_mobility_trans_times_force_numba(r_vectors, force, eta, a, L):
  ''' 
  Returns the product of the mobility at the blob level to the force 
  on the blobs. Mobility for particles above a single wall.  
  
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

  # Determine if the space is pseudo-periodic in any dimension
  # We use a extended unit cell of length L=3*(Lx, Ly, Lz)
  periodic_x = 0
  periodic_y = 0
  periodic_z = 0 
  Lx = L[0]
  Ly = L[1]
  Lz = L[2]
  if Lx > 0:
    periodic_x = 1
  if Ly > 0:
    periodic_y = 1
  if Lz > 0:
    periodic_z = 1
    
  rx_vec = np.copy(r_vectors[:,0])
  ry_vec = np.copy(r_vectors[:,1])
  rz_vec = np.copy(r_vectors[:,2])
  fx_vec = np.copy(force[:,0])
  fy_vec = np.copy(force[:,1])
  fz_vec = np.copy(force[:,2])
  ux_vec = np.zeros(N)
  uy_vec = np.zeros(N)
  uz_vec = np.zeros(N)
    
  # Loop over image boxes and then over particles
  for i in prange(N):
    rxi = rx_vec[i]
    ryi = ry_vec[i]
    rzi = rz_vec[i]
    for boxX in range(-periodic_x, periodic_x+1):
      for boxY in range(-periodic_y, periodic_y+1):
        for boxZ in range(-periodic_z, periodic_z+1):
          for j in range(N):
	  
            # Compute vector between particles i and j
            rx = rxi - rx_vec[j]
            ry = ryi - ry_vec[j]
            rz = rzi - rz_vec[j]

            # Project a vector r to the extended unit cell
            # centered around (0,0,0) and of size L=3*(Lx, Ly, Lz). If 
            # any dimension of L is equal or smaller than zero the 
            # box is assumed to be infinite in that direction.
            if Lx > 0:
              rx = rx - int(rx / Lx + 0.5 * (int(rx>0) - int(rx<0))) * Lx
              rx = rx + boxX * Lx
            if Ly > 0:
              ry = ry - int(ry / Ly + 0.5 * (int(ry>0) - int(ry<0))) * Ly
              ry = ry + boxY * Ly 
            if Lz > 0:
              rz = rz - int(rz / Lz + 0.5 * (int(rz>0) - int(rz<0))) * Lz
              rz = rz + boxZ * Lz            
               
            # 1. Compute mobility for pair i-j, if i==j use self-interation
            j_image = j
            if boxX != 0 or boxY != 0 or boxZ != 0:
              j_image = -1           

            rx = rx * inva 
            ry = ry * inva
            rz = rz * inva
            if i == j_image:
              Mxx = fourOverThree
              Mxy = 0
              Mxz = 0
              Myy = Mxx
              Myz = 0
              Mzz = Mxx           
            else:
              # Normalize distance with hydrodynamic radius
              r2 = rx*rx + ry*ry + rz*rz
              r = np.sqrt(r2)
              
              # TODO: We should not divide by zero 
              invr = 1.0 / r
              invr2 = invr * invr

              if r > 2:
                c1 = 1.0 + 2.0 / (3.0 * r2)
                c2 = (1.0 - 2.0 * invr2) * invr2
                Mxx = (c1 + c2*rx*rx) * invr
                Mxy = (     c2*rx*ry) * invr
                Mxz = (     c2*rx*rz) * invr
                Myy = (c1 + c2*ry*ry) * invr
                Myz = (     c2*ry*rz) * invr
                Mzz = (c1 + c2*rz*rz) * invr 
              else:
                c1 = fourOverThree * (1.0 - 0.28125 * r) # 9/32 = 0.28125
                c2 = fourOverThree * 0.09375 * invr      # 3/32 = 0.09375
                Mxx = c1 + c2 * rx*rx 
                Mxy =      c2 * rx*ry 
                Mxz =      c2 * rx*rz 
                Myy = c1 + c2 * ry*ry 
                Myz =      c2 * ry*rz 
                Mzz = c1 + c2 * rz*rz 
                
            Myx = Mxy
            Mzx = Mxz
            Mzy = Myz

            # Wall correction
            rz = (rzi + rz_vec[j]) * inva
            hj = rz_vec[j] * inva

            if i == j_image:
              invZi = 1.0 / hj
              invZi3 = invZi * invZi * invZi
              invZi5 = invZi3 * invZi * invZi
            
              Mxx += -(9.0 * invZi - 2.0 * invZi3 + invZi5 ) / 12.0
              Myy += -(9.0 * invZi - 2.0 * invZi3 + invZi5 ) / 12.0
              Mzz += -(9.0 * invZi - 4.0 * invZi3 + invZi5 ) / 6.0   
            else:
              h_hat = hj / rz
              invR = 1.0 / np.sqrt(rx*rx + ry*ry + rz*rz)
              ex = rx * invR
              ey = ry * invR
              ez = rz * invR
              invR3 = invR * invR * invR
              invR5 = invR3 * invR * invR
                  
              fact1 = -(3.0*(1.0+2.0*h_hat*(1.0-h_hat)*ez*ez) * invR + 2.0*(1.0-3.0*ez*ez) * invR3 - 2.0*(1.0-5.0*ez*ez) * invR5)  / 3.0
              fact2 = -(3.0*(1.0-6.0*h_hat*(1.0-h_hat)*ez*ez) * invR - 6.0*(1.0-5.0*ez*ez) * invR3 + 10.0*(1.0-7.0*ez*ez) * invR5) / 3.0
              fact3 =  ez * (3.0*h_hat*(1.0-6.0*(1.0-h_hat)*ez*ez) * invR - 6.0*(1.0-5.0*ez*ez) * invR3 + 10.0*(2.0-7.0*ez*ez) * invR5) * 2.0 / 3.0
              fact4 =  ez * (3.0*h_hat*invR - 10.0*invR5) * 2.0 / 3.0
              fact5 = -(3.0*h_hat*h_hat*ez*ez*invR + 3.0*ez*ez*invR3 + (2.0-15.0*ez*ez)*invR5) * 4.0 / 3.0
    
              Mxx += fact1 + fact2 * ex*ex
              Mxy += fact2 * ex*ey
              Mxz += fact2 * ex*ez + fact3 * ex
              Myx += fact2 * ey*ex
              Myy += fact1 + fact2 * ey*ey
              Myz += fact2 * ey*ez + fact3 * ey
              Mzx += fact2 * ez*ex + fact4 * ex
              Mzy += fact2 * ez*ey + fact4 * ey
              Mzz += fact1 + fact2 * ez*ez + fact3 * ez + fact4 * ez + fact5
	  
            # 2. Compute product M_ij * F_j           
            ux_vec[i] += (Mxx * fx_vec[j] + Mxy * fy_vec[j] + Mxz * fz_vec[j]) * norm_fact_f
            uy_vec[i] += (Myx * fx_vec[j] + Myy * fy_vec[j] + Myz * fz_vec[j]) * norm_fact_f
            uz_vec[i] += (Mzx * fx_vec[j] + Mzy * fy_vec[j] + Mzz * fz_vec[j]) * norm_fact_f

    u[i,0] = ux_vec[i]
    u[i,1] = uy_vec[i]
    u[i,2] = uz_vec[i]

  return u.flatten()


@njit(parallel=True)
def in_plane_mobility_trans_times_force_numba(r_vectors, force, eta, a, L):
  ''' 
  Returns the product of the mobility at the blob level to the force 
  on the blobs. Mobility for particles in a fixed plane, it uses
  the standard RPY tensor.  
  
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

  # Determine if the space is pseudo-periodic in any dimension
  # We use a extended unit cell of length L=3*(Lx, Ly, Lz)
  periodic_x = 0
  periodic_y = 0
  periodic_z = 0
  if L[0] > 0:
    periodic_x = 1
  if L[1] > 0:
    periodic_y = 1
  if L[2] > 0:
    periodic_z = 1

  # Loop over image boxes and then over particles
  for i in prange(N):
    for boxX in range(-periodic_x, periodic_x+1):
      for boxY in range(-periodic_y, periodic_y+1):
        for boxZ in range(-periodic_z, periodic_z+1):
          for j in range(N):
	  
            # Compute vector between particles i and j
            rx = r_vectors[i,0] - r_vectors[j,0]
            ry = r_vectors[i,1] - r_vectors[j,1]
            rz = r_vectors[i,2] - r_vectors[j,2]

            # Project a vector r to the extended unit cell
            # centered around (0,0,0) and of size L=3*(Lx, Ly, Lz). If 
            # any dimension of L is equal or smaller than zero the 
            # box is assumed to be infinite in that direction.
            if L[0] > 0:
              rx = rx - int(rx / L[0] + 0.5 * (int(rx>0) - int(rx<0))) * L[0]
              rx = rx + boxX * L[0]
            if L[1] > 0:
              ry = ry - int(ry / L[1] + 0.5 * (int(ry>0) - int(ry<0))) * L[1]
              ry = ry + boxY * L[1]              
            if L[2] > 0:
              rz = rz - int(rz / L[2] + 0.5 * (int(rz>0) - int(rz<0))) * L[2]
              rz = rz + boxZ * L[2]            
               
            # 1. Compute mobility for pair i-j, if i==j use self-interation
            j_image = j
            if boxX != 0 or boxY != 0 or boxZ != 0:
              j_image = -1           

            rx = rx * inva 
            ry = ry * inva
            rz = rz * inva
            if i == j_image:
              Mxx = fourOverThree
              Mxy = 0
              Mxz = 0
              Myy = Mxx
              Myz = 0
              Mzz = Mxx           
            else:
              # Normalize distance with hydrodynamic radius
              r2 = rx*rx + ry*ry + rz*rz
              r = np.sqrt(r2)
              
              # TODO: We should not divide by zero 
              invr = 1.0 / r
              invr2 = invr * invr

              if r > 2:
                c1 = 1.0 + 2.0 / (3.0 * r2)
                c2 = (1.0 - 2.0 * invr2) * invr2
                Mxx = (c1 + c2*rx*rx) * invr
                Mxy = (     c2*rx*ry) * invr
                Mxz = 0.0
                Myy = (c1 + c2*ry*ry) * invr
                Myz = 0.0
                Mzz = 0.0
              else:
                c1 = fourOverThree * (1.0 - 0.28125 * r) # 9/32 = 0.28125
                c2 = fourOverThree * 0.09375 * invr      # 3/32 = 0.09375
                Mxx = c1 + c2 * rx*rx 
                Mxy =      c2 * rx*ry 
                Mxz = 0.0
                Myy = c1 + c2 * ry*ry 
                Myz = 0.0 
                Mzz = 0.0 
                
            Myx = Mxy
            Mzx = Mxz
            Mzy = Myz

            # Wall correction
            rz = (r_vectors[i,2] + r_vectors[j,2]) / a
            hj = r_vectors[j,2] / a

            if i == j_image:
              invZi = 1.0 / hj
              invZi3 = invZi * invZi * invZi
              invZi5 = invZi3 * invZi * invZi
            
              Mxx += -(9.0 * invZi - 2.0 * invZi3 + invZi5 ) / 12.0
              Myy += -(9.0 * invZi - 2.0 * invZi3 + invZi5 ) / 12.0
              Mzz += 0.0;   
            else:
              h_hat = hj / rz
              invR = 1.0 / np.sqrt(rx*rx + ry*ry + rz*rz)
              ex = rx * invR
              ey = ry * invR
              ez = rz * invR
              invR3 = invR * invR * invR
              invR5 = invR3 * invR * invR
    
              fact1 = -(3.0*(1.0+2.0*h_hat*(1.0-h_hat)*ez*ez) * invR + 2.0*(1.0-3.0*ez*ez) * invR3 - 2.0*(1.0-5.0*ez*ez) * invR5)  / 3.0
              fact2 = -(3.0*(1.0-6.0*h_hat*(1.0-h_hat)*ez*ez) * invR - 6.0*(1.0-5.0*ez*ez) * invR3 + 10.0*(1.0-7.0*ez*ez) * invR5) / 3.0
              fact3 =  ez * (3.0*h_hat*(1.0-6.0*(1.0-h_hat)*ez*ez) * invR - 6.0*(1.0-5.0*ez*ez) * invR3 + 10.0*(2.0-7.0*ez*ez) * invR5) * 2.0 / 3.0
              fact4 =  ez * (3.0*h_hat*invR - 10.0*invR5) * 2.0 / 3.0
              fact5 = -(3.0*h_hat*h_hat*ez*ez*invR + 3.0*ez*ez*invR3 + (2.0-15.0*ez*ez)*invR5) * 4.0 / 3.0
    
              Mxx += fact1 + fact2 * ex*ex
              Mxy += fact2 * ex*ey
              Mxz += 0.0
              Myx += fact2 * ey*ex
              Myy += fact1 + fact2 * ey*ey
              Myz += 0.0
              Mzx += 0.0
              Mzy += 0.0
              Mzz += 0.0
	  
            # 2. Compute product M_ij * F_j           
            u[i,0] += (Mxx * force[j,0] + Mxy * force[j,1]) * norm_fact_f
            u[i,1] += (Myx * force[j,0] + Myy * force[j,1]) * norm_fact_f
            u[i,2] += 0.0

  return u.flatten()



@njit(parallel=True)
def no_wall_mobility_trans_times_torque_numba(r_vectors, torque, eta, a, L):
  ''' 
  Returns the product of the mobility translation-rotation at the blob level to the torque 
  on the blobs. Mobility for particles in an unbounded domain, it uses
  the standard RPY tensor.  
  
  This function uses numba.
  '''
  # Variables
  N = r_vectors.size // 3
  r_vectors = r_vectors.reshape(N, 3)
  torque = torque.reshape(N, 3)
  u = np.zeros((N, 3))
  fourOverThree = 4.0 / 3.0
  inva = 1.0 / a
  norm_fact_f = 1.0 / (8.0 * np.pi * eta * a**2)

  # Determine if the space is pseudo-periodic in any dimension
  # We use a extended unit cell of length L=3*(Lx, Ly, Lz)
  periodic_x = 0
  periodic_y = 0 
  periodic_z = 0
  if L[0] > 0:
    periodic_x = 1
  if L[1] > 0:
    periodic_y = 1
  if L[2] > 0:
    periodic_z = 1
  
  # Loop over image boxes and then over particles
  for i in prange(N):
    for boxX in range(-periodic_x, periodic_x+1):
      for boxY in range(-periodic_y, periodic_y+1):
        for boxZ in range(-periodic_z, periodic_z+1):
          for j in range(N):
            
            # Compute vector between particles i and j
            rx = r_vectors[i,0] - r_vectors[j,0]
            ry = r_vectors[i,1] - r_vectors[j,1]
            rz = r_vectors[i,2] - r_vectors[j,2]

            # Project a vector r to the extended unit cell
            # centered around (0,0,0) and of size L=3*(Lx, Ly, Lz). If 
            # any dimension of L is equal or smaller than zero the 
            # box is assumed to be infinite in that direction.
            if L[0] > 0:
              rx = rx - int(rx / L[0] + 0.5 * (int(rx>0) - int(rx<0))) * L[0]
              rx = rx + boxX * L[0]
            if L[1] > 0:
              ry = ry - int(ry / L[1] + 0.5 * (int(ry>0) - int(ry<0))) * L[1]
              ry = ry + boxY * L[1]              
            if L[2] > 0:
              rz = rz - int(rz / L[2] + 0.5 * (int(rz>0) - int(rz<0))) * L[2]
              rz = rz + boxZ * L[2]            
               
            # 1. Compute mobility for pair i-j, if i==j use self-interation
            j_image = j
            if boxX != 0 or boxY != 0 or boxZ != 0:
              j_image = -1           

            # 1. Compute UT mobility for pair i-j
            # mobilityUTRPY(rx,ry,rz, Mxx,Mxy,Mxz,Myy,Myz,Mzz, i,j_image, invaGPU);
            if i == j_image:
              Mxx = 0
              Mxy = 0
              Mxz = 0
              Myy = 0
              Myz = 0
              Mzz = 0
            else:
              # Normalize distance with hydrodynamic radius
              rx = rx * inva 
              ry = ry * inva
              rz = rz * inva
              r2 = rx*rx + ry*ry + rz*rz
              r = np.sqrt(r2)
              r3 = r2*r
              # TODO: We should not divide by zero 
              invr3 = 1.0 / r3
              if r >= 2:
                Mxx =  0
                Mxy =  rz * invr3
                Mxz = -ry * invr3
                Myy =  0
                Myz =  rx * invr3
                Mzz =  0
              else:
                c1 = 0.5 * (1.0 - 0.375 * r) # 3/8 = 0.375
                Mxx =  0
                Mxy =  c1 * rz
                Mxz = -c1 * ry 
                Myy =  0
                Myz =  c1 * rx
                Mzz =  0

            Myx = -Mxy
            Mzx = -Mxz
            Mzy = -Myz
	  
            # 2. Compute product M_ij * T_j
            u[i,0] += (Mxx * torque[j,0] + Mxy * torque[j,1] + Mxz * torque[j,2]) * norm_fact_f
            u[i,1] += (Myx * torque[j,0] + Myy * torque[j,1] + Myz * torque[j,2]) * norm_fact_f
            u[i,2] += (Mzx * torque[j,0] + Mzy * torque[j,1] + Mzz * torque[j,2]) * norm_fact_f

  return u.flatten()


@njit(parallel=True)
def single_wall_mobility_trans_times_torque_numba(r_vectors, torque, eta, a, L):
  ''' 
  Returns the product of the mobility translation-rotation at the blob level to the torque 
  on the blobs. Mobility for particles on top of an infinite wall.  
  
  This function uses numba.
  '''
  # Variables
  N = r_vectors.size // 3
  r_vectors = r_vectors.reshape(N, 3)
  torque = torque.reshape(N, 3)
  u = np.zeros((N, 3))
  fourOverThree = 4.0 / 3.0
  inva = 1.0 / a
  norm_fact_f = 1.0 / (8.0 * np.pi * eta * a**2)

  # Determine if the space is pseudo-periodic in any dimension
  # We use a extended unit cell of length L=3*(Lx, Ly, Lz)
  periodic_x = 0
  periodic_y = 0 
  periodic_z = 0
  if L[0] > 0:
    periodic_x = 1
  if L[1] > 0:
    periodic_y = 1
  if L[2] > 0:
    periodic_z = 1
  
  # Loop over image boxes and then over particles
  for i in prange(N):
    for boxX in range(-periodic_x, periodic_x+1):
      for boxY in range(-periodic_y, periodic_y+1):
        for boxZ in range(-periodic_z, periodic_z+1):
          for j in range(N):
            
            # Compute vector between particles i and j
            rx = r_vectors[i,0] - r_vectors[j,0]
            ry = r_vectors[i,1] - r_vectors[j,1]
            rz = r_vectors[i,2] - r_vectors[j,2]

            # Project a vector r to the extended unit cell
            # centered around (0,0,0) and of size L=3*(Lx, Ly, Lz). If 
            # any dimension of L is equal or smaller than zero the 
            # box is assumed to be infinite in that direction.
            if L[0] > 0:
              rx = rx - int(rx / L[0] + 0.5 * (int(rx>0) - int(rx<0))) * L[0]
              rx = rx + boxX * L[0]
            if L[1] > 0:
              ry = ry - int(ry / L[1] + 0.5 * (int(ry>0) - int(ry<0))) * L[1]
              ry = ry + boxY * L[1]              
            if L[2] > 0:
              rz = rz - int(rz / L[2] + 0.5 * (int(rz>0) - int(rz<0))) * L[2]
              rz = rz + boxZ * L[2]            
               
            # 1. Compute mobility for pair i-j, if i==j use self-interation
            j_image = j
            if boxX != 0 or boxY != 0 or boxZ != 0:
              j_image = -1           

            # 1. Compute UT mobility for pair i-j
            # mobilityUTRPY(rx,ry,rz, Mxx,Mxy,Mxz,Myy,Myz,Mzz, i,j_image, invaGPU);
            rx = rx * inva 
            ry = ry * inva
            rz = rz * inva
            if i == j_image:
              Mxx = 0
              Mxy = 0
              Mxz = 0
              Myy = 0
              Myz = 0
              Mzz = 0
            else:
              # Normalize distance with hydrodynamic radius
              r2 = rx*rx + ry*ry + rz*rz
              r = np.sqrt(r2)
              r3 = r2*r
              # TODO: We should not divide by zero 
              invr3 = 1.0 / r3
              if r >= 2:
                Mxx =  0
                Mxy =  rz * invr3
                Mxz = -ry * invr3
                Myy =  0
                Myz =  rx * invr3
                Mzz =  0
              else:
                c1 = 0.5 * (1.0 - 0.375 * r) # 3/8 = 0.375
                Mxx =  0
                Mxy =  c1 * rz
                Mxz = -c1 * ry 
                Myy =  0
                Myz =  c1 * rx
                Mzz =  0

            Myx = -Mxy
            Mzx = -Mxz
            Mzy = -Myz

            # Wall correction
            # mobilityUTSingleWallCorrection(-rx/a, -ry/a, (-rz+2*x[ioffset+2])/a, Mxx,Mxy,Mxz,Myx,Myy,Myz,Mzx,Mzy, j_image,i, invaGPU, x[ioffset+2]/a);
            rx = -rx
            ry = -ry
            rz = (r_vectors[i,2] + r_vectors[j,2]) * inva
            hj = r_vectors[i,2] * inva

            if i == j_image:
              invZi = 1.0 / hj
              invZi4 = invZi**4
              Mxy -= - invZi4 * 0.125 # 3/24 = 0.125
              Myx -=   invZi4 * 0.125 # 3/24 = 0.125
            else:
              h_hat = hj / rz
              invR = 1.0 / np.sqrt(rx*rx + ry*ry + rz*rz)
              invR2 = invR * invR
              invR4 = invR2 * invR2
              ex = rx * invR
              ey = ry * invR
              ez = rz * invR
    
              fact1 =  invR2
              fact2 = (6.0 * h_hat * ez*ez * invR2 + (1.0-10.0 * ez*ez) * invR4) * 2.0
              fact3 = -ez * (3.0 * h_hat * invR2 - 5.0 * invR4) * 2.0
              fact4 = -ez * (h_hat * invR2 - invR4) * 2.0
    
              Mxx -=                       - fact3*ex*ey        
              Mxy -= - fact1*ez            + fact3*ex*ex - fact4
              Mxz -=   fact1*ey                                 
              Myx -=   fact1*ez            - fact3*ey*ey + fact4
              Myy -=                         fact3*ex*ey        
              Myz -= - fact1*ex                                 
              Mzx -= - fact1*ey - fact2*ey - fact3*ey*ez        
              Mzy -=   fact1*ex + fact2*ex + fact3*ex*ez                 
	  
            # 2. Compute product M_ij * T_j
            u[i,0] += (Mxx * torque[j,0] + Mxy * torque[j,1] + Mxz * torque[j,2]) * norm_fact_f
            u[i,1] += (Myx * torque[j,0] + Myy * torque[j,1] + Myz * torque[j,2]) * norm_fact_f
            u[i,2] += (Mzx * torque[j,0] + Mzy * torque[j,1] + Mzz * torque[j,2]) * norm_fact_f

  return u.flatten()


@njit(parallel=True)
def in_plane_mobility_trans_times_torque_numba(r_vectors, torque, eta, a, L):
  ''' 
  Returns the product of the mobility translation-rotation at the blob level to the torque 
  on the blobs. Mobility for particles on top of an infinite wall fixed in plane.  
  
  This function uses numba.
  '''
  # Variables
  N = r_vectors.size // 3
  r_vectors = r_vectors.reshape(N, 3)
  torque = torque.reshape(N, 3)
  u = np.zeros((N, 3))
  fourOverThree = 4.0 / 3.0
  inva = 1.0 / a
  norm_fact_f = 1.0 / (8.0 * np.pi * eta * a**2)

  # Determine if the space is pseudo-periodic in any dimension
  # We use a extended unit cell of length L=3*(Lx, Ly, Lz)
  periodic_x = 0
  periodic_y = 0 
  periodic_z = 0
  if L[0] > 0:
    periodic_x = 1
  if L[1] > 0:
    periodic_y = 1
  if L[2] > 0:
    periodic_z = 1
  
  # Loop over image boxes and then over particles
  for i in prange(N):
    for boxX in range(-periodic_x, periodic_x+1):
      for boxY in range(-periodic_y, periodic_y+1):
        for boxZ in range(-periodic_z, periodic_z+1):
          for j in range(N):
            
            # Compute vector between particles i and j
            rx = r_vectors[i,0] - r_vectors[j,0]
            ry = r_vectors[i,1] - r_vectors[j,1]
            rz = r_vectors[i,2] - r_vectors[j,2]

            # Project a vector r to the extended unit cell
            # centered around (0,0,0) and of size L=3*(Lx, Ly, Lz). If 
            # any dimension of L is equal or smaller than zero the 
            # box is assumed to be infinite in that direction.
            if L[0] > 0:
              rx = rx - int(rx / L[0] + 0.5 * (int(rx>0) - int(rx<0))) * L[0]
              rx = rx + boxX * L[0]
            if L[1] > 0:
              ry = ry - int(ry / L[1] + 0.5 * (int(ry>0) - int(ry<0))) * L[1]
              ry = ry + boxY * L[1]              
            if L[2] > 0:
              rz = rz - int(rz / L[2] + 0.5 * (int(rz>0) - int(rz<0))) * L[2]
              rz = rz + boxZ * L[2]            
               
            # 1. Compute mobility for pair i-j, if i==j use self-interation
            j_image = j
            if boxX != 0 or boxY != 0 or boxZ != 0:
              j_image = -1           

            # 1. Compute UT mobility for pair i-j
            # mobilityUTRPY(rx,ry,rz, Mxx,Mxy,Mxz,Myy,Myz,Mzz, i,j_image, invaGPU);
            rx = rx * inva 
            ry = ry * inva
            rz = rz * inva
            if i == j_image:
              Mxx = 0
              Mxy = 0
              Mxz = 0
              Myy = 0
              Myz = 0
              Mzz = 0
            else:
              # Normalize distance with hydrodynamic radius
              r2 = rx*rx + ry*ry + rz*rz
              r = np.sqrt(r2)
              r3 = r2*r
              # TODO: We should not divide by zero 
              invr3 = 1.0 / r3
              if r >= 2:
                Mxx =  0
                Mxy =  rz * invr3
                Mxz =  0
                Myy =  0
                Myz =  0
                Mzz =  0
              else:
                c1 = 0.5 * (1.0 - 0.375 * r) # 3/8 = 0.375
                Mxx =  0
                Mxy =  c1 * rz
                Mxz =  0
                Myy =  0
                Myz =  0
                Mzz =  0

            Myx = -Mxy
            Mzx = -Mxz
            Mzy = -Myz

            # Wall correction
            # mobilityUTSingleWallCorrection(-rx/a, -ry/a, (-rz+2*x[ioffset+2])/a, Mxx,Mxy,Mxz,Myx,Myy,Myz,Mzx,Mzy, j_image,i, invaGPU, x[ioffset+2]/a);
            rx = -rx
            ry = -ry
            rz = (r_vectors[i,2] + r_vectors[j,2]) * inva
            hj = r_vectors[i,2] * inva

            if i == j_image:
              invZi = 1.0 / hj
              invZi4 = invZi**4
              Mxy -= - invZi4 * 0.125 # 3/24 = 0.125
              Myx -=   invZi4 * 0.125 # 3/24 = 0.125
            else:
              h_hat = hj / rz
              invR = 1.0 / np.sqrt(rx*rx + ry*ry + rz*rz)
              invR2 = invR * invR
              invR4 = invR2 * invR2
              ex = rx * invR
              ey = ry * invR
              ez = rz * invR
    
              fact1 =  invR2
              fact2 = (6.0 * h_hat * ez*ez * invR2 + (1.0-10.0 * ez*ez) * invR4) * 2.0
              fact3 = -ez * (3.0 * h_hat * invR2 - 5.0 * invR4) * 2.0
              fact4 = -ez * (h_hat * invR2 - invR4) * 2.0
    
              Mxx -=                       - fact3*ex*ey        
              Mxy -= - fact1*ez            + fact3*ex*ex - fact4
              Mxz -=   0                                 
              Myx -=   fact1*ez            - fact3*ey*ey + fact4
              Myy -=                         fact3*ex*ey        
              Myz -=   0                                 
              Mzx -=   0        
              Mzy -=   0                
	  
            # 2. Compute product M_ij * T_j
            u[i,0] += (Mxx * torque[j,0] + Mxy * torque[j,1]) * norm_fact_f
            u[i,1] += (Myx * torque[j,0] + Myy * torque[j,1]) * norm_fact_f
            u[i,2] += 0

  return u.flatten()


@njit(parallel=True)
def no_wall_mobility_rot_times_force_numba(r_vectors, force, eta, a, L):
  ''' 
  Returns the product of the mobility translation-rotation at the blob level to the torque 
  on the blobs. Mobility for particles in an unbounded domain, it uses
  the standard RPY tensor.  
  
  This function uses numba.
  '''
  # Variables
  N = r_vectors.size // 3
  r_vectors = r_vectors.reshape(N, 3)
  force = force.reshape(N, 3)
  u = np.zeros((N, 3))
  fourOverThree = 4.0 / 3.0
  inva = 1.0 / a
  norm_fact_f = 1.0 / (8.0 * np.pi * eta * a**2)

  # Determine if the space is pseudo-periodic in any dimension
  # We use a extended unit cell of length L=3*(Lx, Ly, Lz)
  periodic_x = 0
  periodic_y = 0 
  periodic_z = 0
  if L[0] > 0:
    periodic_x = 1
  if L[1] > 0:
    periodic_y = 1
  if L[2] > 0:
    periodic_z = 1
  
  # Loop over image boxes and then over particles
  for i in prange(N):
    for boxX in range(-periodic_x, periodic_x+1):
      for boxY in range(-periodic_y, periodic_y+1):
        for boxZ in range(-periodic_z, periodic_z+1):
          for j in range(N):
            
            # Compute vector between particles i and j
            rx = r_vectors[i,0] - r_vectors[j,0]
            ry = r_vectors[i,1] - r_vectors[j,1]
            rz = r_vectors[i,2] - r_vectors[j,2]

            # Project a vector r to the extended unit cell
            # centered around (0,0,0) and of size L=3*(Lx, Ly, Lz). If 
            # any dimension of L is equal or smaller than zero the 
            # box is assumed to be infinite in that direction.
            if L[0] > 0:
              rx = rx - int(rx / L[0] + 0.5 * (int(rx>0) - int(rx<0))) * L[0]
              rx = rx + boxX * L[0]
            if L[1] > 0:
              ry = ry - int(ry / L[1] + 0.5 * (int(ry>0) - int(ry<0))) * L[1]
              ry = ry + boxY * L[1]              
            if L[2] > 0:
              rz = rz - int(rz / L[2] + 0.5 * (int(rz>0) - int(rz<0))) * L[2]
              rz = rz + boxZ * L[2]            
               
            # 1. Compute mobility for pair i-j, if i==j use self-interation
            j_image = j
            if boxX != 0 or boxY != 0 or boxZ != 0:
              j_image = -1           

            if i==j_image:
              Mxx = 0
              Mxy = 0
              Mxz = 0
              Myy = 0
              Myz = 0
              Mzz = 0
            else:
              # Normalize distance with hydrodynamic radius
              rx = rx * inva
              ry = ry * inva
              rz = rz * inva
              r2 = rx*rx + ry*ry + rz*rz
              r = np.sqrt(r2)
              r3 = r2*r
              # TODO: We should not divide by zero 
              invr3 = 1.0 / r3
              if r >= 2.0:
                Mxx =  0
                Mxy =  rz * invr3
                Mxz = -ry * invr3
                Myy =  0
                Myz =  rx * invr3
                Mzz =  0
              else:
                c1 =  0.5 * (1.0 - 0.375 * r) # 3/8 = 0.375
                Mxx =  0
                Mxy =  c1 * rz
                Mxz = -c1 * ry 
                Myy =  0
                Myz =  c1 * rx
                Mzz =  0

            Myx = -Mxy
            Mzx = -Mxz
            Mzy = -Myz
	  
            # 2. Compute product M_ij * T_j
            u[i,0] += (Mxx * force[j,0] + Mxy * force[j,1] + Mxz * force[j,2]) * norm_fact_f
            u[i,1] += (Myx * force[j,0] + Myy * force[j,1] + Myz * force[j,2]) * norm_fact_f
            u[i,2] += (Mzx * force[j,0] + Mzy * force[j,1] + Mzz * force[j,2]) * norm_fact_f

  return u.flatten()


@njit(parallel=True)
def single_wall_mobility_rot_times_force_numba(r_vectors, force, eta, a, L):
  ''' 
  Returns the product of the mobility translation-rotation at the blob level to the torque 
  on the blobs. Mobility for particles in an unbounded domain, it uses
  the standard RPY tensor.  
  
  This function uses numba.
  '''
  # Variables
  N = r_vectors.size // 3
  r_vectors = r_vectors.reshape(N, 3)
  force = force.reshape(N, 3)
  u = np.zeros((N, 3))
  fourOverThree = 4.0 / 3.0
  inva = 1.0 / a
  norm_fact_f = 1.0 / (8.0 * np.pi * eta * a**2)

  # Determine if the space is pseudo-periodic in any dimension
  # We use a extended unit cell of length L=3*(Lx, Ly, Lz)
  periodic_x = 0
  periodic_y = 0 
  periodic_z = 0
  if L[0] > 0:
    periodic_x = 1
  if L[1] > 0:
    periodic_y = 1
  if L[2] > 0:
    periodic_z = 1
  
  # Loop over image boxes and then over particles
  for i in prange(N):
    for boxX in range(-periodic_x, periodic_x+1):
      for boxY in range(-periodic_y, periodic_y+1):
        for boxZ in range(-periodic_z, periodic_z+1):
          for j in range(N):
            
            # Compute vector between particles i and j
            rx = r_vectors[i,0] - r_vectors[j,0]
            ry = r_vectors[i,1] - r_vectors[j,1]
            rz = r_vectors[i,2] - r_vectors[j,2]

            # Project a vector r to the extended unit cell
            # centered around (0,0,0) and of size L=3*(Lx, Ly, Lz). If 
            # any dimension of L is equal or smaller than zero the 
            # box is assumed to be infinite in that direction.
            if L[0] > 0:
              rx = rx - int(rx / L[0] + 0.5 * (int(rx>0) - int(rx<0))) * L[0]
              rx = rx + boxX * L[0]
            if L[1] > 0:
              ry = ry - int(ry / L[1] + 0.5 * (int(ry>0) - int(ry<0))) * L[1]
              ry = ry + boxY * L[1]             
            if L[2] > 0:
              rz = rz - int(rz / L[2] + 0.5 * (int(rz>0) - int(rz<0))) * L[2]
              rz = rz + boxZ * L[2]            
               
            # 1. Compute mobility for pair i-j, if i==j use self-interation
            j_image = j
            if boxX != 0 or boxY != 0 or boxZ != 0:
              j_image = -1           

            rx = rx * inva
            ry = ry * inva
            rz = rz * inva
            if i==j_image:
              Mxx = 0
              Mxy = 0
              Mxz = 0
              Myy = 0
              Myz = 0
              Mzz = 0
            else:
              # Normalize distance with hydrodynamic radius
              r2 = rx*rx + ry*ry + rz*rz
              r = np.sqrt(r2)
              r3 = r2*r
              # TODO: We should not divide by zero 
              invr3 = 1.0 / r3
              if r >= 2.0:
                Mxx =  0
                Mxy =  rz * invr3
                Mxz = -ry * invr3
                Myy =  0
                Myz =  rx * invr3
                Mzz =  0
              else:
                c1 =  0.5 * (1.0 - 0.375 * r) # 3/8 = 0.375
                Mxx =  0
                Mxy =  c1 * rz
                Mxz = -c1 * ry 
                Myy =  0
                Myz =  c1 * rx
                Mzz =  0

            Myx = -Mxy
            Mzx = -Mxz
            Mzy = -Myz
	  
            # Wall correction
            # mobilityWFSingleWallCorrection(rx/a, ry/a, (rz+2*x[joffset+2])/a, Mxx,Mxy,Mxz,Myx,Myy,Myz,Mzx,Mzy, i,j_image, invaGPU, x[joffset+2]/a);
            rz = (r_vectors[i,2] + r_vectors[j,2]) * inva
            hj = r_vectors[j,2] * inva
            
            if i == j_image:
              invZi = 1.0 / hj
              invZi4 = invZi**4
              Mxy += -invZi4 * 0.125 # 3/24 = 0.125
              Myx +=  invZi4 * 0.125 # 3/24 = 0.125
            else:
              h_hat = hj / rz
              invR = 1.0 / np.sqrt(rx*rx + ry*ry + rz*rz)
              invR2 = invR * invR
              invR4 = invR2 * invR2
              ex = rx * invR
              ey = ry * invR
              ez = rz * invR
    
              fact1 =  invR2
              fact2 = (6.0 * h_hat * ez*ez * invR2 + (1.0 - 10.0 * ez*ez) * invR4) * 2.0
              fact3 = -ez * (3.0 * h_hat*invR2 - 5.0 * invR4) * 2.0
              fact4 = -ez * (h_hat * invR2 - invR4) * 2.0
    
              Mxx -=                       - fact3*ex*ey
              Mxy -=   fact1*ez            - fact3*ey*ey + fact4
              Mxz -= - fact1*ey - fact2*ey - fact3*ey*ez
              Myx -= - fact1*ez            + fact3*ex*ex - fact4
              Myy -=                         fact3*ex*ey
              Myz -=   fact1*ex + fact2*ex + fact3*ex*ez
              Mzx -=   fact1*ey
              Mzy -= - fact1*ex
  
            # 2. Compute product M_ij * T_j
            u[i,0] += (Mxx * force[j,0] + Mxy * force[j,1] + Mxz * force[j,2]) * norm_fact_f
            u[i,1] += (Myx * force[j,0] + Myy * force[j,1] + Myz * force[j,2]) * norm_fact_f
            u[i,2] += (Mzx * force[j,0] + Mzy * force[j,1] + Mzz * force[j,2]) * norm_fact_f

  return u.flatten()


@njit(parallel=True)
def no_wall_mobility_rot_times_torque_numba(r_vectors, torque, eta, a, L):
  ''' 
  Returns the product of the mobility translation-rotation at the blob level to the torque 
  on the blobs. Mobility for particles in an unbounded domain, it uses
  the standard RPY tensor.  
  
  This function uses numba.
  '''
  # Variables
  N = r_vectors.size // 3
  r_vectors = r_vectors.reshape(N, 3)
  torque = torque.reshape(N, 3)
  u = np.zeros((N, 3))
  fourOverThree = 4.0 / 3.0
  inva = 1.0 / a
  norm_fact_f = 1.0 / (8.0 * np.pi * eta * a**3)

  # Determine if the space is pseudo-periodic in any dimension 
  # We use a extended unit cell of length L=3*(Lx, Ly, Lz)
  periodic_x = 0
  periodic_y = 0 
  periodic_z = 0
  if L[0] > 0:
    periodic_x = 1
  if L[1] > 0:
    periodic_y = 1
  if L[2] > 0:
    periodic_z = 1
  
  # Loop over image boxes and then over particles
  for i in prange(N):
    for boxX in range(-periodic_x, periodic_x+1):
      for boxY in range(-periodic_y, periodic_y+1):
        for boxZ in range(-periodic_z, periodic_z+1):
          for j in range(N):
            
            # Compute vector between particles i and j
            rx = r_vectors[i,0] - r_vectors[j,0]
            ry = r_vectors[i,1] - r_vectors[j,1]
            rz = r_vectors[i,2] - r_vectors[j,2]

            # Project a vector r to the extended unit cell
            # centered around (0,0,0) and of size L=3*(Lx, Ly, Lz). If 
            # any dimension of L is equal or smaller than zero the 
            # box is assumed to be infinite in that direction.
            if L[0] > 0:
              rx = rx - int(rx / L[0] + 0.5 * (int(rx>0) - int(rx<0))) * L[0]
              rx = rx + boxX * L[0]
            if L[1] > 0:
              ry = ry - int(ry / L[1] + 0.5 * (int(ry>0) - int(ry<0))) * L[1]
              ry = ry + boxY * L[1]              
            if L[2] > 0:
              rz = rz - int(rz / L[2] + 0.5 * (int(rz>0) - int(rz<0))) * L[2]
              rz = rz + boxZ * L[2]            
               
            # 1. Compute mobility for pair i-j, if i==j use self-interation
            j_image = j
            if boxX != 0 or boxY != 0 or boxZ != 0:
              j_image = -1           

            # mobilityWTRPY(rx,ry,rz, Mxx,Mxy,Mxz,Myy,Myz,Mzz, i,j_image, invaGPU);
            if i==j_image:
              Mxx = 1.0
              Mxy = 0
              Mxz = 0
              Myy = Mxx
              Myz = 0
              Mzz = Mxx
            else:
              # Normalize distance with hydrodynamic radius
              rx = rx * inva
              ry = ry * inva
              rz = rz * inva
              r2 = rx*rx + ry*ry + rz*rz
              r = np.sqrt(r2)
              r3 = r2*r
              # TODO: We should not divide by zero 
              invr = 1.0 / r
              invr2 = 1.0 / r2
              invr3 = 1.0 / r3
              if r >= 2:
                c1 = -0.5
                c2 = 1.5 * invr2 
                Mxx = (c1 + c2*rx*rx) * invr3
                Mxy = (     c2*rx*ry) * invr3
                Mxz = (     c2*rx*rz) * invr3
                Myy = (c1 + c2*ry*ry) * invr3
                Myz = (     c2*ry*rz) * invr3
                Mzz = (c1 + c2*rz*rz) * invr3
              else:
                c1 =  (1.0 - 0.84375 * r + 0.078125 * r3) # 27/32 = 0.84375, 5/64 = 0.078125
                c2 =  0.28125 * invr - 0.046875 * r       # 9/32 = 0.28125, 3/64 = 0.046875
                Mxx = c1 + c2 * rx*rx 
                Mxy =      c2 * rx*ry 
                Mxz =      c2 * rx*rz 
                Myy = c1 + c2 * ry*ry 
                Myz =      c2 * ry*rz 
                Mzz = c1 + c2 * rz*rz 

            Myx = Mxy
            Mzx = Mxz
            Mzy = Myz
	  
            # 2. Compute product M_ij * T_j
            u[i,0] += (Mxx * torque[j,0] + Mxy * torque[j,1] + Mxz * torque[j,2]) * norm_fact_f
            u[i,1] += (Myx * torque[j,0] + Myy * torque[j,1] + Myz * torque[j,2]) * norm_fact_f
            u[i,2] += (Mzx * torque[j,0] + Mzy * torque[j,1] + Mzz * torque[j,2]) * norm_fact_f

  return u.flatten()


@njit(parallel=True)
def single_wall_mobility_rot_times_torque_numba(r_vectors, torque, eta, a, L):
  ''' 
  Returns the product of the mobility translation-rotation at the blob level to the torque 
  on the blobs. Mobility for particles in an unbounded domain, it uses
  the standard RPY tensor.  
  
  This function uses numba.
  '''
  # Variables
  N = r_vectors.size // 3
  r_vectors = r_vectors.reshape(N, 3)
  torque = torque.reshape(N, 3)
  u = np.zeros((N, 3))
  fourOverThree = 4.0 / 3.0
  inva = 1.0 / a
  norm_fact_f = 1.0 / (8.0 * np.pi * eta * a**3)

  # Determine if the space is pseudo-periodic in any dimension 
  # We use a extended unit cell of length L=3*(Lx, Ly, Lz)
  periodic_x = 0
  periodic_y = 0 
  periodic_z = 0
  if L[0] > 0:
    periodic_x = 1
  if L[1] > 0:
    periodic_y = 1
  if L[2] > 0:
    periodic_z = 1
  
  # Loop over image boxes and then over particles
  for i in prange(N):
    for boxX in range(-periodic_x, periodic_x+1):
      for boxY in range(-periodic_y, periodic_y+1):
        for boxZ in range(-periodic_z, periodic_z+1):
          for j in range(N):
            
            # Compute vector between particles i and j
            rx = r_vectors[i,0] - r_vectors[j,0]
            ry = r_vectors[i,1] - r_vectors[j,1]
            rz = r_vectors[i,2] - r_vectors[j,2]

            # Project a vector r to the extended unit cell
            # centered around (0,0,0) and of size L=3*(Lx, Ly, Lz). If 
            # any dimension of L is equal or smaller than zero the 
            # box is assumed to be infinite in that direction.
            if L[0] > 0:
              rx = rx - int(rx / L[0] + 0.5 * (int(rx>0) - int(rx<0))) * L[0]
              rx = rx + boxX * L[0]
            if L[1] > 0:
              ry = ry - int(ry / L[1] + 0.5 * (int(ry>0) - int(ry<0))) * L[1]
              ry = ry + boxY * L[1]              
            if L[2] > 0:
              rz = rz - int(rz / L[2] + 0.5 * (int(rz>0) - int(rz<0))) * L[2]
              rz = rz + boxZ * L[2]            
               
            # 1. Compute mobility for pair i-j, if i==j use self-interation
            j_image = j
            if boxX != 0 or boxY != 0 or boxZ != 0:
              j_image = -1           

            # mobilityWTRPY(rx,ry,rz, Mxx,Mxy,Mxz,Myy,Myz,Mzz, i,j_image, invaGPU);
            rx = rx * inva
            ry = ry * inva
            rz = rz * inva
            if i==j_image:
              Mxx = 1.0
              Mxy = 0
              Mxz = 0
              Myy = Mxx
              Myz = 0
              Mzz = Mxx
            else:
              # Normalize distance with hydrodynamic radius
              r2 = rx*rx + ry*ry + rz*rz
              r = np.sqrt(r2)
              r3 = r2*r
              # TODO: We should not divide by zero 
              invr = 1.0 / r
              invr2 = 1.0 / r2
              invr3 = 1.0 / r3
              if r >= 2:
                c1 = -0.5
                c2 = 1.5 * invr2 
                Mxx = (c1 + c2*rx*rx) * invr3
                Mxy = (     c2*rx*ry) * invr3
                Mxz = (     c2*rx*rz) * invr3
                Myy = (c1 + c2*ry*ry) * invr3
                Myz = (     c2*ry*rz) * invr3
                Mzz = (c1 + c2*rz*rz) * invr3
              else:
                c1 =  (1.0 - 0.84375 * r + 0.078125 * r3) # 27/32 = 0.84375, 5/64 = 0.078125
                c2 =  0.28125 * invr - 0.046875 * r       # 9/32 = 0.28125, 3/64 = 0.046875
                Mxx = c1 + c2 * rx*rx 
                Mxy =      c2 * rx*ry 
                Mxz =      c2 * rx*rz 
                Myy = c1 + c2 * ry*ry 
                Myz =      c2 * ry*rz 
                Mzz = c1 + c2 * rz*rz 

            Myx = Mxy
            Mzx = Mxz
            Mzy = Myz
	  
            # mobilityWTSingleWallCorrection(rx/a, ry/a, (rz+2*x[joffset+2])/a, Mxx,Mxy,Mxz,Myx,Myy,Myz,Mzx,Mzy,Mzz, i,j_image, invaGPU, x[joffset+2]/a);
            rz = (r_vectors[i,2] + r_vectors[j,2]) * inva
            hj = r_vectors[j,2] * inva
            if i == j_image:
              invZi = 1.0 / hj
              invZi3 = invZi**3
              Mxx += - invZi3 * 0.3125 # 15/48 = 0.3125
              Myy += - invZi3 * 0.3125 # 15/48 = 0.3125
              Mzz += - invZi3 * 0.125  # 3/24 = 0.125
            else:
              invR = 1.0 / np.sqrt(rx*rx + ry*ry + rz*rz)
              invR3 = invR * invR * invR
              ex = rx * invR
              ey = ry * invR
              ez = rz * invR
    
              fact1 =  ((1.0 - 6.0*ez*ez) * invR3 ) * 0.5
              fact2 = -(9.0 * invR3) / 6.0
              fact3 =  (3.0 * invR3 * ez)
              fact4 =  (3.0 * invR3)
    
              Mxx += fact1 + fact2 * ex*ex + fact4 * ey*ey
              Mxy += (fact2 - fact4)* ex*ey
              Mxz += fact2 * ex*ez
              Myx += (fact2 - fact4)* ex*ey
              Myy += fact1 + fact2 * ey*ey + fact4 * ex*ex
              Myz += fact2 * ey*ez
              Mzx += fact2 * ez*ex + fact3 * ex
              Mzy += fact2 * ez*ey + fact3 * ey
              Mzz += fact1 + fact2 * ez*ez + fact3 * ez        
  
            # 2. Compute product M_ij * T_j
            u[i,0] += (Mxx * torque[j,0] + Mxy * torque[j,1] + Mxz * torque[j,2]) * norm_fact_f
            u[i,1] += (Myx * torque[j,0] + Myy * torque[j,1] + Myz * torque[j,2]) * norm_fact_f
            u[i,2] += (Mzx * torque[j,0] + Mzy * torque[j,1] + Mzz * torque[j,2]) * norm_fact_f

  return u.flatten()
