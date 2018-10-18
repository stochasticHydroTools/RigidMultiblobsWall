from __future__ import print_function
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
  
  This function makes use of numba.
  '''
  # Variables
  N = r_vectors.size / 3
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
            if L[2] > 0:
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


@njit(parallel=True)
def single_wall_mobility_trans_times_force_numba(r_vectors, force, eta, a, L):
  ''' 
  Returns the product of the mobility at the blob level to the force 
  on the blobs. Mobility for particles in an unbounded domain, it uses
  the standard RPY tensor.  
  
  This function makes use of numba.
  '''
  # Variables
  N = r_vectors.size / 3
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
            rz = (r_vectors[i,2] + r_vectors[j,2]) / a
            hj = r_vectors[j,2] / a

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
            u[i,0] += (Mxx * force[j,0] + Mxy * force[j,1] + Mxz * force[j,2]) * norm_fact_f
            u[i,1] += (Myx * force[j,0] + Myy * force[j,1] + Myz * force[j,2]) * norm_fact_f
            u[i,2] += (Mzx * force[j,0] + Mzy * force[j,1] + Mzz * force[j,2]) * norm_fact_f

  return u.flatten()


@njit(parallel=True)
def no_wall_mobility_trans_times_torque_numba(r_vectors, torque, eta, a, L):
  ''' 
  Returns the product of the mobility translation-rotation at the blob level to the torque 
  on the blobs. Mobility for particles in an unbounded domain, it uses
  the standard RPY tensor.  
  
  This function makes use of numba.
  '''
  # Variables
  N = r_vectors.size / 3
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
