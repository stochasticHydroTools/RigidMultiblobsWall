import argparse
import numpy as np
import scipy.optimize as scop
import subprocess
from functools import partial
import sys 
import os 
import time
# Try to import numba
try:
  from numba import njit, prange
except ImportError:
  print('numba not found')


# Find project functions
found_functions = False
path_to_append = ''  
while found_functions is False:
  try: 
    import general_application_utils as utils
    from body import body
    import multi_bodies
    from quaternion_integrator.quaternion import Quaternion
    from read_input import read_vertex_file
    found_functions = True 
  except ImportError as exc:
    sys.stderr.write('Error: failed to import settings module ({})\n'.format(exc))
    path_to_append += '../'
    print('searching functions in path ', path_to_append)
    sys.path.append(path_to_append)
    if len(path_to_append) > 21: 
      print('\nProjected functions not found. Edit path in create_laplace_file.py')
      sys.exit()

@njit(parallel=True, fastmath=True)
def double_layer_matrix_source_target_numba(source, target, normals, weights):
  '''
  Stokes double operator, diagonals are set to zero.
  '''
  # Prepare vectors
  num_targets = target.size // 3
  num_sources = source.size // 3
  source = source.reshape(num_sources, 3)
  target = target.reshape(num_targets, 3)
  normals = normals.reshape(num_sources, 3)
  D = np.zeros((3*num_targets,3*num_sources))
  factor = -3.0 / (4.0 * np.pi)

  # Copy to one dimensional vectors
  rx_src = np.copy(source[:,0])
  ry_src = np.copy(source[:,1])
  rz_src = np.copy(source[:,2])
  rx_trg = np.copy(target[:,0])
  ry_trg = np.copy(target[:,1])
  rz_trg = np.copy(target[:,2])
  nx_vec = np.copy(normals[:,0])
  ny_vec = np.copy(normals[:,1])
  nz_vec = np.copy(normals[:,2])

  # Loop over image boxes and then over particles
  for i in prange(num_targets):
    rxi = rx_trg[i]
    ryi = ry_trg[i]
    rzi = rz_trg[i]

    for j in range(num_sources):
      # Compute vector between particles i and j
      rx = rxi - rx_src[j]
      ry = ryi - ry_src[j]
      rz = rzi - rz_src[j]


      # Compute interaction without wall
      r2 = rx*rx + ry*ry + rz*rz
      r = np.sqrt(r2)
      if r < 1e-14:
        continue
      r5 = r**5

      # 2. Compute product T_ijk * n_k 
      rxnx = rx * nx_vec[j]
      ryny = ry * ny_vec[j]
      rznz = rz * nz_vec[j]
      rdotn = rxnx + ryny + rznz
      facr5 = rdotn * weights[j] / r5
      Dxx = rx * rx * facr5
      Dyx = ry * rx * facr5
      Dzx = rz * rx * facr5
      Dyy = ry * ry * facr5
      Dzy = rz * ry * facr5
      Dzz = rz * rz * facr5
      Dxy = Dyx
      Dxz = Dzx
      Dyz = Dzy

      D[3*i,3*j] = Dxx
      D[3*i,3*j+1] = Dxy
      D[3*i,3*j+2] = Dxz
      D[3*i+1,3*j] = Dyx
      D[3*i+1,3*j+1] = Dyy
      D[3*i+1,3*j+2] = Dyz
      D[3*i+2,3*j] = Dzx
      D[3*i+2,3*j+1] = Dzy
      D[3*i+2,3*j+2] = Dzz

  return D*factor


def error_mobilities(Mob, Mob_slip, Mob_ref, Sigma_US_ref, Sigma_OS_ref, Nref, choice_ratio, choice_error, verbose):

  MUF = Mob[0:3,0:3]
  MUT = Mob[0:3,3:6]
  MOF = Mob[3:6,0:3]
  MOT = Mob[3:6,3:6]

  MUF_ref = Mob_ref[0:3,0:3]
  MUT_ref = Mob_ref[0:3,3:6]
  MOF_ref = Mob_ref[3:6,0:3]
  MOT_ref = Mob_ref[3:6,3:6]

  error_UF = np.linalg.norm(MUF_ref - MUF)/np.linalg.norm(MUF_ref)

  norm_MUT_ref = np.linalg.norm(MUT_ref)
  if norm_MUT_ref > 1e-12:
    error_UT = np.linalg.norm(MUT_ref - MUT)/np.linalg.norm(MUT_ref) 
  else:
    error_UT = 0 

  norm_MOF_ref = np.linalg.norm(MUT_ref)
  if norm_MOF_ref > 1e-12:
    error_OF = np.linalg.norm(MOF_ref - MOF)/np.linalg.norm(MOF_ref) 
  else:
    error_OF = 0
 
  error_OT = np.linalg.norm(MOT_ref - MOT)/np.linalg.norm(MOT_ref)

  # svd_mob_slip_low
  MslipUS = Mob_slip[0:3,:]
  MslipOS = Mob_slip[3:6,:]

  U_US, Sigma_US, VT_US = np.linalg.svd( MslipUS )
  U_OS, Sigma_OS, VT_OS = np.linalg.svd( MslipOS )

  if choice_ratio == 'sqrt':
    Nlow = MslipUS.shape[1]//3
    ratio_US = np.sqrt(Nlow/Nref)
    ratio_OS = ratio_US
  elif choice_ratio == 'sigma':
    ratio_US = np.linalg.norm(Sigma_US_ref)/np.linalg.norm(Sigma_US)
    ratio_OS = np.linalg.norm(Sigma_OS_ref)/np.linalg.norm(Sigma_OS)

  error_US = np.linalg.norm(Sigma_US_ref - ratio_US*Sigma_US)/np.linalg.norm(Sigma_US_ref)
  error_OS = np.linalg.norm(Sigma_OS_ref - ratio_OS*Sigma_OS)/np.linalg.norm(Sigma_OS_ref)

  # Total error
  if choice_error == 'minmax' :
    error = np.amax([error_UF, error_UT, error_OF, error_OT, error_US, error_OS])
  elif choice_error == 'sum' :
    error = error_UF + error_UT + error_OF + error_OT + error_US + error_OS

  if verbose == 1:
   print('error_UF = ', error_UF) 
   print('error_UT = ', error_UT) 
   print('error_OF = ', error_OF) 
   print('error_OT = ', error_OT) 
   print('Sigma_US')
   print(Sigma_US)
   print('Sigma_US*ratio_US')
   print(Sigma_US*ratio_US)
   print('Sigma_US_ref')
   print(Sigma_US_ref)
   print('Sigma_OS')
   print(Sigma_OS)
   print('Sigma_OS*ratio_OS')
   print(Sigma_OS*ratio_OS)
   print('Sigma_OS_ref')
   print(Sigma_OS_ref)
   print('error_US = ', error_US) 
   print('error_OS = ', error_OS) 
   print('error = ', error)
#   input()
 
  return error 

def cost_function(x, N, Nref, eta, double_layer, choice_ratio, choice_error, normals, weights, struct_orig_config, Rg_orig, Mob_ref, Sigma_US_ref, Sigma_OS_ref, verbose):
  Rg = x[0]
  a = x[1]
  
  mobility_blobs_implementation = 'python_no_wall'
  struct_ref_config = struct_orig_config * Rg/Rg_orig

  # Create rigid bodies
  bodies = []
  blobs_offset = 0
  struct_location = np.array([0, 0, 0])
  struct_orientation = Quaternion(np.array([1, 0, 0, 0]))
  b = body.Body(struct_location, struct_orientation, struct_ref_config, a)
  b.mobility_blobs = multi_bodies.set_mobility_blobs(mobility_blobs_implementation)
  # Compute the blobs offset for lambda in the whole system array
  b.blobs_offset = blobs_offset
  blobs_offset += b.Nblobs
  # Append bodies to total bodies list
  bodies.append(b)

  multi_bodies.mobility_blobs = multi_bodies.set_mobility_blobs(mobility_blobs_implementation)

  r_vectors_blobs = multi_bodies.get_blobs_r_vectors(bodies, b.Nblobs)
  mobility_blobs = multi_bodies.mobility_blobs(r_vectors_blobs, eta, a)
  resistance_blobs = np.linalg.inv(mobility_blobs)
  K = multi_bodies.calc_K_matrix(bodies, b.Nblobs)
  resistance_bodies = np.dot(K.T, np.dot(resistance_blobs, K))
  Mob = np.linalg.pinv(np.dot(K.T, np.dot(resistance_blobs, K)))
  Mob_slip = np.dot(Mob, np.dot(K.T,resistance_blobs))
  
  if double_layer == 1:
    Io2 = np.eye(3*b.Nblobs)*0.5
    D = double_layer_matrix_source_target_numba(r_vectors_blobs, r_vectors_blobs, normals, weights)
    I2pD = Io2 + D
    Mob_slip = np.dot(Mob_slip,I2pD)

  return error_mobilities(Mob, Mob_slip, Mob_ref, Sigma_US_ref, Sigma_OS_ref, Nref, choice_ratio, choice_error, verbose)

if __name__ ==  '__main__':
 eta = 1
 # Choose to optimize double_layer (=1) or single_layer (=0) problem
 double_layer = 0
 print('double_layer  = ', double_layer)
 # Objective function uses either the 'sum' of all errors or the 'minmax'
 choice_error = 'sum'
 print('choice_error  = ', choice_error)
 # Scaling factor to compute the error on the singular values between ref and current mesh. 
 # scaling is either 'sqrt' or 'sigma' (see code above)
 choice_ratio = 'sqrt'
 print('choice_ratio  = ', choice_ratio)

 N = 642
 print('N = ', N)

 # Load reference solution
 Nb_ref = 10242
 print('Nb_ref = ', Nb_ref)
 basename_ref = 'data_ref/run.body_mobility_double_layer_N_' + str(Nb_ref)
 filename_mob_ref =  basename_ref + '.body_mobility'
 if double_layer == 1:
   filename_mob_slip_ref =  basename_ref + '.body_slip_mobility_double_layer'
 else:
   filename_mob_slip_ref =  basename_ref + '.body_slip_mobility'
 Mob_ref = np.loadtxt(filename_mob_ref + '.dat') 
 Mob_slip_ref = np.loadtxt(filename_mob_slip_ref + '.dat')
 MslipUS_ref = Mob_slip_ref[0:3,:]
 MslipOS_ref = Mob_slip_ref[3:6,:]
 U_US_ref, Sigma_US_ref, VT_US_ref = np.linalg.svd( MslipUS_ref )
 U_OS_ref, Sigma_OS_ref, VT_OS_ref = np.linalg.svd( MslipOS_ref )

 # Load initial mesh to be optimised
 if N==12:
   str_Rg_orig = '0_7921'
 elif N==42:
   str_Rg_orig = '0_8913'
 elif N==162:
   str_Rg_orig = '0_9497'
 elif N==642:
   str_Rg_orig = '0_9767'
 elif N==2562:
   str_Rg_orig = '0_9888'
 elif N==10242:
   str_Rg_orig = '0_994578'
 orig_filename = '../../Structures/shell_N_' + str(N) + '_Rg_' + str_Rg_orig + '_Rh_1'
 struct_orig_config = read_vertex_file.read_vertex_file(orig_filename + '.vertex')
 Rg_orig = np.linalg.norm(struct_orig_config[0,:])
 Laplace_orig = np.loadtxt(orig_filename + '.Laplace')
 normals = np.copy(Laplace_orig[:,0:3])  
 weights = np.copy(Laplace_orig[:,6])

###### ORIGINAL GRIDS ###############################################
# xin = [0.7921, 0.41642068286674966] # Initial grid for N=12
# xin = [8.912655971483167e-01, 0.243553056072] # Initial grid for N=42


 # First guess
 xin = [0.9, 0.4] # Initial grid for N=12
 # Tolerance nonlinear solver
 tol = 1e-11
 # Set bounds for nonlinear solver
 bounds = [(0.5,1),(1e-6,0.5)]
 # Display iterations
 disp = True

 verbose = 1
 print(cost_function(xin, N, Nb_ref, eta, double_layer, choice_ratio, choice_error, normals, weights, struct_orig_config, Rg_orig, Mob_ref,Sigma_US_ref, Sigma_OS_ref,verbose))


 result = scop.differential_evolution(cost_function,
                             bounds=bounds,
                             maxiter = 1000,
                             tol = tol,
                             disp =  disp,
                             args=(N, Nb_ref, eta, double_layer, choice_ratio, choice_error, normals, weights, struct_orig_config, Rg_orig, Mob_ref,Sigma_US_ref, Sigma_OS_ref, 0) ) 

 print(result.x)
 print(result.message)
 print(result.nfev)

 print(cost_function(result.x, N, Nb_ref, eta, double_layer, choice_ratio, choice_error, normals, weights, struct_orig_config, Rg_orig, Mob_ref,Sigma_US_ref, Sigma_OS_ref,verbose))

