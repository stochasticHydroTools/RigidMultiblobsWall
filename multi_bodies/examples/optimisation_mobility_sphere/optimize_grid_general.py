'''
This script finds the optimal grid (S, a) that best approximates the mobility (N) and slip mobility (tilde{N})
of a rigid body with arbitrary shape with respect to a reference solution.
 - S: scale factor that sets the size of the body (S=R_new / Rg for a sphere)
 - a: radius of the blobs discretizing the body's surface
The routine first reads the reference mobility matrices and then iterates to find the doublet (S,a)
the minimizes the cost function that computes  the distance between the current mobility and the reference solution.

The slip mobilities are optional.
'''

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


def error_mobilities(Mob, Mob_slip, Mob_ref, Sigma_US_ref, Sigma_OS_ref, Nref, choice_error, verbose):

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
  
  # Compute scaling ratio between the two resolutions 
  Nlow = MslipUS.shape[1]//3
  ratio_US = np.sqrt(Nlow/Nref)
  ratio_OS = ratio_US

  if Sigma_US_ref:
    error_US = np.linalg.norm(Sigma_US_ref - ratio_US*Sigma_US)/np.linalg.norm(Sigma_US_ref)
    error_OS = np.linalg.norm(Sigma_OS_ref - ratio_OS*Sigma_OS)/np.linalg.norm(Sigma_OS_ref)
  else:
    error_US = 0
    error_OS = 0

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
    if Sigma_US_ref:
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
 
  return error 

def cost_function(x, N, Nref, eta, double_layer, choice_error, normals, weights, struct_orig_config, S_orig, Mob_ref, Sigma_US_ref, Sigma_OS_ref, verbose):
  S = x[0]
  a = x[1]
  
  mobility_blobs_implementation = 'python_no_wall'
  struct_ref_config = struct_orig_config * S / S_orig

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
  return error_mobilities(Mob, Mob_slip, Mob_ref, Sigma_US_ref, Sigma_OS_ref, Nref, choice_error, verbose)


if __name__ ==  '__main__':
  
  # Set parameters
  eta = 1
  prefix_output = '/home/fbalboa/simulations/RigidMultiblobsWall/chiral/data/run3000/run3110/run3110.optimized.0.0'
  filename_mob_ref = '/home/fbalboa/simulations/RigidMultiblobsWall/chiral/data/run3000/run3110/run3110.inf.0.0.body_mobility.dat'
  filename_mob_slip_ref = None
  filename_Laplace = None
  filename_vertex = 'Structures/superellipsoid_Lg_2_r_5_N_26.vertex'
  tol = 1e-11
  display_iterations = True
  verbose = 1
  blob_radius_guess = 0.4
  N = 26
  Nb_ref = 9999
  double_layer = 0
  choice_error = 'sum'
  
  # Load mobilities
  Mob_ref = np.loadtxt(filename_mob_ref)
  if filename_mob_slip_ref:
    Mob_slip_ref = np.loadtxt(filename_mob_slip_ref)

  if filename_mob_slip_ref:
    MslipUS_ref = Mob_slip_ref[0:3,:]
    MslipOS_ref = Mob_slip_ref[3:6,:]
    U_US_ref, Sigma_US_ref, VT_US_ref = np.linalg.svd( MslipUS_ref )
    U_OS_ref, Sigma_OS_ref, VT_OS_ref = np.linalg.svd( MslipOS_ref )    
  else:
    U_US_ref, Sigma_US_ref, VT_US_ref = None, None, None
    U_OS_ref, Sigma_OS_ref, VT_OS_ref = None, None, None

  # Read vertex file
  struct_orig_config = read_vertex_file.read_vertex_file(filename_vertex)
  S_orig = 1.0
  print('S_orig = ', S_orig)
 
  if filename_Laplace:
    Laplace_orig = np.loadtxt(orig_filename)
    normals = np.copy(Laplace_orig[:,0:3])  
    weights = np.copy(Laplace_orig[:,6])
  else:
    normals = None
    weights = None

  # First guess
  xin = [S_orig - blob_radius_guess, blob_radius_guess] 
 
  # Set bounds for nonlinear solver
  bounds = [(0.5,1),(1e-6,0.5)]

  # Print original error
  print(cost_function(xin, N, Nb_ref, eta, double_layer, choice_error, normals, weights, struct_orig_config, S_orig, Mob_ref,Sigma_US_ref, Sigma_OS_ref,verbose))

  # Optimize grid
  result = scop.differential_evolution(cost_function,
                                       bounds=bounds,
                                       maxiter = 1000,
                                       tol = tol,
                                       disp = display_iterations,
                                       args=(N, Nb_ref, eta, double_layer, choice_error, normals, weights, struct_orig_config, S_orig, Mob_ref,Sigma_US_ref, Sigma_OS_ref, 0) ) 

  # Print final result
  print(result.x)
  print(result.message)
  print(result.nfev) 
  print(cost_function(result.x, N, Nb_ref, eta, double_layer, choice_error, normals, weights, struct_orig_config, S_orig, Mob_ref,Sigma_US_ref, Sigma_OS_ref,verbose))

  if prefix_output:
    # Extract solution
    S = result.x[0]
    a = result.x[1]
    
    # Save mobility and vertex file
    mobility_blobs_implementation = 'python_no_wall'
    struct_ref_config = struct_orig_config * S / S_orig

    # Create rigid bodies
    bodies = []
    struct_location = np.array([0, 0, 0])
    struct_orientation = Quaternion(np.array([1, 0, 0, 0]))
    b = body.Body(struct_location, struct_orientation, struct_ref_config, a)
    b.mobility_blobs = multi_bodies.set_mobility_blobs(mobility_blobs_implementation)
    # Append bodies to total bodies list
    bodies.append(b)
    multi_bodies.mobility_blobs = multi_bodies.set_mobility_blobs(mobility_blobs_implementation)

    # Compute mobility
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

    # Save vertex file
    name = prefix_output + '.blobs.vertex'
    with open(name, 'w') as f_handle:
      f_handle.write(str(r_vectors_blobs.shape[0]) + '\n')
      np.savetxt(f_handle, r_vectors_blobs)

    # Save mobilities
    name = prefix_output + '.body_mobility.dat'
    np.savetxt(name, Mob)
    name = prefix_output + '.body_mobility_slip.dat'
    np.savetxt(name, Mob_slip)
    
