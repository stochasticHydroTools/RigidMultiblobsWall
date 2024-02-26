'''
Solve the linear system [M_SB, w_1 * K.T] * lambda_B = [v_S, 0]

where v_S is defined in a surface surrounding swimmer B and lambda_B is defined on the blobs of swimmer B.
v_S is read from a file.
The code assume the rigid body is formed by N_real_blobs and M_ghost_blobs in that order.

How to use:
1. Edit parameters at the top of main.
2. Run as python compute_slip_from_flow.py.

Parameters:
1. eta = fluid viscosity.
2. name_U_A = file name with the swimmer velocity.
3. name_v_S_A = name of the file with the flow on the shell.
4. name_v_S_A_check = name of the file with the flow on another shell for verification.
5. blob_radius_B =  blob radius of swimmer B
6. name_vertex_B = list of strings with the names of the vertex files.
7. name_clones_B = name of the clones file.
8. skiprows_vertex_B = number of rows to skip in the vertex files.
9. check_flow = check the flow in the second shell.
10. tolerance_force_free = tolerance to impose the force free condition.

Outputs:
1. output_prefix.blob_forces_B.dat: all blobs forces
2. output_prefix.slip: slip on all blobs
3. output_prefix.s.p.16.shell_radius.4.0.dat: singular values of the matrix defining the least squared problem.
'''
import numpy as np
from numpy import savetxt
from functools import partial
from decimal import Decimal
import sys
import scipy
import scipy.optimize
try:
  import matplotlib.pyplot as plt
except ImportError as e:
  print(e)

# Add path to RigidMultiblobsWall
sys.path.append('../../')
sys.path.append('../../../')

# Import local modules
import user_defined_functions as udf
from body import body
from quaternion_integrator.quaternion import Quaternion
from mobility import mobility as mob


if __name__ == '__main__':
  # Set parameters
  eta = 1e-03
  name_U_A = 'data/run_squirmer.U_A.dat'
  name_v_S_A = 'data/run_squirmer.shell_radius.4.v_S.p.16.dat'
  name_v_S_A_check = 'data/run_squirmer.shell_radius.4.v_S.p.32.dat'
  blob_radius_B =  0.131
  name_vertex_B = ['../../Structures/shell_N_162_Rg_0_9497_Rh_1.vertex']
  name_clones_B = 'Structures/swimmer.clones'
  skiprows_vertex_B = 1
  check_flow = True
  tolerance_force_free = 1e-10
  output_prefix = './data/run_squirmer'

  # Read swimmer A velocity
  U_A = np.loadtxt(name_U_A)
  U_A = U_A.reshape((U_A.size // 6, 6))[0]

  # Read flow velocity on the shell
  data_S = np.loadtxt(name_v_S_A)
  r_sphere = np.ascontiguousarray(data_S[:,0:3])
  v_S = np.ascontiguousarray(data_S[:,4:7]).flatten()
  shell_radius = np.linalg.norm(r_sphere[0])
  p = int(np.sqrt(r_sphere.shape[0] // 2) - 1)
  uv, uv_weights = udf.parametrization(p)
  print('p            = ', p)
  print('shell_radius = ', shell_radius, '\n')
  if name_v_S_A_check:
    data_S_check = np.loadtxt(name_v_S_A_check)
    r_sphere_check = np.ascontiguousarray(data_S_check[:,0:3])
    v_S_check = np.ascontiguousarray(data_S_check[:,4:7]).flatten()
    shell_radius_check = np.linalg.norm(r_sphere_check[0])
    
  # Read input files swimmer B
  x_B = np.loadtxt(name_clones_B, skiprows=1)
  x_B = x_B.reshape((x_B.size // 7, 7))
  r_blobs_B = []
  blobs_radius_B = []
  for k, name in enumerate(name_vertex_B):
    r_blobs = np.loadtxt(name, skiprows=skiprows_vertex_B)
    print('r_blobs     = ', r_blobs.shape)

    # Rotate and transtale vectors
    quaternion_body = Quaternion(x_B[k,3:7])      
    R = quaternion_body.rotation_matrix()
    r_blobs_rotated = np.dot(r_blobs[:,0:3], R.T) + x_B[k,0:3]

    # Append blobs
    r_blobs_B.append(r_blobs_rotated)
    if r_blobs.shape[1] == 4:
      blobs_radius_B.append(r_blobs[:,3])
    else:
      blobs_radius_B.append(np.ones(r_blobs.shape[0]) * blob_radius_B)
  r_blobs_B = np.ascontiguousarray(np.concatenate(r_blobs_B))
  blobs_radius_B = np.ascontiguousarray(np.concatenate(blobs_radius_B))
   
  # Build body B
  orientation_B = Quaternion(np.array([1, 0, 0, 0]))
  b_B = body.Body(np.zeros(3), orientation_B, r_blobs_B, blob_radius_B)
  b_B.mobility_blobs = mob.rotne_prager_tensor
  K_B = b_B.calc_K_matrix()
  M_B = b_B.calc_mobility_blobs(eta, blob_radius_B)
  L_B, lower_B = scipy.linalg.cho_factor(M_B)
  L_B = np.triu(L_B)  
  N_B = np.linalg.pinv(np.dot(K_B.T, scipy.linalg.cho_solve((L_B,lower_B), K_B, check_finite=False)))

  # Build matrix M_SB, to solve linear system M_SB * lambda_tilde = v_S
  num_rows = v_S.size 
  num_columns = r_blobs_B.size
  print('num_rows    = ', num_rows)
  print('num_columns = ', num_columns)

  # Build linear operator
  def mobility_wrapper(force, source, target, radius_source, radius_target, eta):
    return mob.no_wall_mobility_trans_times_force_source_target_numba(source, target, force, radius_source, radius_target, eta)    
  linear_operator = partial(mobility_wrapper, source=b_B.get_r_vectors(), target=r_sphere, radius_source=blobs_radius_B, radius_target=np.zeros(r_sphere.shape[0]), eta=eta)
  LO = scipy.sparse.linalg.LinearOperator((num_rows, num_columns), matvec = linear_operator, dtype='float64')

  # Build matrix for least squares
  M_SB = np.zeros((num_rows, num_columns))
  for i in range(num_columns):
    x = np.zeros(num_columns)
    x[i] = 1
    M_SB[:,i] = LO.matvec(x)

  # Print some info about M_SB
  rank_M_SB = np.linalg.matrix_rank(M_SB)
  Z = scipy.linalg.null_space(M_SB)
  cond_M_SB = np.linalg.cond(M_SB)
  print('M_SB.shape        = ', M_SB.shape)
  print('Z.shape           = ', Z.shape)
  print('rank(M_SB)        = ', rank_M_SB)
  print('null(M_SB)        = ', Z.shape[1])
  print('rank() + null()   = ', rank_M_SB + Z.shape[1])
  print('cond(M_SB)        = %.4E ' % Decimal(cond_M_SB))
  print('\n')

  # Compute lambda_B
  if True:
    # First solve M_SB * lambda_B_tilde = v_S
    u, s, vh = np.linalg.svd(M_SB)
    s_inv_vec = 1.0 / s     
    sel = abs(s_inv_vec / s_inv_vec[0]) > 1e+12
    s_inv_vec[sel] = 0
    s_inv = np.zeros((vh.shape[0], u.shape[0]))
    s_inv[0:s.size, 0:s.size] = np.diag(s_inv_vec)
    lambda_B_tilde = (np.dot(np.dot(vh.T, s_inv), np.dot(u.T, v_S.flatten())))

    # Compute norm of K.T * lambda_B_tilde
    weight_force_free = np.linalg.norm(np.dot(K_B.T, lambda_B_tilde)) / (eta * blob_radius_B * np.linalg.norm(lambda_B_tilde) * tolerance_force_free)
    print('weight_force_free                             =  %.4E ' % weight_force_free)
    print('|K_B.T * lambda_B_tilde|_2                    = ', np.linalg.norm(np.dot(K_B.T, lambda_B_tilde)))
    print(' K_B.T * lambda_B_tilde                       = \n', np.dot(K_B.T, lambda_B_tilde))
    print('\n\n')

    # Build matrix of big linear system
    Matrix_B = np.vstack((M_SB, weight_force_free * K_B.T))
    
    # Compute SVD decomposition 
    u, s, vh = np.linalg.svd(Matrix_B)
    name = output_prefix + '.s.p.' + str(p) + '.shell_radius.' + str(shell_radius) + '.dat'
    np.savetxt(name, s / s[0])
    
    # Set inverse singular values
    s_inv_vec = 1.0 / s     
    sel = abs(s_inv_vec / s_inv_vec[0]) > 1e+12
    s_inv_vec[sel] = 0
    s_inv = np.zeros((vh.shape[0], u.shape[0]))
    s_inv[0:s.size, 0:s.size] = np.diag(s_inv_vec)

    # Build RHS
    RHS = np.concatenate((v_S.flatten(), np.zeros(6)))
    
    # Solve least square problem
    lambda_B = (np.dot(np.dot(vh.T, s_inv), np.dot(u.T, RHS.flatten())))

    # Save blob forces
    with open(output_prefix + '.blob_forces_B.dat', 'w') as f_handle:
      f_handle.write(str(lambda_B.size // 3) + '\n')
      np.savetxt(f_handle, lambda_B.reshape((lambda_B.size // 3, 3)))
    print('|K_B.T * lambda_B|_2                          = ', np.linalg.norm(np.dot(K_B.T, lambda_B)))
    print(' K_B.T * lambda_B                             = \n', np.dot(K_B.T, lambda_B))
    print('\n')

  # Compute active slip
  if True:
    slip_B = np.dot(M_B, lambda_B) - np.dot(K_B, U_A)

    # Save active slip
    with open(output_prefix + '.slip', 'w') as f_handle:
      f_handle.write(str(slip_B.size // 3) + '\n')
      np.savetxt(f_handle, slip_B.reshape((slip_B.size // 3, 3)))

  # Compute errors
  if True:
    # Recompute some variables of swimmer B and compare with swimmer A to see if both agree
    v_S_B = mob.no_wall_mobility_trans_times_force_source_target_numba(b_B.get_r_vectors(), r_sphere, lambda_B, blobs_radius_B, np.zeros(r_sphere.shape[0]), eta)
    v_diff_2 = np.sqrt(np.sum(((v_S - v_S_B)**2).reshape((v_S.size // 3, 3)) * uv_weights[:,None] * shell_radius**2))
    v_S_2 = np.sqrt(np.sum((v_S**2).reshape((v_S.size // 3, 3)) * uv_weights[:,None] * shell_radius**2))
    print('lambda_B.norm      = %.4E' % np.linalg.norm(lambda_B))
    print('slip_B.norm        = %.4E' % np.linalg.norm(slip_B))
    print('|v_S|_2            = %.4E' % np.linalg.norm(v_S), '\n')
    print('Errors in computation surface')
    print('|v_diff|_2         = ', v_diff_2)
    print('|v_diff|_infty     = ', np.linalg.norm(v_S - v_S_B, ord=np.inf))
    print('rel |v_diff|_2     = ', v_diff_2 / v_S_2)
    print('rel |v_diff|_infty = ', np.linalg.norm((v_S - v_S_B), ord=np.inf) / np.linalg.norm(v_S, ord=np.inf))
    print('\n')

    if check_flow:
      # Flow in larger sphere
      v_S_Large = v_S_check
      p_check = int(np.sqrt(r_sphere_check.shape[0] // 2) - 1)
      uv, uv_weights = udf.parametrization(p_check)
      v_S_Large_B = mob.no_wall_mobility_trans_times_force_source_target_numba(b_B.get_r_vectors(), r_sphere_check, lambda_B, blobs_radius_B, np.zeros(r_sphere_check.shape[0]), eta)
      v_diff_2 = np.sqrt(np.sum(((v_S_Large - v_S_Large_B)**2).reshape((v_S_Large.size // 3, 3)) * uv_weights[:,None] * shell_radius**2))
      v_S_2 = np.sqrt(np.sum((v_S_Large**2).reshape((v_S_Large.size // 3, 3)) * uv_weights[:,None] * shell_radius**2))
      print('Errors in check surface')
      print('|v_diff|_2         = ', v_diff_2)
      print('|v_diff|_infty     = ', np.linalg.norm(v_S_Large - v_S_Large_B, ord=np.inf))
      print('rel |v_diff|_2     = ', v_diff_2 / v_S_2)
      print('rel |v_diff|_infty = ', np.linalg.norm(v_S_Large - v_S_Large_B, ord=np.inf) / np.linalg.norm(v_S_Large, ord=np.inf))
      print('\n')

    # Compute velocity swimmer
    U_B = udf.calc_velocity_swimmer(slip_B, N_B, K_B, L_B, lower_B)
    U_diff = U_B - U_A
    U_diff_relative = np.linalg.norm(U_diff) / np.linalg.norm(U_A) if np.linalg.norm(U_A) > 0 else 0
    print('U_A            = ', U_A)
    print('|U_diff|_2     = ', np.linalg.norm(U_diff))
    print('rel |U_diff|_2 = ', U_diff_relative)
    
