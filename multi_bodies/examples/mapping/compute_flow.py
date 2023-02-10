'''
Compute the flow created by a swimmer with active slip on a shell.
The shell is centered in the frame of reference of the body "frame_body".
If frame_body < 0 then the shell is centered around (0,0,0) in the laboratory frame of reference.
We assume that all the blobs have the same hydrodynamic radius.

How to use:
1. Edit parameters at the top of the main function.
2. Run as python compute_flow.py

Parameters:
1. eta = fluid viscosity.
2. blob_radius_A = blob radius.
3. name_vertex_A = list of strings with the names of the vertex files.
4. name_clones_A = name of the clones file.
5. name_slip_A = name of the slip file.
6. skiprows_vertex_A = number of rows to skip in the vertex file.
7. skiprows_slip_A = number of rows to skip in the slip file.
8. shell_radius = radius of the shell.
9. p = order of the discretization. The shell has 2 * (p+1)**2 points.
10. frame_body = body used as frame of reference.
11. output_prefix = prefix of the output files.

Outputs:
1. output_prefix.U_A.dat: velocity of the swimmer.
2. output_prefix.v_s.p.p.shell_radius.shell_radius.dat: flow on the shell with format
   r_shell (3 numbers), weight for quadrature (1 number), flow velocity (3 numbers).
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
  blob_radius_A = 6.84099578379999268E-002 / 2
  name_vertex_A = ['../../Structures/shell_N_2562_Rg_0_9888_Rh_1.vertex']
  name_clones_A = 'Structures/swimmer.clones'
  name_slip_A = 'data/squirmer_N_2562_B1_1_B2_1.slip'
  skiprows_vertex_A = 1
  skiprows_slip_A = 0
  shell_radius = 4
  p = 16
  frame_body = 0
  output_prefix = 'data/run_squirmer'

  # Create shell
  uv, uv_weights = udf.parametrization(p)
  r_shell = udf.sphere(shell_radius, uv)
    
  # Read input files swimmer A
  x_A = np.loadtxt(name_clones_A, skiprows=1)
  x_A = x_A.reshape((x_A.size // 7, 7))

  # Create r_blobs
  r_blobs_A = []
  blobs_radius_A = []
  for k, name in enumerate(name_vertex_A):
    r_blobs = np.loadtxt(name, skiprows=  skiprows_vertex_A, comments='#')

    # Rotate and transtale vectors
    quaternion_body = Quaternion(x_A[k,3:7])      
    R = quaternion_body.rotation_matrix()
    r_blobs_rotated = np.dot(r_blobs[:,0:3], R.T) + x_A[k,0:3]

    # Append blobs
    r_blobs_A.append(r_blobs_rotated)
    if r_blobs.shape[1] == 4:
      blobs_radius_A.append(r_blobs[:,3])
    else:
      blobs_radius_A.append(np.ones(r_blobs.shape[0]) * blob_radius_A)
  r_blobs_A = np.ascontiguousarray(np.concatenate(r_blobs_A))
  blobs_radius_A = np.ascontiguousarray(np.concatenate(blobs_radius_A))
    
  # Read slip of swimmer A
  slip_A = np.loadtxt(name_slip_A, skiprows=skiprows_slip_A)
      
  # Build body A
  orientation_A = Quaternion(np.array([1, 0, 0, 0]))
  b_A = body.Body(np.zeros(3), orientation_A, r_blobs_A, blob_radius_A)
  b_A.mobility_blobs = mob.rotne_prager_tensor
                                 
  # Compute lambda_A and U_A 
  # Build matrices
  M_A = b_A.calc_mobility_blobs(eta, blob_radius_A) 
  L_A, lower_A = scipy.linalg.cho_factor(M_A)
  L_A = np.triu(L_A)  
  K_A = b_A.calc_K_matrix()
  N_A = np.linalg.pinv(np.dot(K_A.T, scipy.linalg.cho_solve((L_A,lower_A), K_A, check_finite=False))) 
  
  # Compute blob foces
  lambda_A = udf.calc_blob_forces_swimmer(slip_A, N_A, K_A, L_A, lower_A)
  U_A = udf.calc_velocity_swimmer(slip_A, N_A, K_A, L_A, lower_A)
  output_name = output_prefix + '.U_A.dat'
  np.savetxt(output_name, U_A.reshape((1, 6)))

  # Rotate shell
  if frame_body > -1:
    quaternion_body = Quaternion(x_A[frame_body,3:7]) 
    R = quaternion_body.rotation_matrix()
    r_shell_rotated = np.dot(r_shell, R.T) + x_A[frame_body,0:3]

  # Compute velocity flow on sphere around swimmer
  v_S = mob.no_wall_mobility_trans_times_force_source_target_numba(b_A.get_r_vectors(), r_shell_rotated, lambda_A, blobs_radius_A, np.zeros(r_shell.shape[0]), eta)

  # Save velocity on shell
  result = np.zeros((v_S.size // 3, 7))
  result[:,0:3] = r_shell
  result[:,3] = uv_weights
  result[:,4:7] = v_S.reshape((v_S.size // 3, 3))
  output_name = output_prefix + '.shell_radius.' + str(shell_radius) + '.v_S.p.' + str(p) + '.dat'
  np.savetxt(output_name, result)

