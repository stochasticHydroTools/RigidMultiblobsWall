import numpy as np
import sys
import imp
from functools import partial

try: 
  imp.find_module('pycuda')
  found_pycuda = True
except ImportError:
  found_pycuda = False
try:
  import mobility_cpp
  found_cpp = True
except ImportError:
  try:
    from .mobility import mobility_cpp
    found_cpp = True
  except ImportError:
    pass

sys.path.append('../')
import mobility as mob
from general_application_utils import timer


if __name__ == '__main__':

  print('# Start')
    
  # Create blobs
  N = 1000
  eta = 7.0
  a = 0.13
  r_vectors = 5 * a * np.random.rand(N, 3)
  L = np.array([0., 0., 0.])
  N_max_to_print_velocities = 2
  test_no_wall = True
  test_single_wall = True
  test_free_surface = True
  test_rot = True
    

  # Generate random forces
  force = np.random.randn(len(r_vectors), 3) 

  if test_no_wall:
    # ================================================================
    # NO WALL TESTS
    # ================================================================
    timer('zz_no_wall_loops_full_matrix')
    mobility_no_wall_loops = mob.rotne_prager_tensor_loops(r_vectors, eta, a)
    u_no_wall_loops_full = np.dot(mobility_no_wall_loops, force.flatten())
    timer('zz_no_wall_loops_full_matrix')
    
    timer('zz_no_wall_full_matrix')
    mobility_no_wall = mob.rotne_prager_tensor(r_vectors, eta, a)
    u_no_wall_full = np.dot(mobility_no_wall, force.flatten())
    timer('zz_no_wall_full_matrix')
    
    u_no_wall_numba = mob.no_wall_mobility_trans_times_force_numba(r_vectors, force, eta, a)
    timer('zz_no_wall_numba')
    u_no_wall_numba = mob.no_wall_mobility_trans_times_force_numba(r_vectors, force, eta, a)
    timer('zz_no_wall_numba')

    if found_pycuda:
      u_no_wall_pycuda = mob.no_wall_mobility_trans_times_force_pycuda(r_vectors, force, eta, a)
      timer('zz_no_wall_pycuda')
      u_no_wall_pycuda = mob.no_wall_mobility_trans_times_force_pycuda(r_vectors, force, eta, a)
      timer('zz_no_wall_pycuda')

    print('=================== No wall tests ===================')
    if N <= N_max_to_print_velocities:
      np.set_printoptions(precision=6)
      print('u_no_wall_full       = ', u_no_wall_full)
      print('u_no_wall_full_loops = ', u_no_wall_loops_full)
      print('u_no_wall_numba      = ', u_no_wall_numba)
      if found_pycuda:      
        print('u_no_wall_pycuda     = ', u_no_wall_gpu)
      print(' ')
    
    print('|u_no_wall_full - u_no_wall_loops_full| / |u_no_wall_loops_full|   = ', np.linalg.norm(u_no_wall_full - u_no_wall_loops_full) / np.linalg.norm(u_no_wall_loops_full))
    if found_pycuda:
        print('|u_no_wall_pycuda - u_no_wall_loops_full| / |u_no_wall_loops_full| = ', np.linalg.norm(u_no_wall_pycuda - u_no_wall_loops_full) / np.linalg.norm(u_no_wall_loops_full))
    print('|u_no_wall_numba - u_no_wall_loops_full| / |u_no_wall_loops_full|  = ', np.linalg.norm(u_no_wall_numba - u_no_wall_loops_full) / np.linalg.norm(u_no_wall_loops_full))
    print('\n\n\n')


  if test_single_wall:
    # ================================================================
    # WALL TESTS
    # ================================================================
    timer('python_loops')
    mobility_loops = mob.single_wall_fluid_mobility_loops(r_vectors, eta, a)
    u_loops = np.dot(mobility_loops, force.flatten())
    timer('python_loops')
    
    timer('python')
    mobility = mob.single_wall_fluid_mobility(r_vectors, eta, a)
    u = np.dot(mobility, force.flatten())
    timer('python')

    u_numba = mob.single_wall_mobility_trans_times_force_numba(r_vectors, force, eta, a)
    timer('numba')
    u_numba = mob.single_wall_mobility_trans_times_force_numba(r_vectors, force, eta, a, periodic_length = L)
    timer('numba')

    if found_pycuda:
      u_gpu = mob.single_wall_mobility_trans_times_force_pycuda(r_vectors, force, eta, a)
      timer('pycuda')
      u_gpu = mob.single_wall_mobility_trans_times_force_pycuda(r_vectors, force, eta, a, periodic_length = L)
      timer('pycuda')
      
    if found_cpp:
      timer('cpp')
      u_cpp = mob.single_wall_mobility_trans_times_force_cpp(r_vectors, force, eta, a)
      timer('cpp')

    print('=================== Wall tests ===================')      
    if N <= N_max_to_print_velocities:
      np.set_printoptions(precision=6)
      print('u      = ', u)
      if found_pycuda:      
        print('pycuda =  ', u_gpu)      
      print('numba  = ', u_numba)
      if found_cpp:
        print('cpp    = ', u_cpp)
      print(' ')

    print('|u - u_loops| / |u_loops|                                          = ', np.linalg.norm(u - u_loops) / np.linalg.norm(u_loops))
    print('|u_numba - u_loops| / |u_loops|                                    = ', np.linalg.norm(u_numba - u_loops) / np.linalg.norm(u_loops))
    if found_pycuda:
        print('|u_gpu - u_loops| / |u_loops|                                      = ', np.linalg.norm(u_gpu - u_loops) / np.linalg.norm(u_loops))

    if found_cpp:
        print('|u_cpp - u_loops| / |u_loops|                                      = ', np.linalg.norm(u_cpp - u_loops) / np.linalg.norm(u_loops))

    if L[0] > 0. or L[1] > 0.:
        print('===================== Pseudo-periodic tests =============================')
        if found_pycuda:
            print('|u_numba - u_gpu| / |u_gpu|                                    = ', np.linalg.norm(u_numba - u_gpu) / np.linalg.norm(u_gpu))
    print('\n\n\n')

    
  if test_rot:
    # ==========================================================
    # Rot tests
    # ==========================================================
    print('=================== Rot tests ====================')
    if found_pycuda:
      timer('u_no_wall_trans_times_torque_gpu')
      u_no_wall_trans_times_torque_gpu = mob.no_wall_mobility_trans_times_torque_pycuda(r_vectors, force, eta, a, periodic_length = L)
      timer('u_no_wall_trans_times_torque_gpu')
      
      u_no_wall_trans_times_torque_numba = mob.no_wall_mobility_trans_times_torque_numba(r_vectors, force, eta, a, periodic_length = L)
      timer('u_no_wall_trans_times_torque_numba')
      u_no_wall_trans_times_torque_numba = mob.no_wall_mobility_trans_times_torque_numba(r_vectors, force, eta, a, periodic_length = L)
      timer('u_no_wall_trans_times_torque_numba')
      print('|u_no_wall_trans_times_torque_numba - u_no_wall_trans_times_torque_gpu| / |u_no_wall_trans_times_torque_gpu|      = ', \
            np.linalg.norm(u_no_wall_trans_times_torque_numba - u_no_wall_trans_times_torque_gpu) / np.linalg.norm(u_no_wall_trans_times_torque_gpu))

      u_wall_trans_times_torque_gpu = mob.single_wall_mobility_trans_times_torque_pycuda(r_vectors, force, eta, a, periodic_length = L)
      timer('u_wall_trans_times_torque_gpu')
      u_wall_trans_times_torque_gpu = mob.single_wall_mobility_trans_times_torque_pycuda(r_vectors, force, eta, a, periodic_length = L)
      timer('u_wall_trans_times_torque_gpu')

      u_wall_trans_times_torque_numba = mob.single_wall_mobility_trans_times_torque_numba(r_vectors, force, eta, a, periodic_length = L)
      timer('u_wall_trans_times_torque_numba')
      u_wall_trans_times_torque_numba = mob.single_wall_mobility_trans_times_torque_numba(r_vectors, force, eta, a, periodic_length = L)
      timer('u_wall_trans_times_torque_numba')
      print('|u_wall_trans_times_torque_numba - u_wall_trans_times_torque_gpu| / |u_wall_trans_times_torque_gpu|               = ', \
            np.linalg.norm(u_wall_trans_times_torque_numba - u_wall_trans_times_torque_gpu) / np.linalg.norm(u_wall_trans_times_torque_gpu))


      timer('u_no_wall_rot_times_force_gpu')
      u_no_wall_rot_times_force_gpu = mob.no_wall_mobility_rot_times_force_pycuda(r_vectors, force, eta, a, periodic_length = L)
      timer('u_no_wall_rot_times_force_gpu')

      u_no_wall_rot_times_force_numba = mob.no_wall_mobility_rot_times_force_numba(r_vectors, force, eta, a, periodic_length = L)
      timer('u_no_wall_rot_times_force_numba')
      u_no_wall_rot_times_force_numba = mob.no_wall_mobility_rot_times_force_numba(r_vectors, force, eta, a, periodic_length = L)
      timer('u_no_wall_rot_times_force_numba')
      print('|u_no_wall_rot_times_force_numba - u_no_wall_rot_times_force_gpu| / |u_no_wall_rot_times_force_gpu|               = ', 
            np.linalg.norm(u_no_wall_rot_times_force_numba - u_no_wall_rot_times_force_gpu) / np.linalg.norm(u_no_wall_rot_times_force_gpu))


      timer('u_single_wall_rot_times_force_gpu')
      u_single_wall_rot_times_force_gpu = mob.single_wall_mobility_rot_times_force_pycuda(r_vectors, force, eta, a, periodic_length = L)
      timer('u_single_wall_rot_times_force_gpu')

      u_single_wall_rot_times_force_numba = mob.single_wall_mobility_rot_times_force_numba(r_vectors, force, eta, a, periodic_length = L)
      timer('u_single_wall_rot_times_force_numba')
      u_single_wall_rot_times_force_numba = mob.single_wall_mobility_rot_times_force_numba(r_vectors, force, eta, a, periodic_length = L)
      timer('u_single_wall_rot_times_force_numba')
      print('|u_single_wall_rot_times_force_numba - u_single_wall_rot_times_force_gpu| / |u_single_wall_rot_times_force_gpu|   = ', 
            np.linalg.norm(u_single_wall_rot_times_force_numba - u_single_wall_rot_times_force_gpu) / np.linalg.norm(u_single_wall_rot_times_force_gpu))


      timer('u_no_wall_rot_times_torque_gpu')
      u_no_wall_rot_times_torque_gpu = mob.no_wall_mobility_rot_times_torque_pycuda(r_vectors, force, eta, a, periodic_length = L)
      timer('u_no_wall_rot_times_torque_gpu')
      
      u_no_wall_rot_times_torque_numba = mob.no_wall_mobility_rot_times_torque_numba(r_vectors, force, eta, a, periodic_length = L)
      timer('u_no_wall_rot_times_torque_numba')
      u_no_wall_rot_times_torque_numba = mob.no_wall_mobility_rot_times_torque_numba(r_vectors, force, eta, a, periodic_length = L)
      timer('u_no_wall_rot_times_torque_numba')
      print('|u_no_wall_rot_times_torque_numba - u_no_wall_rot_times_torque_gpu| / |u_no_wall_rot_times_torque_gpu|            = ', 
            np.linalg.norm(u_no_wall_rot_times_torque_numba - u_no_wall_rot_times_torque_gpu) / np.linalg.norm(u_no_wall_rot_times_torque_gpu))
      
      timer('u_single_wall_rot_times_force_gpu')
      u_single_wall_rot_times_torque_gpu = mob.single_wall_mobility_rot_times_torque_pycuda(r_vectors, force, eta, a, periodic_length = L)
      timer('u_single_wall_rot_times_force_gpu')
      
      u_single_wall_rot_times_torque_numba = mob.single_wall_mobility_rot_times_torque_numba(r_vectors, force, eta, a, periodic_length = L)
      timer('u_single_wall_rot_times_torque_numba')
      u_single_wall_rot_times_torque_numba = mob.single_wall_mobility_rot_times_torque_numba(r_vectors, force, eta, a, periodic_length = L)
      timer('u_single_wall_rot_times_torque_numba')
      print('|u_single_wall_rot_times_torque_numba - u_single_wall_rot_times_torque_gpu| / |u_single_wall_rot_times_torque_gpu| = ', 
            np.linalg.norm(u_single_wall_rot_times_torque_numba - u_single_wall_rot_times_torque_gpu) / np.linalg.norm(u_single_wall_rot_times_torque_gpu))

      if False:
        np.set_printoptions(precision=6)
        print('no_wall = ', u_no_wall_trans_times_torque_gpu)
        print('gpu     = ', u_wall_trans_times_torque_gpu)
        print('numba   = ', u_wall_trans_times_torque_numba)
      print('\n\n\n')


  if test_free_surface:
    # ================================================================
    # Free surface
    # ================================================================

    def free_surface(r_vectors, force, eta, a):
      # Self interaction above surface
      M = mob.rotne_prager_tensor(r_vectors, eta, a)

      # Interaction with images below surface
      N = r_vectors.shape[0]
      r_vectors_tmp = np.zeros((N * 2, 3))
      r_vectors_tmp[0:N] = r_vectors
      r_vectors_tmp[N:] = r_vectors
      r_vectors_tmp[N:,2] = -r_vectors[:,2]
      M_image = mob.rotne_prager_tensor(r_vectors_tmp, eta, a)

      # Add mobilities
      M[0::3, 0::3] += M_image[0:3*N:3, 3*N::3]
      M[0::3, 1::3] += M_image[0:3*N:3, 3*N+1::3]
      M[0::3, 2::3] -= M_image[0:3*N:3, 3*N+2::3]
      M[1::3, 0::3] += M_image[1:3*N:3, 3*N::3]
      M[1::3, 1::3] += M_image[1:3*N:3, 3*N+1::3]
      M[1::3, 2::3] -= M_image[1:3*N:3, 3*N+2::3]
      M[2::3, 0::3] += M_image[2:3*N:3, 3*N::3]
      M[2::3, 1::3] += M_image[2:3*N:3, 3*N+1::3]
      M[2::3, 2::3] -= M_image[2:3*N:3, 3*N+2::3]

      # Return velocities
      return np.dot(M, force.flatten())     
            
    u_free_surface = free_surface(r_vectors, force, eta, a)
    u_free_surface_numba = mob.free_surface_mobility_trans_times_force_numba(r_vectors, force, eta, a)
    timer('numba')
    u_free_surface_numba = mob.free_surface_mobility_trans_times_force_numba(r_vectors, force, eta, a, periodic_length = L)
    timer('numba')
    
    free_surface_mobility_trans_times_force_source_target_numba = partial(mob.mobility_radii_trans_times_force, radius_blobs=np.ones(N) * a, function=mob.free_surface_mobility_trans_times_force_source_target_numba)    
    u_free_surface_source_target_numba = free_surface_mobility_trans_times_force_source_target_numba(r_vectors, force, eta, a)
    timer('numba')
    u_free_surface_source_target_numba = free_surface_mobility_trans_times_force_source_target_numba(r_vectors, force, eta, a, periodic_length = L)
    timer('numba')
    
    if found_pycuda:
      u_free_surface_gpu = mob.free_surface_mobility_trans_times_force_pycuda(r_vectors, force, eta, a)
      timer('pycuda')
      u_free_surface_gpu = mob.free_surface_mobility_trans_times_force_pycuda(r_vectors, force, eta, a, periodic_length = L)
      timer('pycuda')
    
    print('=================== Free surface tests ===========')
    if N <= N_max_to_print_velocities:
      np.set_printoptions(precision=6)
      print('u        = ', u_free_surface)
      if found_pycuda:      
        print('u_pycuda =  ', u_free_surface_gpu)
      print('u_numba  = ', u_free_surface_numba)
      print('u_src_trg= ', u_free_surface_source_target_numba)      
      print(' ')

    print('|u_numba - u| / |u|                                                = ', np.linalg.norm(u_free_surface_numba - u_free_surface) / np.linalg.norm(u_free_surface))
    print('|u_sour_target_numba - u_numba| / |u_numba|                        = ', np.linalg.norm(u_free_surface_source_target_numba - u_free_surface_numba) / np.linalg.norm(u_free_surface_numba)) 
    if found_pycuda:
        print('|u_gpu - u_numba| / |u_numba|                                      = ', np.linalg.norm(u_free_surface_gpu - u_free_surface_numba) / np.linalg.norm(u_free_surface_numba))
    print('\n\n\n')
    

  # Print times
  timer('', print_all=True, clean_all=True)
  print('# End')
