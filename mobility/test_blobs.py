
import numpy as np
import sys
import imp

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
from . import mobility as mob
from general_application_utils import timer




if __name__ == '__main__':


    print('# Start')
    
    # Create blobs
    N = 1000
    eta = 7.0
    a = 0.13
    r_vectors = 5 * a * np.random.rand(N, 3) 
    L = np.array([0., 0., 0.])

    # Generate random forces
    force = np.random.randn(len(r_vectors), 3) 

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

    if False:
        np.set_printoptions(precision=6)
        print('no_wall_numba  ', u_no_wall_numba)
        print('numba          ', u_numba)
        print('pycuda         ', u_gpu)
        print('diff           ', u_numba - u_gpu)


    #print 'mobility_no_wall_loops \n', mobility_no_wall_loops
    #print 'mobility_no_wall \n', mobility_no_wall

    

    print('=================== No wall tests ===================')
    print('|u_no_wall_full - u_no_wall_loops_full| / |u_no_wall_loops_full|   = ', np.linalg.norm(u_no_wall_full - u_no_wall_loops_full) / np.linalg.norm(u_no_wall_loops_full))
    if found_pycuda:
        print('|u_no_wall_pycuda - u_no_wall_loops_full| / |u_no_wall_loops_full| = ', np.linalg.norm(u_no_wall_pycuda - u_no_wall_loops_full) / np.linalg.norm(u_no_wall_loops_full))
    print('|u_no_wall_numba - u_no_wall_loops_full| / |u_no_wall_loops_full|  = ', np.linalg.norm(u_no_wall_numba - u_no_wall_loops_full) / np.linalg.norm(u_no_wall_loops_full))
    print('=================== Wall tests ===================')
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

    timer('', print_all=True, clean_all=True)




    # ==========================================================
    # Rot tests
    # ==========================================================
    print('\n\n\n\n')
    print('==========================================================')
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



        timer('', print_all=True)
    
    print('# End')
