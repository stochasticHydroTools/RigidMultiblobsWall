import numpy as np
import sys
import imp

try: 
    imp.find_module('pycuda')
    found_pycuda = True
except ImportError:
    found_pycuda = False

sys.path.append('../')
from quaternion_integrator.quaternion import Quaternion
import mobility as mob
from utils import timer




if __name__ == '__main__':


    print '# Start'
    
    # Create blobs
    # np.random.seed(4)
    eta = 7.0
    a = 1.1
    location = [0., 0., 1]
    theta = np.random.normal(0., 1., 4)
    orientation = Quaternion([1., 0., 0., 0.])
    r_vectors = 5 * a * np.random.rand(1000, 3) + np.array([0., 0., 0.])
    
    L = np.array([10., 10., 0.])

    # Generate random forces
    force = np.random.randn(len(r_vectors), 3) 
    force[:,:] = 0.
    force[0, 0] = 1.0
    
    print 'r_vectors \n', r_vectors, '\n\n'
    # print 'force \n', force, '\n\n'

    print 'r_vectors =', r_vectors.shape
    print 'force =', force.shape, '\n\n'


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
    
    timer('boost_full_matrix')
    mobility_boost = mob.boosted_single_wall_fluid_mobility(r_vectors, eta, a)
    u_boost_full = np.dot(mobility_boost, force.flatten())
    timer('boost_full_matrix')

    timer('boost')
    force_hstack=np.hstack(force)
    u_boost =  mob.boosted_mobility_vector_product(r_vectors, force_hstack, eta, a, periodic_length = L) 
    timer('boost')

    u_numba = mob.single_wall_mobility_trans_times_force_numba(r_vectors, force, eta, a)
    timer('numba')
    u_numba = mob.single_wall_mobility_trans_times_force_numba(r_vectors, force, eta, a, periodic_length = L)
    timer('numba')

    if found_pycuda:
        u_gpu = mob.single_wall_mobility_trans_times_force_pycuda(r_vectors, force, eta, a)
        timer('pycuda')
        u_gpu = mob.single_wall_mobility_trans_times_force_pycuda(r_vectors, force, eta, a, periodic_length = L)
        timer('pycuda')
    

    if False:
        np.set_printoptions(precision=6)
        print 'no_wall_numba  ', u_no_wall_numba
        print 'numba          ', u_numba
        print 'pycuda         ', u_gpu
        print 'diff           ', u_numba - u_gpu


    #print 'mobility_no_wall_loops \n', mobility_no_wall_loops
    #print 'mobility_no_wall \n', mobility_no_wall

    

    print('===================================================')
    print '|u_no_wall_full - u_no_wall_loops_full| / |u_no_wall_loops_full|   = ', np.linalg.norm(u_no_wall_full - u_no_wall_loops_full) / np.linalg.norm(u_no_wall_loops_full)
    print '|u_no_wall_pycuda - u_no_wall_loops_full| / |u_no_wall_loops_full| = ', np.linalg.norm(u_no_wall_pycuda - u_no_wall_loops_full) / np.linalg.norm(u_no_wall_loops_full)
    print '|u_no_wall_numba - u_no_wall_loops_full| / |u_no_wall_loops_full|  = ', np.linalg.norm(u_no_wall_numba - u_no_wall_loops_full) / np.linalg.norm(u_no_wall_loops_full)
    print('===================================================')
    print '|u - u_loops| / |u_loops|                                          = ', np.linalg.norm(u - u_loops) / np.linalg.norm(u_loops)
    print '|u_boost_full - u_loops| / |u_loops|                               = ', np.linalg.norm(u_boost_full - u_loops) / np.linalg.norm(u_loops)
    print '|u_boost - u_loops| / |u_loops|                                    = ', np.linalg.norm(u_boost - u_loops) / np.linalg.norm(u_loops)
    print '|u_numba - u_loops| / |u_loops|                                    = ', np.linalg.norm(u_numba - u_loops) / np.linalg.norm(u_loops)
    if found_pycuda:
        print '|u_gpu - u_loops| / |u_loops|                                      = ', np.linalg.norm(u_gpu - u_loops) / np.linalg.norm(u_loops)

    if L[0] > 0. or L[1] > 0.:
        print('===================================================')
        print '|u_numba - u_boost| / |u_boost|                                    = ', np.linalg.norm(u_numba - u_boost) / np.linalg.norm(u_boost)
        print '|u_gpu - u_boost| / |u_boost|                                      = ', np.linalg.norm(u_gpu - u_boost) / np.linalg.norm(u_boost)



    timer('', print_all=True)

    print '# End'
