
import numpy as np
import sys

#import rod_gmres as rod
sys.path.append('../')
from quaternion_integrator.quaternion import Quaternion
from . import mobility as mob


def get_r_vectors(location, orientation):
  '''
  '''
  initial_configuration = [np.array([2.1, 0., 0.]),
                           np.array([1.8, 0., 0.]),
                           np.array([1.5, 0., 0.]),
                           np.array([1.2, 0., 0.]),
                           np.array([0.9, 0., 0.]),
                           np.array([0.6, 0., 0.]),
                           np.array([0.3, 0., 0.]),
                           np.array([0., 0., 0.]),
                           np.array([0., 0.3, 0.]),
                           np.array([0., 0.6, 0.]),
                           np.array([0., 0.9, 0.]),
                           np.array([0., 1.2, 0.]),
                           np.array([0., 1.5, 0.]),
                           np.array([0., 1.8, 0.]),
                           np.array([0., 2.1, 0.])]

  for i in range(len(initial_configuration)):
      initial_configuration[i] += location
  
  return initial_configuration 



if __name__ == '__main__':


    print('# Start')
    
    # Create rod
    eta = 1.0
    a = 1.0
    location = [0., 0., 1]
    theta = np.random.normal(0., 1., 4)
    orientation = Quaternion([1., 0., 0., 0.])
    r_vectors = get_r_vectors(location, orientation)
    
    # Generate random forces
    force = np.random.randn(len(r_vectors), 3)
    
    print('r_vectors \n', r_vectors, '\n\n')
    print('force \n', force, '\n\n')

    
    velocities_from_gpu = mob.single_wall_mobility_trans_times_force_pycuda_single(r_vectors, force, eta, a)
    print('velocities_from_gpu \n', velocities_from_gpu, '\n\n')

    mobility = mob.boosted_single_wall_fluid_mobility(r_vectors, eta, a)

    u = np.zeros(len(r_vectors) * 3)
    for i in range(len(r_vectors)*3):
        for j in range(len(r_vectors)):
            for axis_j in range(3):
                u[i] += mobility[i][j*3+axis_j] * force[j][axis_j]
                pass
	   
    print('velocities_cpu \n', u, '\n\n')
    
    force_hstack=np.hstack(force)
    # print('force_hstack \n', force_hstack, '\n\n')
    u2 =  mob.boosted_mobility_vector_product(r_vectors, eta, a,force_hstack) 
    print('velocities_cpu Matrix_vector C++ \n', u2, '\n\n')
    
    print('Are the two velocities equal for tol=1e-08?', np.allclose(velocities_from_gpu, u, atol=1e-08))
    print('Are the two velocities equal for tol=1e-06?', np.allclose(velocities_from_gpu, u, atol=1e-06))
    print('Are the two velocities equal for tol=1e-04?', np.allclose(velocities_from_gpu, u, atol=1e-04))

    print('# End')
