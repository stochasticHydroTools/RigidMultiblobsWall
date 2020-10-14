
import numpy as np
import sys

sys.path.append('../')
import boomerang as bm
from quaternion_integrator.quaternion import Quaternion
from . import mobility as mob
try:
  import mobility_fmm as fmm
  fmm_found = True
except ImportError:
  fmm_found = False

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

  # rotation_matrix = orientation.rotation_matrix()
  rotated_configuration = np.empty([len(initial_configuration), 3])
  for i, vec in enumerate(initial_configuration):
    rotated_configuration[i] = vec + np.array(location)

  # for i in range(len(initial_configuration)):
  # initial_configuration[i] += location
  
  return rotated_configuration



if __name__ == '__main__':

    print('# Start')
    
    # Create rod
    eta = 2e+06
    a = 1e-06
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
    print('Are the two velocities equal to tol=1e-08?', np.allclose(velocities_from_gpu, u, atol=1e-08))
    print('Are the two velocities equal to tol=1e-06?', np.allclose(velocities_from_gpu, u, atol=1e-06))
    print('Are the two velocities equal to tol=1e-04?', np.allclose(velocities_from_gpu, u, atol=1e-04))
    print('relative difference = ', np.linalg.norm(velocities_from_gpu - u) / np.linalg.norm(u))
    
    if fmm_found == True:
      print('\n\n\n')
      print(fmm.fmm_stokeslet_half.__doc__)
      print('\n\n\n')
    
      ier = 0
      iprec = 5
      n = r_vectors.size // 3
      n3 = r_vectors.size
      print('n = ', n, '  n3 = ', n3)
      r_vectors_fortran = np.copy(r_vectors.T, order='F')
      print('r_vectors_fortran \n', r_vectors_fortran.shape)
      print('r_vectors_fortran \n', r_vectors_fortran)
      force_fortran = np.copy(force.T, order='F')
      u_fortran = np.empty_like(r_vectors_fortran, order='F')
   
      fmm.fmm_stokeslet_half(r_vectors_fortran, force_fortran, u_fortran, ier, iprec, a, eta, n)
      print('ier =', ier)
      u_fortran = np.reshape(u_fortran.T, u_fortran.size)
      print('u', u_fortran)
      
      print('Are the two velocities equal to tol=1e-08?', np.allclose(u_fortran, u, atol=1e-08))
      print('Are the two velocities equal to tol=1e-06?', np.allclose(u_fortran, u, atol=1e-06))
      print('Are the two velocities equal to tol=1e-04?', np.allclose(u_fortran, u, atol=1e-04))
      print('relative difference = ', np.linalg.norm(u_fortran - u) / np.linalg.norm(u))
    
    print('# End')
