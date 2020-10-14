
import numpy as np
from functools import partial
from . import stochastic_forcing as stoch


import sys
sys.path.append('..')
from mobility import mobility as mob
from boomerang import boomerang as bm
from quaternion_integrator import quaternion as q


def mobility_mult(force = None, r_vectors = None, eta = None, radius = None):
  # return mob.single_wall_mobility_trans_times_force_pycuda(r_vectors, np.reshape(force, (len(force)/3, 3)), eta, radius)
  return mob.single_wall_fluid_mobility_product(r_vectors, np.reshape(force, force.size), eta, radius)

def get_eta(eta):
  return eta
  
def create_mobility_blobs(r_vectors, eta, a):
  mobility = []
  for i in range(len(r_vectors)):
    for j in range(3):
      f = np.zeros( (len(r_vectors),3) )
      f[i,j] = 1.0
      # v = np.reshape(mob.single_wall_mobility_trans_times_force_pycuda(r_vectors, f, eta, a), len(r_vectors)*3)
      v = np.reshape(mob.single_wall_fluid_mobility_product(r_vectors, np.reshape(f, f.size), eta, a), r_vectors.size)
      mobility.append(v)
  return mobility



if __name__ == '__main__':

  # Define objects
  location = [0., 0., 100000.]
  theta = np.random.normal(0., 1., 4)
  # orientation = Quaternion(theta/np.linalg.norm(theta))
  orientation = q.Quaternion([1., 0., 0., 0.])
  r_vectors = bm.get_boomerang_r_vectors_15(location, orientation)
  mobility_body = bm.force_and_torque_boomerang_mobility(r_vectors, location)
  mobility_blobs = create_mobility_blobs(r_vectors, bm.ETA, bm.A)
  mobility_body = mobility_blobs
  

  # print mobility_body

  # Create random vector
  np.random.seed(0)
  z = np.random.normal(0.0, 1.0, len(mobility_body))
  # z = np.ones(len(mobility_body))
  # z = np.zeros(len(mobility_body))
  # z[0] = 1.0
    
  # mobility_body = np.zeros( (6,6) )
  # for i in range(len(mobility_body)):
  #   mobility_body[i,i] = 1.0
  # mobility_body[0,1] = 1.0
  # mobility_body = 0.5 * (mobility_body + mobility_body.T)

  # Compute noise 
  noise_chol = stoch.stochastic_forcing_cholesky(mobility_body, 1.0, z = z)
  print('noise_chol     ', noise_chol, '\n')
  noise_eig = stoch.stochastic_forcing_eig(mobility_body, 1.0, z = z)
  print('noise_eig      ', noise_eig, '\n')
  noise_eig = stoch.stochastic_forcing_eig_symm(mobility_body, 1.0, z = z)
  print('noise_eig_symm ', noise_eig, '\n')

  
  noise_lanczos = np.copy(noise_eig)
  noise_lanczos[0] = 0.
  if 1:
    a = bm.A
    eta = bm.ETA
    mobility_mult_partial = partial(mobility_mult, 
                                    r_vectors = r_vectors, 
                                    eta = eta,
                                    radius = a)   
    noise_lanczos, it = stoch.stochastic_forcing_lanczos(factor = 1.0, 
                                                         max_iter=1000, 
                                                         tolerance=1e-08, 
                                                         name='residual.dat', 
                                                         mobility_mult=mobility_mult_partial,
                                                         z = z)
  else:
    noise_lanczos, it = stoch.stochastic_forcing_lanczos(factor = 1.0, 
                                                         max_iter=500, 
                                                         tolerance=1e-8, 
                                                         name='residual.dat', 
                                                         mobility = mobility_body,
                                                         z = z)
  print('\n', 'noise     ', noise_lanczos, '\n')
  print('number of iterations', it)

  if 1:
    print('all close, tol=1e-04', np.allclose(noise_eig, noise_lanczos, atol=1e-04))
    print('all close, tol=1e-06', np.allclose(noise_eig, noise_lanczos, atol=1e-06))
    print('all close, tol=1e-08', np.allclose(noise_eig, noise_lanczos, atol=1e-08))
    print('all close, tol=1e-10', np.allclose(noise_eig, noise_lanczos, atol=1e-10))
    print('relative error =    ', np.linalg.norm(noise_eig - noise_lanczos) / np.linalg.norm(noise_eig))
    # print('\nmobility_body\n', mobility_body)
  else:
    print('\nmobility_blobs\n', mobility_blobs)

  # v = [np.random.normal(0.0, 1.0, len(r_vectors)*3)]
  # vv = []
  # print v[0]
  # for i in range(len(r_vectors)):
  #   vec = np.zeros(3)
  #   print vec
  #   for j in range(3):
  #     vec[j] = v[0][i*3 + j]
  #   vv.append(vec)

  # vv = np.random.randn(len(r_vectors), 3)
  # w = mobility_mult(r_vectors, v[0], eta, a)
  # w = mob.single_wall_mobility_times_force_pycuda(r_vectors, vv, eta, a)
  # single_wall_mobility_times_force_pycuda(r_vectors, force, eta, a)



  
  print('# End')
  
