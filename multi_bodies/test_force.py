
import numpy as np
import sys
import imp
sys.path.append('../')

from general_application_utils import timer
import forces_numba
import multi_bodies_functions as mbf

try: 
  imp.find_module('pycuda')
  found_pycuda = True
except ImportError:
  found_pycuda = False
if found_pycuda:
  from . import forces_pycuda   

try:
  import forces_cpp
  found_cpp = True
except ImportError:
  found_cpp = False
  pass


if __name__ == '__main__':
  print('# Start')

  N = 100
  a = 0.13
  b = 0.01
  eps = 3.92
  L = np.array([0.0, 0.0, 0.0])
  r_vectors = np.random.rand(N, 3) * 10.0

  if found_pycuda:
    force_pycuda = forces_pycuda.calc_blob_blob_forces_pycuda(r_vectors, blob_radius=a, debye_length=b, repulsion_strength=eps, periodic_length=L)
    timer('pycuda')
    force_pycuda = forces_pycuda.calc_blob_blob_forces_pycuda(r_vectors, blob_radius=a, debye_length=b, repulsion_strength=eps, periodic_length=L)
    timer('pycuda')

  force_numba_tree = forces_numba.calc_blob_blob_forces_tree_numba(r_vectors+1, blob_radius=a, debye_length=b, repulsion_strength=eps, periodic_length=L)
  timer(' ')
  timer(' ', clean_all=True)
  timer('numba_tree')
  force_numba_tree = forces_numba.calc_blob_blob_forces_tree_numba(r_vectors, blob_radius=a, debye_length=b, repulsion_strength=eps, periodic_length=L)
  timer('numba_tree')
    
  force_numba = forces_numba.calc_blob_blob_forces_numba(r_vectors, blob_radius=a, debye_length=b, repulsion_strength=eps, periodic_length=L)
  timer('numba')
  force_numba = forces_numba.calc_blob_blob_forces_numba(r_vectors, blob_radius=a, debye_length=b, repulsion_strength=eps, periodic_length=L)
  timer('numba')

  if N < 2000:
    timer('python')
    force_python = mbf.calc_blob_blob_forces_python(r_vectors, blob_radius=a, debye_length=b, repulsion_strength=eps, periodic_length=L)
    timer('python')

  if found_cpp:
    timer('cpp')
    force_cpp = mbf.calc_blob_blob_forces_cpp(r_vectors, blob_radius=a, debye_length=b, repulsion_strength=eps, periodic_length=L)
    timer('cpp')



  if N < 3:
    if found_pycuda:
      print('pycuda = ', force_pycuda)
    print('numba = ', force_numba)
    print('\n\n')


  print('norm(force_numba) = ', np.linalg.norm(force_numba))
  if N < 2000:
    print('|f_numba - f_python| / |f_python| = ', np.linalg.norm(force_numba - force_python) / np.linalg.norm(force_python))
    print('|f_numba_tree - f_python| / |f_python| = ', np.linalg.norm(force_numba_tree - force_python) / np.linalg.norm(force_python))
  else:
    print('|f_numba_tree - f_numba| / |f_numba| = ', np.linalg.norm(force_numba_tree - force_numba) / np.linalg.norm(force_numba))
  if found_pycuda:
    print('|f_numba - f_pycuda| / |f_pycuda| = ', np.linalg.norm(force_numba - force_pycuda) / np.linalg.norm(force_pycuda))
  if found_cpp:
    print('|f_cpp - f_numba| / |f_numba| = ', np.linalg.norm(force_cpp - force_numba) / np.linalg.norm(force_numba))




  timer('', print_all=True)
