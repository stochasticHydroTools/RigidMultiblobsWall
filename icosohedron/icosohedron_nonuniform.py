''' Just overwrite masses and make a torque calculator. '''

import numpy as np

import icosohedron as ic


M = [0.05/12. for _ in range(12)]
M[11] += 0.05

def nonuniform_torque_calculator(location, orientation):
  ''' Calculate torque based on a nonuniform icosohedron. '''
  r_vectors = ic.get_icosohedron_r_vectors(location[0], orientation[0])
  forces = []
  for mass in M:
    forces += [0., 0., -1.*mass]

  R = ic.calc_icosohedron_rot_matrix(r_vectors, location[0])
  return np.dot(R.T, forces)



