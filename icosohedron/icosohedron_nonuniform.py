''' Just overwrite masses and make a torque calculator. '''

import argparse
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


def bin_height_and_theta(location, orientation, height_hist, theta_hist):
  ''' Bin the height (of the geometric center) of the icosohedron, and 
  theta, the angle between the vector to the heavy blob (last r_vector) 
  and the negative z axis.'''
  r_vectors  = ic.get_icosohedron_r_vectors(location, orientation)
  
  heavy_blob_vector =  (location - r_vectors[-1])

  theta = np.arccos(heavy_blob_vector[2]/)


if __name__ == '__main__':
  # Get command line arguments.
  parser = argparse.ArgumentParser(description='Run Simulation of nonuniform '
                                   'icosohedron with Fixman and RFD '
                                   'schemes, and bin the resulting '
                                   'height and theta distribution.')
  parser.add_argument('-dt', dest='dt', type=float,
                      help='Timestep to use for runs.')
  parser.add_argument('-N', dest='n_steps', type=int,
                      help='Number of steps to take for runs.')
  parser.add_argument('--data-name', dest='data_name', type=str,
                      default='',
                      help='Optional name added to the end of the '
                      'data file.  Useful for multiple runs '
                      '(--data_name=run-1).')
  parser.add_argument('--profile', dest='profile', type=bool, default=False,
                      help='True or False: Do we profile this run or not.')


