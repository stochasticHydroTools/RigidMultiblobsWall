''' Check the eigenvalues of Oseen with zero diagonal. '''

import sys
import numpy as np
from matplotlib import pyplot

def oseen_tensor_zero_diagonal(points):
  '''
  Calculate the oseen tensor with diagonal blocks set to zero.  This
  is just for testing to see if the mobility without the diagonal
  piece is PSD.  Assume points is a list of np arrays, each member is
  an array of length 3 indicating coords.
  '''
  dim = len(points)
  oseen_tensor = np.array([np.zeros(3*dim) for _ in range(3*dim)])
  for i in range(dim):
    for j in range(dim):
      if i != j:
        r = points[i] - points[j]
        r_norm = np.linalg.norm(r)
        for k in range(3):
          for l in range(3):
            oseen_tensor[i*3 + k, j*3 + l] = ((k == l)*1./r_norm + 
                                              r[k]*r[l]/(r_norm**3))

  # Unit viscosity, doesn't matter for SPD.
  oseen_tensor = oseen_tensor/(8*np.pi)   
  return oseen_tensor


if __name__ == '__main__':
  n_times = int(sys.argv[1])
  n_parts = 10
  def min_eig(x):
    return min(np.linalg.eigvals(x))

  def min_dist(points):
    min_distance = 100000000.
    for i in range(len(points)):
      for j in range(i+1, len(points)):
        dist = np.linalg.norm(points[i] - points[j])
        if dist < min_distance:
          min_distance = dist
    return min_distance


  dists = []
  smallest_eigs = []
  for _ in range(n_times):  
    points = [np.random.normal(0., 2., 3) for _ in range(n_parts)]
    dists.append(min_dist(points))
    oseen_tensor = oseen_tensor_zero_diagonal(points)
    smallest_eigs.append(min_eig(oseen_tensor))
   
  x = np.linspace(0.2, 3., 100)
  pyplot.plot(dists, 1./(6.*np.pi*np.array(smallest_eigs)), 'b*', label='Minimum O eigenvalue')
#  pyplot.plot(x,  -0.1/x, 'k--', label='-const/x')
  pyplot.legend(loc='best', prop={'size': 9})
  pyplot.title('Min Eigenvalues of Oseen Tensor with 0 diag, %d particles' % n_parts)
  pyplot.ylabel('1./(6 * pi * Minimum Eigenvalue)')
  pyplot.xlabel('Minimum distance between points')
  pyplot.savefig('./OseenEigs.pdf')
  
