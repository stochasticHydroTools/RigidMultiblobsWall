''' 
Script to calculate the average parallel, perpendicular, and
rotational mobilities for a tetrahedron and compare to the sphere case
with a scatter plot.
'''

from matplotlib import pyplot
import numpy as np
import sys
sys.path.append('..')

import icosohedron as ic
import sphere.sphere as sph
from quaternion_integrator.quaternion import Quaternion

def plot_scatter_icosohedron_mobilities(a, heights):
  '''
  Calculate parallel, perpendicular, and rotational mobilities
  at the given heights for an icosohedron with vertices radius a.
  Here we vary the ratio between Icosohedron length and vertex radius
  from 1.5 to 2.5.
  Compare these results to a sphere of radius 0.5
  '''
  symbols = {1.5: '*', 2.0: '.', 2.5: 's'}
  for d in [1.5, 2.0, 2.5]:
    ic.VERTEX_A = a
    ic.A = d*a
    x = []
    mu_parallel = []
    mu_perp = []
    mu_rotation = []
    mu_center_parallel = []
    mu_center_perp = []
    mu_center_rotation = []

    for h in heights:
      # Calculate 5 random orientations for heights.
      for k in range(5):
        theta = np.random.normal(0., 1., 4)
        theta = Quaternion(theta/np.linalg.norm(theta))
        location = [0., 0., h]
        mobility = ic.icosohedron_mobility([location], [theta])
        mobility_center = ic.icosohedron_center_mobility([location], [theta])
        x.append(h)
        mu_parallel.append(mobility[0, 0])
        mu_perp.append(mobility[2, 2])
        mu_rotation.append(mobility[3, 3])
        mu_center_parallel.append(mobility_center[0, 0])
        mu_center_perp.append(mobility_center[2, 2])
        mu_center_rotation.append(mobility_center[3, 3])
        
    pyplot.figure(1)
    pyplot.plot(x, mu_parallel, 'b' + symbols[d], label="d = %s a" % d)
    pyplot.plot(x, mu_center_parallel, 'g' + symbols[d], 
                label="d = %s a, center" % d)
    
    pyplot.figure(2)
    pyplot.plot(x, mu_perp, 'b' + symbols[d], label="d = %s a" % d)
    pyplot.plot(x, mu_center_perp, 'g' + symbols[d], 
                label="d = %s a, center" % d)

    
    pyplot.figure(3)
    pyplot.plot(x, mu_rotation, 'b' + symbols[d], label="d = %s a" % d)
    pyplot.plot(x, mu_center_rotation, 'g' + symbols[d], 
                label="d = %s a, center" % d)
        

  sphere_parallel = []
  sphere_perp  = []
  sphere_rotation = []
  orientation = [Quaternion([1., 0., 0., 0.])]
  for h in heights:
    location = [[0., 0., h]]
    sphere_mobility = sph.sphere_mobility(location, orientation)
    sphere_parallel.append(sphere_mobility[0, 0])
    sphere_perp.append(sphere_mobility[2, 2])
    sphere_rotation.append(sphere_mobility[3, 3])
  
  pyplot.figure(1)
  pyplot.plot(heights, sphere_parallel, 'k--', label='Sphere')
  pyplot.title('Parallel Mobility for Icosohedron, a = %s' % a)
  pyplot.legend(loc='best', prop={'size': 9})
  pyplot.savefig('./figures/IcosohedronParallelMobility.pdf')

  pyplot.figure(2)
  pyplot.plot(heights, sphere_perp, 'k--', label='Sphere')
  pyplot.title('Perpendicular Mobility for Icosohedron, a = %s' % a)
  pyplot.legend(loc='best', prop={'size': 9})
  pyplot.savefig('./figures/IcosohedronPerpendicularMobility.pdf')

  pyplot.figure(3)
  pyplot.plot(heights, sphere_rotation, 'k--', label='Sphere')
  pyplot.title('Rotational Mobility for Icosohedron, a = %s' % a)
  pyplot.legend(loc='best', prop={'size': 9})
  pyplot.savefig('./figures/IcosohedronRotationalMobility.pdf')
  

if __name__ == '__main__':
  
  a = 0.2
  heights = np.linspace(0.5, 7.0, 40)
  plot_scatter_icosohedron_mobilities(a, heights)
  


      

