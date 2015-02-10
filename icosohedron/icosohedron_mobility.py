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
  symbols = {1.5: '^', 2.0: '.', 2.5: 's', 3.0: 'v'}
  orientation = [Quaternion([1., 0., 0., 0.])]
  far_location = [[0., 0., 200.]]
  sphere_mobility_theory = sph.sphere_mobility(far_location, orientation)
  for d in [1.5, 2.0, 2.5, 3.0]:
    ic.VERTEX_A = a
    ic.A = d*a
    x = []
    # Compute theoretical mobility.
    location = [[0., 0., 100.]]
    orientation = [Quaternion([1., 0., 0., 0.])]
    mobility_theory = ic.icosohedron_mobility(location, orientation)
    a_eff = 1.0/(6.*np.pi*ic.ETA*mobility_theory[0, 0])
    a_rot_eff = (sphere_mobility_theory[3, 3]*(sph.A**3)/
                 mobility_theory[3, 3])**(1./3.)
    print 'a_effective for d = %f is %f' % (d, a_eff)
    print 'a_effective Rotation for d = %f is %f' % (d, a_rot_eff)
    print 'ratio for d = %f is %f' %(d, a_eff/a_rot_eff)
    mu_parallel = []
    mu_perp = []
    mu_rotation = []

    for h in heights:
      # Calculate 5 random orientations for heights.
      for k in range(5):
        theta = np.random.normal(0., 1., 4)
        theta = Quaternion(theta/np.linalg.norm(theta))
        location = [0., 0., h]
        mobility = ic.icosohedron_mobility([location], [theta])
        x.append(h/a_eff)
        mu_parallel.append(mobility[0, 0]/
                           mobility_theory[0, 0])
        mu_perp.append(mobility[2, 2]/
                       mobility_theory[2, 2])
        mu_rotation.append(mobility[3, 3]/
                           mobility_theory[3, 3])
    pyplot.figure(1)
    pyplot.plot(x, mu_parallel, 'b' + symbols[d], label="d = %s a" % d)
    pyplot.figure(2)
    pyplot.plot(x, mu_perp, 'b' + symbols[d], label="d = %s a" % d)
    pyplot.figure(3)
    pyplot.plot(x, mu_rotation, 'b' + symbols[d], label="d = %s a" % d)


  sphere_parallel = []
  sphere_perp  = []
  sphere_rotation = []
  for h in heights:
    location = [[0., 0., h]]
    sphere_mobility = sph.sphere_mobility(location, orientation)
    sphere_parallel.append(sphere_mobility[0, 0]*6.*np.pi*sph.ETA*sph.A)
    sphere_perp.append(sphere_mobility[2, 2]*6.*np.pi*sph.ETA*sph.A)
    sphere_rotation.append(sphere_mobility[3, 3]/
                           sphere_mobility_theory[3, 3])
  
  pyplot.figure(1)
  pyplot.plot(np.array(heights)/sph.A, sphere_parallel, 'k--', label='Sphere')
  pyplot.title('Parallel Mobility for Icosohedron, a = %s' % a)
  pyplot.legend(loc='best', prop={'size': 9})
  pyplot.xlabel('H/a_effective')
  pyplot.ylabel('Mobility/Bulk Mobility')
  pyplot.savefig('./figures/IcosohedronParallelMobility.pdf')

  pyplot.figure(2)
  pyplot.plot(np.array(heights)/sph.A, sphere_perp, 'k--', label='Sphere')
  pyplot.title('Perpendicular Mobility for Icosohedron, a = %s' % a)
  pyplot.legend(loc='best', prop={'size': 9})
  pyplot.xlabel('H/a_effective')
  pyplot.ylabel('Mobility/Bulk Mobility') 
  pyplot.savefig('./figures/IcosohedronPerpendicularMobility.pdf')

  pyplot.figure(3)
  pyplot.plot(np.array(heights)/sph.A, sphere_rotation, 'k--', label='Sphere')
  pyplot.title('Rotational Mobility for Icosohedron, a = %s' % a)
  pyplot.legend(loc='best', prop={'size': 9})
  pyplot.xlabel('H/a_effective')
  pyplot.ylabel('Mobility/Bulk Mobility')
  pyplot.savefig('./figures/IcosohedronRotationalMobility.pdf')
  

if __name__ == '__main__':
  
  a = 0.2
  heights = np.linspace(0.8, 12.0, 40)
  plot_scatter_icosohedron_mobilities(a, heights)
  


      

