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
  Here we vary the ratio between Icosohedron length and vertex radius.
  Compare these results to a sphere with the corresponding effective radius.
  '''
  symbols = {1.63: '^', 2.0: '.', 2.5: 's', 3.2: 'v', 4.0: 'x', 5.0: 'o'}
  orientation = [Quaternion([1., 0., 0., 0.])]
  far_location = [[0., 0., 30000.*a]]
  sphere_mobility_theory = sph.sphere_mobility(far_location, orientation)
  for d in [1.63, 2.0, 2.5, 3.2, 4.0, 5.0]:
    ic.VERTEX_A = a
    ic.A = d*a
    x = []
    # Compute theoretical mobility.
    location = [[0., 0., 30000.*a]]
    orientation = [Quaternion([1., 0., 0., 0.])]
    mobility_theory = ic.icosohedron_mobility(location, orientation)
    a_eff = 1.0/(6.*np.pi*ic.ETA*mobility_theory[0, 0])
    a_rot_eff = (1./(mobility_theory[3, 3]*8.*np.pi*ic.ETA))**(1./3.)
    print 'a_effective for d = %f is %f' % (d, a_eff)
    print 'a_effective Rotation for d = %f is %f' % (d, a_rot_eff)
    print 'ratio for d = %f is %f' %(d, a_eff/a_rot_eff)
    mu_parallel = []
    mu_perp = []
    mu_rotation = []

    for r in heights:
      # Calculate 2 random orientations for heights.
      h = r*a_eff
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
    pyplot.plot(np.array(x)*a_eff/a_rot_eff,
                mu_rotation, 'b' + symbols[d], label="d = %s a" % d)


  sphere_parallel = []
  sphere_perp  = []
  sphere_rotation = []
  for h in np.array(heights)*sph.A:
    location = [[0., 0., h]]
    sphere_mobility = sph.sphere_mobility(location, orientation)
    sphere_parallel.append(sphere_mobility[0, 0]*6.*np.pi*sph.ETA*sph.A)
    sphere_perp.append(sphere_mobility[2, 2]*6.*np.pi*sph.ETA*sph.A)
    sphere_rotation.append(sphere_mobility[3, 3]/
                           sphere_mobility_theory[3, 3])
  
  pyplot.figure(1)
  pyplot.plot(np.array(heights), sphere_parallel, 'k--', label='Sphere')
  pyplot.title('Parallel Mobility for Icosohedron, a = %s' % a)
  pyplot.legend(loc='best', prop={'size': 9})
  pyplot.xlabel('H/a_effective')
  pyplot.ylabel('Mobility/Bulk Mobility')
  pyplot.savefig('./figures/IcosohedronParallelMobility.pdf')
  pyplot.clf()

  pyplot.figure(2)
  pyplot.plot(np.array(heights), sphere_perp, 'k--', label='Sphere')
  pyplot.title('Perpendicular Mobility for Icosohedron, a = %s' % a)
  pyplot.legend(loc='best', prop={'size': 9})
  pyplot.xlabel('H/a_effective')
  pyplot.ylabel('Mobility/Bulk Mobility') 
  pyplot.savefig('./figures/IcosohedronPerpendicularMobility.pdf')
  pyplot.clf()

  pyplot.figure(3)
  pyplot.plot(np.array(heights), sphere_rotation, 'k--', label='Sphere')
  pyplot.title('Rotational Mobility for Icosohedron, a = %s' % a)
  pyplot.legend(loc='best', prop={'size': 9})
  pyplot.xlabel('H/a_effective')
  pyplot.ylabel('Mobility/Bulk Mobility')
  pyplot.savefig('./figures/IcosohedronRotationalMobility.pdf')
  pyplot.clf()
  


def plot_icosohedron_mobilities_at_wall(h_over_a, a, r):
  ''' 
  Plot the icosohedron mobilities at the wall.
  r is the ratio of icosohedron vertex radius to distance
  from vertices to the center.  a is the icosohedron vertex radius.
  h_over_a is where we will measure the mobilities in nondimensionalized
  units (h = a_effective*h_over_a)
  '''
  # Put icosohedron in contact with wall.
  for r in r_list:
    ic.VERTEX_A = a
    ic.A = r*a
    orientation = [Quaternion([1., 0., 0., 0.])]
    far_location = [[0., 0., 30000.*a]]
    # Compute theoretical mobility far from wall.
    mobility_theory = ic.icosohedron_mobility(far_location, orientation)
    a_eff = 1.0/(6.*np.pi*ic.ETA*mobility_theory[0, 0])
    print 'a_effective is %f' % a_eff
    sph.A = a_eff
    h = h_over_a*a_eff
    sphere_mobility_near_wall = sph.sphere_mobility([[0., 0., h]],
                                                  orientation)
    mobility_t_xx = []
    mobility_t_zz = []
    mobility_r_xx = []
    mobility_r_zz = []
    mobility_tr = []
    components = []
    for k in range(100):
      # Generate 100 random orientations of the icosohedron near the wall.
      theta = np.random.normal(0., 1., 4)
      theta = [Quaternion(theta/np.linalg.norm(theta))]
      mobility = ic.icosohedron_mobility([[0., 0., h]], theta)
      mobility_t_xx.append((mobility[0, 0]/
                            sphere_mobility_near_wall[0, 0]))
      mobility_t_zz.append((mobility[2, 2]/
                            sphere_mobility_near_wall[2, 2]))
      mobility_r_xx.append((mobility[3, 3]/
                            sphere_mobility_near_wall[3, 3]))
      mobility_r_zz.append((mobility[5, 5]/
                            sphere_mobility_near_wall[5, 5]))
      mobility_tr.append((mobility[0, 4]/
                          sphere_mobility_near_wall[0, 4]))
      
    pyplot.figure(1)
    pyplot.plot(r*np.ones(100), mobility_t_xx, 'g.')
    pyplot.figure(2)
    pyplot.plot(r*np.ones(100), mobility_t_zz, 'g.')
    pyplot.figure(3)
    pyplot.plot(r*np.ones(100), mobility_r_xx, 'g.')
    pyplot.figure(4)
    pyplot.plot(r*np.ones(100), mobility_r_zz, 'g.')
    pyplot.figure(5)
    pyplot.plot(r*np.ones(100), mobility_tr, 'g.')

  pyplot.figure(1)
  pyplot.title('Mu Parallel')
  pyplot.ylabel('Mobility Component / Sphere')
  pyplot.xlabel('d/a')
  pyplot.xlim([0.0, 7.0])
  pyplot.savefig('./figures/IcosohedronTranslation-xx-NearWall.pdf')
  pyplot.clf()
  pyplot.figure(2)
  pyplot.title('Mu Perpendicular')
  pyplot.ylabel('Mobility Component / Sphere')
  pyplot.xlabel('d/a')
  pyplot.xlim([0.0, 7.0])
  pyplot.savefig('./figures/IcosohedronTranslation-zz-NearWall.pdf')
  pyplot.clf()
  pyplot.figure(3)
  pyplot.title('Mu Rotational X')
  pyplot.ylabel('Mobility Component / Sphere')
  pyplot.xlabel('d/a')
  pyplot.xlim([0.0, 7.0])
  pyplot.savefig('./figures/IcosohedronRotation-xx-NearWall.pdf')
  pyplot.clf()
  pyplot.figure(4)
  pyplot.title('Mu Rotation Z')
  pyplot.ylabel('Mobility Component / Sphere')
  pyplot.xlabel('d/a')
  pyplot.xlim([0.0, 7.0])
  pyplot.savefig('./figures/IcosohedronRotation-zz-NearWall.pdf')
  pyplot.clf()
  pyplot.figure(5)
  pyplot.title('Mu Translation-Rotation X-Y')
  pyplot.ylabel('Mobility Component / Sphere')
  pyplot.xlabel('d/a')
  pyplot.xlim([0.0, 7.0])
  pyplot.savefig('./figures/IcosohedronTR-xy-NearWall.pdf')


if __name__ == '__main__':
  
  a = 0.3
  heights = np.linspace(1.4, 3.0, 50)
  plot_scatter_icosohedron_mobilities(a, heights)
  h_over_a = 300.0
  r_list = [1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5,  5.0, 6.0]
  plot_icosohedron_mobilities_at_wall(h_over_a, a, r_list)

