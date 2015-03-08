''' Unit tests for icosohedron '''

import unittest
import numpy as np

import icosohedron as ic
import icosohedron_nonuniform as icn
from quaternion_integrator.quaternion import Quaternion

class TestIcosohedron(unittest.TestCase):
    
  def setUp(self):
    pass

  def test_get_icosohedron_r_vectors(self):
    ''' Test that we get the correct r vectors for a simple orientation.'''

    # pi/2 rotation around the x axis.
    theta = Quaternion([np.cos(np.pi/4.), np.sin(np.pi/4.), 0., 0.])
    location = [0., 0., 10.]
    r_vectors = ic.get_icosohedron_r_vectors(location, theta)
    
    self.assertAlmostEqual(r_vectors[1][0], 0.0)
    self.assertAlmostEqual(r_vectors[1][1], -1.*ic.A)
    self.assertAlmostEqual(r_vectors[1][2], 10.0)
    self.assertAlmostEqual(r_vectors[11][0], 0.0)
    self.assertAlmostEqual(r_vectors[11][1], ic.A)
    self.assertAlmostEqual(r_vectors[11][2], 10.0)

  def test_icosohedron_mobility_spd(self):
    ''' 
    Test that the icosohedron mobility is Symmetric and Positive Definite
    for a random allowable state.
    '''
    theta = np.random.normal(0., 1., 4)
    theta = Quaternion(theta/np.linalg.norm(theta))
    location = [np.random.uniform(1., 3.) for _ in range(3)]
    
    mobility = ic.icosohedron_mobility([location], [theta])
    
    def is_pos_def(x):
      return np.all(np.linalg.eigvals(x) > 0)    

    self.assertTrue(is_pos_def(mobility))
    for j in range(6):
      for k in range(j+1, 6):
        self.assertAlmostEqual(mobility[j][k], mobility[k][j])

  def test_icosohedron_rotation(self):
    '''Test that rotations of the icosohedron make sense.'''
    theta = np.random.normal(0., 1., 4)
    theta = Quaternion(theta/np.linalg.norm(theta))
    location = [np.random.uniform(1., 3.) for _ in range(3)]
    mobility = ic.icosohedron_mobility([location], [theta])
    # Rotation in the positive x direction should produce positive y velocity
    self.assertTrue(mobility[1, 3] > 0.0)
    # Rotation in the positive y direction should produce negative x velocity
    self.assertTrue(mobility[0, 4] < 0.0)
    # Push in the positive y direction should produce positive x rotation.
    # (This should already be true by symmetry, but test anyway)
    self.assertTrue(mobility[3, 1] > 0.0)
    # Push in the positive x direction should produce negative y rotation.
    # (This should already be true by symmetry, but test anyway)
    self.assertTrue(mobility[4, 0] < 0.0)
    

  def test_nonuniform_torque(self):
    ''' Test that the nonuniform torque makes sense for the heavy particle.
    on the side.'''
    
    theta = Quaternion([1./(np.sqrt(2.)), 1./np.sqrt(2.), 0., 0.])
    location = [0., 0., 3.]

    torque = icn.nonuniform_torque_calculator([location], [theta])
    
    self.assertTrue(torque[0] < 0.0)
    self.assertAlmostEqual(torque[1], 0.0)
    self.assertAlmostEqual(torque[2], 0.0)
    
    
    
if __name__ == '__main__':
  unittest.main()    
