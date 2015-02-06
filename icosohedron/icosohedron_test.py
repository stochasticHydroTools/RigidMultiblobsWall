''' Unit tests for icosohedron '''

import sys
sys.path.append('..')

import unittest
import numpy as np

import icosohedron as ic
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
    
    
    
    
if __name__ == '__main__':
  unittest.main()    
