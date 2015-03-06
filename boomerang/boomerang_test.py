''' Unit tests for boomerang. '''

import unittest
import numpy as np

import boomerang as bmr
from quaternion_integrator.quaternion import Quaternion

class TestBoomerang(unittest.TestCase):
  
  def setUp(self):
    pass

  def test_get_boomerang_r_vectors(self):
    ''' Test that we get the correct R vectors for some simple orientations.'''
    # pi/2 rotation around the x axis.
    theta = Quaternion([np.cos(np.pi/4.), np.sin(np.pi/4.), 0., 0.])
    location = [0., 0., 5.]
    r_vectors = bmr.get_boomerang_r_vectors(location, theta)

    self.assertAlmostEqual(r_vectors[0][0], 1.575)
    self.assertAlmostEqual(r_vectors[0][1], 0.0)
    self.assertAlmostEqual(r_vectors[0][2], 5.0)
    self.assertAlmostEqual(r_vectors[3][0], 0.0)
    self.assertAlmostEqual(r_vectors[3][1], 0.0)
    self.assertAlmostEqual(r_vectors[3][2], 5.525)
    self.assertAlmostEqual(r_vectors[5][0], 0.0)
    self.assertAlmostEqual(r_vectors[5][1], 0.0)
    self.assertAlmostEqual(r_vectors[5][2], 6.575)


  def test_boomerang_mobility_spd(self):
    ''' Test that the mobility is SPD for random configuration. '''
    theta = np.random.normal(0., 1., 4)
    theta = Quaternion(theta/np.linalg.norm(theta))
    location = [np.random.uniform(2.5, 4.) for _ in range(3)]
    
    mobility = bmr.boomerang_mobility([location], [theta])
    
    def is_pos_def(x):
      return np.all(np.linalg.eigvals(x) > 0)    

    self.assertTrue(is_pos_def(mobility))
    for j in range(6):
      for k in range(j+1, 6):
        self.assertAlmostEqual(mobility[j][k], mobility[k][j])

    
if __name__ == '__main__':
  unittest.main()    
