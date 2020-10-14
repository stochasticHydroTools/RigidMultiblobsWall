''' Unit tests for boomerang. '''

import unittest
import numpy as np

from . import boomerang as bmr
from quaternion_integrator.quaternion import Quaternion
from general_application_utils import transfer_mobility

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
    self.assertAlmostEqual(r_vectors[4][0], 0.0)
    self.assertAlmostEqual(r_vectors[4][1], 0.0)
    self.assertAlmostEqual(r_vectors[4][2], 5.525)
    self.assertAlmostEqual(r_vectors[6][0], 0.0)
    self.assertAlmostEqual(r_vectors[6][1], 0.0)
    self.assertAlmostEqual(r_vectors[6][2], 6.575)


    # pi/2 rotation around the y axis.
    theta = Quaternion([np.cos(np.pi/4.), 0., np.sin(np.pi/4.), 0.])
    r_vectors = bmr.get_boomerang_r_vectors(location, theta)

    self.assertAlmostEqual(r_vectors[0][0], 0.0)
    self.assertAlmostEqual(r_vectors[0][1], 0.0)
    self.assertAlmostEqual(r_vectors[0][2], 5.0 - 1.575)
    self.assertAlmostEqual(r_vectors[4][0], 0.0)
    self.assertAlmostEqual(r_vectors[4][1], 0.525)
    self.assertAlmostEqual(r_vectors[4][2], 5.0)
    self.assertAlmostEqual(r_vectors[6][0], 0.0)
    self.assertAlmostEqual(r_vectors[6][1], 1.575)
    self.assertAlmostEqual(r_vectors[6][2], 5.0)

    # pi/2 rotation around the z axis.
    theta = Quaternion([np.cos(np.pi/4.), 0., 0., np.sin(np.pi/4.)])
    r_vectors = bmr.get_boomerang_r_vectors(location, theta)

    self.assertAlmostEqual(r_vectors[0][0], 0.0)
    self.assertAlmostEqual(r_vectors[0][1], 1.575)
    self.assertAlmostEqual(r_vectors[0][2], 5.0)
    self.assertAlmostEqual(r_vectors[4][0], -0.525)
    self.assertAlmostEqual(r_vectors[4][1], 0.0)
    self.assertAlmostEqual(r_vectors[4][2], 5.0)
    self.assertAlmostEqual(r_vectors[6][0], -1.575)
    self.assertAlmostEqual(r_vectors[6][1], 0.0)
    self.assertAlmostEqual(r_vectors[6][2], 5.0)


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


  def test_change_mobility_origin(self):
    ''' This tests the function in utils.py that transforms a mobility
    about one point to a mobility about another.
    '''
    # Random location and orientation
    location = [[0., 0., np.random.uniform(4., 7.)]]
    orientation = np.random.normal(0., 1., 4)
    orientation = [Quaternion(orientation/np.linalg.norm(orientation))]

    mobility = bmr.boomerang_mobility(location, orientation)

    # Choose a random other point, evaluate mobility.
    point = location[0] + np.random.normal(0., 1., 3)
    mobility_2 = bmr.boomerang_mobility_at_arbitrary_point(location, orientation,
                                                          point)
    # Transfer mobility to point using util function.
    transferred_mobility_2 = transfer_mobility(mobility, location[0], point)
    
    # Compare results.
    for j in range(0, 6):
      for k in range(0, 6):
        self.assertAlmostEqual(mobility_2[j, k], transferred_mobility_2[j, k])

    
if __name__ == '__main__':
  unittest.main()    
