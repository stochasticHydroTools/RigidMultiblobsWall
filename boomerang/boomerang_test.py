''' Unit tests for boomerang. '''

import unittest
import numpy as np
import sys
sys.path.append('..')


import boomerang as bm
from quaternion_integrator.quaternion import Quaternion

class TestBoomerang(unittest.TestCase):
  
  def setUp(self):
    pass

  def test_get_boomerang_r_vectors(self):
    ''' Test that we get the correct R vectors for some simple orientations.'''
    # pi/2 rotation around the x axis.
    theta = Quaternion([np.cos(np.pi/4.), np.sin(np.pi/4.), 0., 0.])
    location = [0., 0., 5.]
    r_vectors = bm.get_boomerang_r_vectors(location, theta)

    self.assertAlmostEqual(r_vectors[0][0], 1.575)
    self.assertAlmostEqual(r_vectors[0][1], 0.0)
    self.assertAlmostEqual(r_vectors[0][2], 5.0)
    self.assertAlmostEqual(r_vectors[3][0], 0.0)
    self.assertAlmostEqual(r_vectors[3][1], 0.0)
    self.assertAlmostEqual(r_vectors[3][2], 5.525)
    self.assertAlmostEqual(r_vectors[5][0], 0.0)
    self.assertAlmostEqual(r_vectors[5][1], 0.0)
    self.assertAlmostEqual(r_vectors[5][2], 6.575)

    
if __name__ == '__main__':
  unittest.main()    
