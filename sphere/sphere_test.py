''' Tests for the sphere mobilities etc. '''


import numpy as np
import sys
import unittest

import sphere as sph
from quaternion_integrator.quaternion import Quaternion

class TestSphere(unittest.TestCase):
    
  def setUp(self):
    print(' setUp')
    pass

  def test_sphere_mobility_spd(self):
    '''Test that the sphere mobility is SPD.'''
    location = [np.random.normal(4.0, 1.0, 3)]
    theta = np.random.normal(0., 1., 4)
    theta = Quaternion(theta/np.linalg.norm(theta))
    
    mobility = sph.sphere_mobility(location, theta)
    
    def is_pos_def(x):
      return np.all(np.linalg.eigvals(x) > 0)

    self.assertTrue(is_pos_def(mobility))
    for j in range(6):
      for k in range(j+1, 6):
        self.assertAlmostEqual(mobility[j][k], mobility[k, j])
        
    print(' test_sphere_mobility_spd')


  def test_sphere_mobility_entries_make_sense(self):
    #'''Test that we get the expected sign for the entries of the sphere mobility.'''
    print(' sphere_mobility_entries_make_sense')
    location = [np.random.normal(4.0, 1.0, 3)]
    theta = np.random.normal(0., 1., 4)
    theta = Quaternion(theta/np.linalg.norm(theta))
    mobility = sph.sphere_mobility(location, theta)

    # Check that translation results are positive
    self.assertTrue(mobility[0, 0] > 0.)
    self.assertAlmostEqual(mobility[0, 0], mobility[1, 1])
    self.assertTrue(mobility[2, 2] > 0.)
    
    # Check rotation results are positive
    self.assertTrue(mobility[3, 3] > 0.)
    self.assertAlmostEqual(mobility[4, 4], mobility[3, 3])
    self.assertTrue(mobility[5, 5] > 0.)
    

if __name__ == '__main__':
  unittest.main()        

    








