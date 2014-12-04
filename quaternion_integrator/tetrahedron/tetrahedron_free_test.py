''' Unit tests for tetrahedron_free '''

import sys
sys.path.append('../')

import unittest
import numpy as np
import tetrahedron_free as tf
from quaternion import Quaternion
from quaternion_integrator import QuaternionIntegrator
import tetrahedron as tdn

class TestFreeTetrahedron(unittest.TestCase):
  
  def setUp(self):
    pass

  def test_get_free_r_vectors(self):
    ''' Test that we get the correct r_vectors with a free tetrahedron.'''
    theta = Quaternion([1., 0., 0., 0.])
    location = [20., 20., 20.]
    r_vectors = tf.get_free_r_vectors(location, theta)
    
    # Check r1.
    self.assertAlmostEqual(r_vectors[0][0], 20.)
    self.assertAlmostEqual(r_vectors[0][1], 20. + 2./np.sqrt(3.))
    self.assertAlmostEqual(r_vectors[0][2], 20. - 2.*np.sqrt(2.)/np.sqrt(3.))

    


if __name__ == '__main__':
  unittest.main()

    
    
    

