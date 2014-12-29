import sys
import unittest
import numpy as np
import uniform_analyzer as ua
from quaternion_integrator.quaternion import Quaternion

class TestUnifSphereDist(unittest.TestCase):
  ''' Test the Uniform Sphere Distribution analyzer functions '''

  def setUp(self):
    pass


  def test_generate_from_quaternions(self):
    samples = [[Quaternion([1., 0., 0., 0.])],
               [Quaternion([0., 1., 0., 0.])]]

    test_analyzer = ua.UniformAnalyzer(samples, 'test')
    
    self.assertEqual(test_analyzer.dim, 4)
    # First sample
    self.assertAlmostEqual(test_analyzer.samples[0][0], 1.)
    self.assertAlmostEqual(test_analyzer.samples[0][1], 0.)
    self.assertAlmostEqual(test_analyzer.samples[0][2], 0.)
    self.assertAlmostEqual(test_analyzer.samples[0][3], 0.)

    # Second Sample
    self.assertAlmostEqual(test_analyzer.samples[1][0], 0.)
    self.assertAlmostEqual(test_analyzer.samples[1][1], 1.)
    self.assertAlmostEqual(test_analyzer.samples[1][2], 0.)
    self.assertAlmostEqual(test_analyzer.samples[1][3], 0.)
    

  def test_generate_xi_eta(self):
    ''' Test that the xi and eta generated are orthogonal and unit norm '''
    samples = [[0., 1., 0., 0.]]
    
    unif_sphere_dist_analyzer = ua.UniformAnalyzer(samples, 'test')
    
    xi, eta = unif_sphere_dist_analyzer.generate_xi_eta()
    
    self.assertAlmostEqual(np.linalg.norm(xi), 1.0)
    self.assertAlmostEqual(np.linalg.norm(eta), 1.0)
    self.assertAlmostEqual(np.inner(eta, xi), 0.0)
    
if __name__ == "__main__":
  unittest.main()
