import unittest
import numpy as np
import uniform_analyzer as ua

class TestUnifSphereDist(unittest.TestCase):
  ''' Test the Uniform Sphere Distribution analyzer functions '''

  def setUp(self):
    pass

  def test_GenerateXiEta(self):
    ''' Test that the xi and eta generated are orthogonal and unit norm '''
    samples = [[0., 1., 0., 0.]]
    
    unif_sphere_dist_analyzer = ua.UniformAnalyzer(samples)
    
    xi, eta = unif_sphere_dist_analyzer.GenerateXiEta()
    
    self.assertAlmostEqual(np.linalg.norm(xi), 1.0)
    self.assertAlmostEqual(np.linalg.norm(eta), 1.0)
    self.assertAlmostEqual(np.inner(eta, xi), 0.0)

if __name__ == "__main__":
  unittest.main()
