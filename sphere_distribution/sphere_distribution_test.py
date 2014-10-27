import unittest
import numpy as np
import random
import sphere_distribution

class TestSphereDistribution(unittest.TestCase):

  def setUp(self):
    pass

  def test_matrix_to_quaterion(self):
    ''' Test that we get the right quaternion back from a rotation matrix '''
    s = random.random()
    p1 = random.random()*(1. - s)
    p2 = random.random()*(1. - s - p1)
    p3 = np.sqrt(1. - s**2 - p1**2 - p2**2)
    
    R = np.matrix([[p1**2 + s**2 - 0.5, p2*p1 + s*p3, p3*p1 - s*p2],
                   [p1*p2 - s*p3, p2**2 + s**2 - 0.5, p3*p2 + s*p1],
                   [p1*p3 + s*p2, p2*p3 - s*p1, p3**2 + s**2 - 0.5]])

    quaternion = sphere_distribution.MatrixToQuaterion(R)
    self.assertAlmostEqual(quaternion[0], s)
    self.assertAlmostEqual(quaternion[1], p1)
    self.assertAlmostEqual(quaternion[2], p2)
    self.assertAlmostEqual(quaternion[3], p3)


if __name__ == "__main__":
  unittest.main()
