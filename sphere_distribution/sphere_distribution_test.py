import unittest
import numpy as np
import random
import sphere_distribution

class TestSphereDistribution(unittest.TestCase):

  def setUp(self):
    pass

  def test_generate_random_matrix(self):
    ''' Test that the matrix generated is a rotation. '''

    A = sphere_distribution.generate_random_rotation_matrix()

    # Check for unit norm.
    self.assertAlmostEqual(np.linalg.norm(A[0]), 1.0)
    self.assertAlmostEqual(np.linalg.norm(A[1]), 1.0)
    self.assertAlmostEqual(np.linalg.norm(A[2]), 1.0)

    # Check for orthogonality.
    self.assertAlmostEqual(np.inner(A[0], A[1]), 0.)
    self.assertAlmostEqual(np.inner(A[1], A[2]), 0.)
    self.assertAlmostEqual(np.inner(A[0], A[2]), 0.)

  def test_generate_random_matrix_manual(self):
    ''' Test that the matrix generated is a rotation. '''

    A = sphere_distribution.generate_random_rotation_matrix_manual()

    # Check for unit norm.
    self.assertAlmostEqual(np.linalg.norm(A[0]), 1.0)
    self.assertAlmostEqual(np.linalg.norm(A[1]), 1.0)
    self.assertAlmostEqual(np.linalg.norm(A[2]), 1.0)

    # Check for orthogonality.
    self.assertAlmostEqual(np.inner(A[0], A[1]), 0.)
    self.assertAlmostEqual(np.inner(A[1], A[2]), 0.)
    self.assertAlmostEqual(np.inner(A[0], A[2]), 0.)


  def test_matrix_to_quaternion(self):
    ''' Test that we get the right quaternion back from a rotation matrix '''
    # First construct any random unit quaternion.
    s = 2*random.random() - 1.
    p1 = (2. - 2*np.abs(s))*random.random() - (1. - np.abs(s))
    p2 = ((2. - 2.*np.abs(s) - 2.*np.abs(p1))*random.random() - 
          (1. - np.abs(s) - np.abs(p1)))
    p3 = np.sqrt(1. - s**2 - p1**2 - p2**2)
    
    # Create the rotation matrix associated with (s, p).
    R = 2.0*np.matrix([[p1**2 + s**2 - 0.5, p2*p1 + s*p3, p3*p1 - s*p2],
                       [p1*p2 - s*p3, p2**2 + s**2 - 0.5, p3*p2 + s*p1],
                       [p1*p3 + s*p2, p2*p3 - s*p1, p3**2 + s**2 - 0.5]])
    # Get the Quaternion from the matrix.
    quaternion = sphere_distribution.MatrixToQuaternion(R)
    # The + and - of a quaternion indicate the same rotation. We choose the sign 
    # with 50% probability each.
    if (quaternion[0]/s) > 0:
      self.assertAlmostEqual(quaternion[0], s)
      self.assertAlmostEqual(quaternion[1], p1)
      self.assertAlmostEqual(quaternion[2], p2)
      self.assertAlmostEqual(quaternion[3], p3)
    else:
      self.assertAlmostEqual(quaternion[0], -1.*s)
      self.assertAlmostEqual(quaternion[1], -1.*p1)
      self.assertAlmostEqual(quaternion[2], -1.*p2)
      self.assertAlmostEqual(quaternion[3], -1.*p3)


if __name__ == "__main__":
  unittest.main()
