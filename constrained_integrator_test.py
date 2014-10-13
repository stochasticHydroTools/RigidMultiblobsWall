import unittest
import numpy as np
from constrained_integrator import ConstrainedIntegrator

class TestConstrainedIntegrator(unittest.TestCase):

  def setUp(self):
    self.mobility = [[1.0, 0.0], [0.0, 1.0]]
    
  def empty_constraint(self, x):
    return 0.0

  def test_initialize(self):
    """ Test that the integrator is set up correctly. """
    scheme = "RFD"
    initial_position = [0.0, 0.0]
    
    test_integrator = ConstrainedIntegrator(
      self.empty_constraint, self.mobility, scheme, initial_position)
    # Test dimensions
    self.assertEqual(test_integrator.dim, 2)
    # Test Mobility
    self.assertEqual(test_integrator.mobility[0][0], 1.0)
    self.assertEqual(test_integrator.mobility[1][0], 0.0)
    self.assertEqual(test_integrator.mobility[0][1], 0.0)
    self.assertEqual(test_integrator.mobility[1][1], 1.0)
    # Test constraint.
    self.assertEqual(test_integrator.surface_function(10.0), 0.0)
    # Test scheme.
    self.assertEqual(test_integrator.scheme, "RFD")

  def test_scheme_check(self):
    """ Test that a nonexistant scheme isn't accepted. """
    scheme = "DOESNTEXIST"
    initial_position = [0.0, 0.0]
    self.assertRaises(
      NotImplementedError,
      ConstrainedIntegrator,
      self.empty_constraint, self.mobility, scheme, initial_position)
    
  def test_normal_vector(self):
    """ Test that normal vector points in the right direction """
    scheme = "RFD"
    initial_position = [1.2, 0.0]
    def sphere_constraint(x):
      return np.sqrt(x[0]*x[0] + x[1]*x[1]) - 1.2
    
    test_integrator = ConstrainedIntegrator(
      sphere_constraint, self.mobility, scheme, initial_position)
    
    normal_vector = test_integrator.NormalVector()
    self.assertEqual(normal_vector[0], 1.0)
    self.assertEqual(normal_vector[1], 0.0)


if __name__ == "__main__":
  unittest.main()
    
