import unittest
from constrained_integrator import ConstrainedIntegrator

class TestConstrainedIntegrator(unittest.TestCase):

  def setUp(self):
    self.mobility = [[1.0, 0.0], [0.0, 1.0]]
    
  def empty_constraint(self, x):
    return 0.0

  def test_initialize(self):
    """ Test that the integrator is set up correctly. """
    scheme = "RFD"
    
    test_integrator = ConstrainedIntegrator(
      self.empty_constraint, self.mobility, scheme)
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
    self.assertRaises(
      NotImplementedError,
      ConstrainedIntegrator,
      self.empty_constraint, self.mobility, scheme)

if __name__ == "__main__":
  unittest.main()
    
