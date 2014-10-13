import unittest
import constrained-integrator

Class TestConstrainedIntegrator(unittests.TestCase):

  def SetUp(self):
    pass
  
  def test_initialize(self):
    self.AssertEqual(1.0, 1.0)

if __name__ == "__main__":
  unittest.main()
    
