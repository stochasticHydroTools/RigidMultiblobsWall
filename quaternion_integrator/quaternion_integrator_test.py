import unittest
import numpy as np
import random
from quaternion import Quaternion
from quaternion_integrator import QuaternionIntegrator


class TestQuaternionIntegrator(unittest.TestCase):
  
  def setUp(self):
    pass
  

  def test_initialize_integrator(self):
    ''' Simple test to make sure we can initialize the integrator. '''
    def identity_mobility(orientation):
      return np.array([[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]])

    def identity_torque_calculator(orientation):
      return np.array([1., 1., 1.])

    initial_orientation = [Quaternion([1., 0., 0., 0.])]
    quaternion_integrator = QuaternionIntegrator(identity_mobility,
                                                 initial_orientation,
                                                 identity_torque_calculator)
    self.assertEqual(quaternion_integrator.dim, 1)


  def test_deterministic_fixman(self):
    ''' test the deterministic part of the Fixman step '''
    def test_mobility(orientation):
      return np.array([[2.0 + orientation[0].s, 0., 0.], 
                       [0., 1., 0.], 
                       [0., 0., 2.]])

    def identity_torque_calculator(orientation):
      return np.array([1., 1., 1.])

    initial_orientation = [Quaternion([1., 0., 0., 0.])]
    quaternion_integrator = QuaternionIntegrator(test_mobility,
                                                 initial_orientation,
                                                 identity_torque_calculator)
    quaternion_integrator.kT = 0.0
    quaternion_integrator.fixman_time_step(0.1)

    # TODO: Fill this out with the correct endpoint for the deterministic part.
    
  
  def test_deterministic_fixman_with_location(self):
    ''' Test the deterministic part of the Fixman step with location.'''
    def identity_mobility(location, orientation):
      return np.identity(6)
      
    def identity_torque_calculator(location, orientation):
      return np.array([1., 1., 1.])
      
    def identity_force_calculator(location, orientation):
      return np.array([1., 1., 1.])

    initial_orientation = [Quaternion([1., 0., 0., 0.])]
    initial_location = [[1., 1., 1.]]
    quaternion_integrator = QuaternionIntegrator(identity_mobility,
                                                 initial_orientation,
                                                 identity_torque_calculator,
                                                 has_location = True,
                                                 initial_location = initial_location,
                                                 force_calculator = identity_force_calculator)
    quaternion_integrator.kT = 0.0
    quaternion_integrator.fixman_time_step(1.0)
    # Check location
    self.assertAlmostEqual(quaternion_integrator.location[0][0], 2.)
    self.assertAlmostEqual(quaternion_integrator.location[0][1], 2.)
    self.assertAlmostEqual(quaternion_integrator.location[0][2], 2.)
    # Check orientation
    self.assertAlmostEqual(quaternion_integrator.orientation[0].s, 1.)
    
    
if __name__ == "__main__":
  unittest.main()      
