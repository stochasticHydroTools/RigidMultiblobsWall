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
    def identity_mobility(position):
      return np.array([[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]])

    def identity_torque_calculator(position):
      return np.array([1., 1., 1.])

    initial_position = [Quaternion([1., 0., 0., 0.])]
    quaternion_integrator = QuaternionIntegrator(identity_mobility,
                                                 initial_position,
                                                 identity_torque_calculator)
    self.assertEqual(quaternion_integrator.dim, 1)


  def test_deterministic_fixman(self):
    ''' test the deterministic part of the Fixman step '''
    def test_mobility(position):
      return np.array([[2.0 + position[0].s, 0., 0.], 
                       [0., 1., 0.], 
                       [0., 0., 2.]])

    def identity_torque_calculator(position):
      return np.array([1., 1., 1.])

    initial_position = [Quaternion([1., 0., 0., 0.])]
    quaternion_integrator = QuaternionIntegrator(test_mobility,
                                                 initial_position,
                                                 identity_torque_calculator)
    quaternion_integrator.kT = 0.0
    quaternion_integrator.fixman_time_step(0.1)

    # TODO: Fill this out with the correct endpoint for the deterministic part.
    

if __name__ == "__main__":
  unittest.main()      
