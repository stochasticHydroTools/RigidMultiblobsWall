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
    ''' 
    Test the deterministic part of the Fixman step with location.
    This does not test the drift or stochastic pieces, kT = 0.
    '''
    def identity_mobility(location, orientation):
      return np.identity(6)
      
    def e1_torque_calculator(location, orientation):
      return np.array([1., 0., 0.])
      
    def identity_force_calculator(location, orientation):
      return np.array([1., 1., 1.])

    initial_orientation = [Quaternion([1., 0., 0., 0.])]
    initial_location = [[1., 1., 1.]]
    quaternion_integrator = QuaternionIntegrator(identity_mobility,
                                                 initial_orientation,
                                                 e1_torque_calculator,
                                                 has_location = True,
                                                 initial_location = initial_location,
                                                 force_calculator = identity_force_calculator)
    quaternion_integrator.kT = 0.0
    quaternion_integrator.fixman_time_step(1.0)

    quaternion_dt = Quaternion.from_rotation([1., 0., 0.])
    new_orientation = quaternion_dt*initial_orientation[0]
    # Check location
    self.assertAlmostEqual(quaternion_integrator.location[0][0], 2.)
    self.assertAlmostEqual(quaternion_integrator.location[0][1], 2.)
    self.assertAlmostEqual(quaternion_integrator.location[0][2], 2.)
    # Check orientation
    self.assertAlmostEqual(quaternion_integrator.orientation[0].s, 
                           new_orientation.s)
    self.assertAlmostEqual(quaternion_integrator.orientation[0].p[0], 
                           new_orientation.p[0])
    self.assertAlmostEqual(quaternion_integrator.orientation[0].p[1], 
                           new_orientation.p[1])
    self.assertAlmostEqual(quaternion_integrator.orientation[0].p[2], 
                           new_orientation.p[2])


    def test_deterministic_rfd_with_location(self):
      ''' 
      Test the deterministic part of the RFD timestepping with location.
      This does not test the drift. 
      '''
    def identity_mobility(location, orientation):
      return np.identity(6)
      
    def e1_torque_calculator(location, orientation):
      return np.array([1., 0., 0.])
      
    def identity_force_calculator(location, orientation):
      return np.array([1., 1., 1.])

    initial_orientation = [Quaternion([1., 0., 0., 0.])]
    initial_location = [[1., 1., 1.]]
    quaternion_integrator = QuaternionIntegrator(identity_mobility,
                                                 initial_orientation,
                                                 e1_torque_calculator,
                                                 has_location = True,
                                                 initial_location = initial_location,
                                                 force_calculator = identity_force_calculator)
    quaternion_integrator.kT = 0.0
    quaternion_integrator.rfd_time_step(1.0)

    quaternion_dt = Quaternion.from_rotation([1., 0., 0.])
    new_orientation = quaternion_dt*initial_orientation[0]
    # Check location
    self.assertAlmostEqual(quaternion_integrator.location[0][0], 2.)
    self.assertAlmostEqual(quaternion_integrator.location[0][1], 2.)
    self.assertAlmostEqual(quaternion_integrator.location[0][2], 2.)
    # Check orientation
    self.assertAlmostEqual(quaternion_integrator.orientation[0].s, 
                           new_orientation.s)
    self.assertAlmostEqual(quaternion_integrator.orientation[0].p[0], 
                           new_orientation.p[0])
    self.assertAlmostEqual(quaternion_integrator.orientation[0].p[1], 
                           new_orientation.p[1])
    self.assertAlmostEqual(quaternion_integrator.orientation[0].p[2], 
                           new_orientation.p[2])


  def test_fixman_drift_and_cov(self):
    ''' Test that the drift and covariance from the fixman scheme is correct. '''
    TOL = 5e-2
    initial_orientation = [Quaternion([1., 0., 0., 0.])]

    def test_mobility(orientation):
      return np.array([
        [1., orientation[0].s*orientation[0].p[0], 0.],
        [orientation[0].s*orientation[0].p[0], 1., 0.],
        [0., 0., 1.],])

    def zero_torque_calculator(orientation):
      return np.zeros(3)
    

    test_integrator = QuaternionIntegrator(test_mobility, initial_orientation,
                                           zero_torque_calculator)

    [avg_drift, avg_cov] = test_integrator.estimate_drift_and_covariance(0.01, 80000, 'FIXMAN')
    self.assertLess(abs(avg_drift[0]), TOL)
    self.assertLess(abs(avg_drift[1] - 0.5), TOL)
    self.assertLess(abs(avg_drift[2]), TOL)

    true_covariance = test_mobility(initial_orientation)
    
    for j in range(3):
      for k in range(3):
        self.assertLess(abs(avg_cov[j, k] - true_covariance[j, k]), TOL)



  def test_rfd_drift_and_cov(self):
    ''' Test that the drift and covariance from the RFD scheme is correct. '''
    TOL = 5e-2
    initial_orientation = [Quaternion([1., 0., 0., 0.])]

    def test_mobility(orientation):
      return np.array([
        [1., orientation[0].s*orientation[0].p[0], 0.],
        [orientation[0].s*orientation[0].p[0], 1., 0.],
        [0., 0., 1.],])

    def zero_torque_calculator(orientation):
      return np.zeros(3)
    
    test_integrator = QuaternionIntegrator(test_mobility, initial_orientation,
                                           zero_torque_calculator)

    [avg_drift, avg_cov] = test_integrator.estimate_drift_and_covariance(0.01, 80000, 'RFD')
    self.assertLess(abs(avg_drift[0]), TOL)
    self.assertLess(abs(avg_drift[1] - 0.5), TOL)
    self.assertLess(abs(avg_drift[2]), TOL)

    true_covariance = test_mobility(initial_orientation)
    
    for j in range(3):
      for k in range(3):
        self.assertLess(abs(avg_cov[j, k] - true_covariance[j, k]), TOL)


    
if __name__ == "__main__":
  unittest.main()      

