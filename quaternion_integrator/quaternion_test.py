import unittest
import numpy as np
import random
from quaternion import Quaternion

class TestQuaternion(unittest.TestCase):

  def setUp(self):
    pass

  def test_quaternion_from_rotation(self):
    ''' Test that we correctly create a quaternion from an angle '''
    # Generate a random rotation vector
    phi = np.random.rand(3)
    phi_norm = np.linalg.norm(phi)
    theta = Quaternion.from_rotation(phi)
    
    self.assertAlmostEqual(theta.entries[0], np.cos(phi_norm/2.))
    self.assertAlmostEqual(theta.entries[1], np.sin(phi_norm/2.)*phi[0]/phi_norm)
    self.assertAlmostEqual(theta.entries[2], np.sin(phi_norm/2.)*phi[1]/phi_norm)
    self.assertAlmostEqual(theta.entries[3], np.sin(phi_norm/2.)*phi[2]/phi_norm)

  def test_quaternion_rot_matrix_det_one(self):
    ''' Test that the determinant of the rotation matrix is 1.'''
    for _ in range(10):
      theta = np.random.normal(0., 1., 4)
      theta = Quaternion(theta/np.linalg.norm(theta))
      R = theta.rotation_matrix()
      self.assertAlmostEqual(np.linalg.det(R), 1.0)

  
  def test_multiply_quaternions(self):
    ''' Test that quaternion multiplication works '''
    # First construct any random unit quaternion. Not uniform.
    s = 2*random.random() - 1.
    p1 = (2. - 2*np.abs(s))*random.random() - (1. - np.abs(s))
    p2 = ((2. - 2.*np.abs(s) - 2.*np.abs(p1))*random.random() - 
          (1. - np.abs(s) - np.abs(p1)))
    p3 = np.sqrt(1. - s**2 - p1**2 - p2**2)
    theta1 = Quaternion(np.array([s, p1, p2, p3]))

    # Construct another quaternion.
    t = 2*random.random() - 1.
    q1 = (2. - 2*np.abs(s))*random.random() - (1. - np.abs(s))
    q2 = ((2. - 2.*np.abs(s) - 2.*np.abs(p1))*random.random() - 
          (1. - np.abs(s) - np.abs(p1)))
    q3 = np.sqrt(1. - s**2 - p1**2 - p2**2)
    theta2 = Quaternion(np.array([t, q1, q2, q3]))
    
    product = theta1*theta2
    
    self.assertAlmostEqual(product.s, t*s - p1*q1 - p2*q2 - p3*q3)
    self.assertAlmostEqual(product.entries[1], s*q1 + t*p1 + p2*q3 - p3*q2)
    self.assertAlmostEqual(product.entries[2], s*q2 + t*p2 + p3*q1 - p1*q3)
    self.assertAlmostEqual(product.entries[3], s*q3 + t*p3 + p1*q2 - p2*q1)

    
  def test_quaternion_rotation_matrix(self):
    ''' Test that we create the correct rotation matrix for a quaternion. '''
    # First construct any random unit quaternion. Not uniform.
    s = 2*random.random() - 1.
    p1 = (2. - 2*np.abs(s))*random.random() - (1. - np.abs(s))
    p2 = ((2. - 2.*np.abs(s) - 2.*np.abs(p1))*random.random() - 
          (1. - np.abs(s) - np.abs(p1)))
    p3 = np.sqrt(1. - s**2 - p1**2 - p2**2)
    theta = Quaternion(np.array([s, p1, p2, p3]))
    
    R = theta.rotation_matrix()

    self.assertAlmostEqual(R[0][0], 2.*(theta.s**2 + theta.p[0]**2 - 0.5))
    self.assertAlmostEqual(R[0][1], 2.*(theta.p[0]*theta.p[1] - 
                                        theta.s*theta.p[2]))
    self.assertAlmostEqual(R[1][0], 2.*(theta.p[0]*theta.p[1] +
                                        theta.s*theta.p[2]))
    self.assertAlmostEqual(R[1][1], 2.*(theta.s**2 + theta.p[1]**2 - 0.5))
    self.assertAlmostEqual(R[2][2], 2.*(theta.s**2 + theta.p[2]**2 - 0.5))
    self.assertAlmostEqual(R[2][0], 2.*(theta.p[0]*theta.p[2] - theta.s*theta.p[1]))


  def test_rot_matrix_against_rodriguez(self):
    ''' 
    Test that given an angle of rotation, the quaternion
    rotation matrix matches Rodriguez formula.
    '''
    # Generate a random rotation, phi.
    phi = np.random.normal(0., 1., 3)
    phi = phi/np.linalg.norm(phi)
    magnitude = np.random.uniform(0., np.pi)

    theta = Quaternion.from_rotation(phi*magnitude)
    R = theta.rotation_matrix()
    omega = np.array([[0., -1.*phi[2], phi[1]],
                     [phi[2], 0., -1.*phi[0]],
                     [-1.*phi[1], phi[0], 0.]])
    R_rodriguez = (np.identity(3) + np.sin(magnitude)*omega +
                   np.inner(omega, omega.T)*(1. - np.cos(magnitude)))

    for j in range(3):
      for k in range(3):
        self.assertAlmostEqual(R[j, k], R_rodriguez[j, k])
                                        

  def test_quaternion_inverse(self):
    '''Test that the quaternion inverse works.'''
    # First construct any random unit quaternion. Not uniform.
    s = 2*random.random() - 1.
    p1 = (2. - 2*np.abs(s))*random.random() - (1. - np.abs(s))
    p2 = ((2. - 2.*np.abs(s) - 2.*np.abs(p1))*random.random() - 
          (1. - np.abs(s) - np.abs(p1)))
    p3 = np.sqrt(1. - s**2 - p1**2 - p2**2)
    theta = Quaternion(np.array([s, p1, p2, p3]))

    theta_inv = theta.inverse()
    
    identity = theta*theta_inv
    self.assertAlmostEqual(identity.s, 1.0)
    self.assertAlmostEqual(identity.p[0], 0.0)
    self.assertAlmostEqual(identity.p[1], 0.0)
    self.assertAlmostEqual(identity.p[2], 0.0)

    
  def test_quaternion_rotation_angle(self):
    ''' Test generating rotation angle from quaternion. '''
    # First construct any random unit quaternion. Not uniform.
    s = 2*random.random() - 1.
    p1 = (2. - 2*np.abs(s))*random.random() - (1. - np.abs(s))
    p2 = ((2. - 2.*np.abs(s) - 2.*np.abs(p1))*random.random() - 
          (1. - np.abs(s) - np.abs(p1)))
    p3 = np.sqrt(1. - s**2 - p1**2 - p2**2)
    theta = Quaternion(np.array([s, p1, p2, p3]))

    rotation_angle = theta.rotation_angle()
    phi = Quaternion.from_rotation(rotation_angle)

    self.assertAlmostEqual(phi.s, theta.s)
    self.assertAlmostEqual(phi.p[0], theta.p[0])
    self.assertAlmostEqual(phi.p[1], theta.p[1])
    self.assertAlmostEqual(phi.p[2], theta.p[2])

  def test_quaternion_stability(self):
    ''' Test numerical stability of quaternion multiplication.'''
    # This test is just to see roughly how bad things are and
    # how quickly the quaternions become un-normalized.  Seems somewhat slow, 
    # so that's good.
    orientation = Quaternion([1., 0., 0., 0.])
    max_err = 0.0
    for k in range(100000):
      theta = np.random.normal(0., 1., 4)
      theta = Quaternion(theta/np.linalg.norm(theta))
      orientation = theta*orientation
      norm_err = abs(orientation.s**2 + np.dot(orientation.p, orientation.p)
                     - 1.0)
      if norm_err > max_err:
        max_err = norm_err 
    print('max_err is ', max_err)
    self.assertAlmostEqual(max_err, 0.0)


  def test_rot_matrix_stability(self):
    ''' Test numerical stability of rotation matrices'''
    # This test is just to see roughly how bad things are and
    # how quickly rotation matrices become not orthogonal.
    max_norm_err = 0.
    max_orthogonal_err = 0.
    R = np.identity(3)
    for k in range(100000):
      phi = np.random.normal(0., 1., 3)
      phi = phi/np.linalg.norm(phi)
      magnitude = np.random.uniform(0., np.pi)
      omega = np.array([[0., -1.*phi[2], phi[1]],
                        [phi[2], 0., -1.*phi[0]],
                        [-1.*phi[1], phi[0], 0.]])
      R_increment = (np.identity(3) + np.sin(magnitude)*omega +
                     np.inner(omega, omega.T)*(1. - np.cos(magnitude)))
      R = np.inner(R_increment, R.T)
      
      norm_err = max([abs(np.linalg.norm(R[0]) - 1.0),
                      abs(np.linalg.norm(R[1]) - 1.0),
                      abs(np.linalg.norm(R[2]) - 1.0)])
      if norm_err > max_norm_err:
        max_norm_err = norm_err 
      orthogonal_err = max([abs(np.inner(R[0], R[1])),
                            abs(np.inner(R[0], R[2])),
                            abs(np.inner(R[1], R[2]))])
      if orthogonal_err > max_orthogonal_err:
        max_orthogonal_err = orthogonal_err 


    print('max_norm_err is ', max_norm_err)
    print('max_orthogonal_err is ', max_orthogonal_err)
    self.assertAlmostEqual(max_norm_err, 0.0)
    self.assertAlmostEqual(max_orthogonal_err, 0.0)


    
if __name__ == '__main__':
  unittest.main()
