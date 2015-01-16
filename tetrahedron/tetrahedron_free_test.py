''' Unit tests for tetrahedron_free '''

import sys
sys.path.append('..')

import unittest
import numpy as np
import random
import tetrahedron_free as tf
from quaternion_integrator.quaternion import Quaternion
from quaternion_integrator.quaternion_integrator import QuaternionIntegrator
import tetrahedron as tdn

class TestFreeTetrahedron(unittest.TestCase):
  
  def setUp(self):
    pass

  def test_get_free_r_vectors(self):
    ''' Test that we get the correct r_vectors with a free tetrahedron.'''
    # Test the identity orientation and flipped over by rotating 180 degrees 
    # around the x axis.
    # Identity first.
    theta = Quaternion([1., 0., 0., 0.])
    location = [20., 20., 20.]
    r_vectors = tf.get_free_r_vectors(location, theta)
    
    # Check r1.
    self.assertAlmostEqual(r_vectors[0][0], 20.)
    self.assertAlmostEqual(r_vectors[0][1], 20. + 2./np.sqrt(3.))
    self.assertAlmostEqual(r_vectors[0][2], 20. - 2.*np.sqrt(2.)/np.sqrt(3.))
    # Check r2.
    self.assertAlmostEqual(r_vectors[1][0], 19.)
    self.assertAlmostEqual(r_vectors[1][1], 20. - 1./np.sqrt(3.))
    self.assertAlmostEqual(r_vectors[1][2], 20. - 2.*np.sqrt(2.)/np.sqrt(3.))
    # Check r3.
    self.assertAlmostEqual(r_vectors[2][0], 21.)
    self.assertAlmostEqual(r_vectors[2][1], 20. - 1./np.sqrt(3.))
    self.assertAlmostEqual(r_vectors[2][2], 20. - 2.*np.sqrt(2.)/np.sqrt(3.))

    # Orientation upside down, rotate around x axis by 180 degrees.
    theta = Quaternion([0., 1., 0., 0.])
    r_vectors = tf.get_free_r_vectors(location, theta)

    # Check r1.
    self.assertAlmostEqual(r_vectors[0][0], 20.)
    self.assertAlmostEqual(r_vectors[0][1], 20. - 2./np.sqrt(3.))
    self.assertAlmostEqual(r_vectors[0][2], 20. + 2.*np.sqrt(2.)/np.sqrt(3.))
    # Check r2.
    self.assertAlmostEqual(r_vectors[1][0], 19.)
    self.assertAlmostEqual(r_vectors[1][1], 20. + 1./np.sqrt(3.))
    self.assertAlmostEqual(r_vectors[1][2], 20. + 2.*np.sqrt(2.)/np.sqrt(3.))
    # Check r3.
    self.assertAlmostEqual(r_vectors[2][0], 21.)
    self.assertAlmostEqual(r_vectors[2][1], 20. + 1./np.sqrt(3.))
    self.assertAlmostEqual(r_vectors[2][2], 20. + 2.*np.sqrt(2.)/np.sqrt(3.))

  
  def test_calc_free_rot_matrix(self):
    '''Test that we get the correct rotation matrix.'''
    # First construct any random unit quaternion. Not uniform.
    s = 2*random.random() - 1.
    p1 = (2. - 2*np.abs(s))*random.random() - (1. - np.abs(s))
    p2 = ((2. - 2.*np.abs(s) - 2.*np.abs(p1))*random.random() - 
          (1. - np.abs(s) - np.abs(p1)))
    p3 = np.sqrt(1. - s**2 - p1**2 - p2**2)
    theta = Quaternion(np.array([s, p1, p2, p3]))
    # Construct location and get r vectors.
    location = [10., 20., 30.]
    r_vectors = tf.get_free_r_vectors(location, theta)

    rot_matrix = tf.calc_free_rot_matrix(r_vectors, location)

    for j in range(3):
      r = r_vectors[j] - location
      block = [[0., r[2], -1.*r[1]],
               [-1.*r[2], 0., r[0]],
               [r[1], -1.*r[0], 0.]]
      for k in range(3):
        for l in range(3):
          self.assertAlmostEqual(block[k][l], rot_matrix[3*j + k, l])


  def test_free_torque(self):
    ''' Test that the torque is correct for a semi-random configuration. '''
    # First construct any random unit quaternion. Not uniform.
    s = 2*random.random() - 1.
    p1 = (2. - 2*np.abs(s))*random.random() - (1. - np.abs(s))
    p2 = ((2. - 2.*np.abs(s) - 2.*np.abs(p1))*random.random() - 
          (1. - np.abs(s) - np.abs(p1)))
    p3 = np.sqrt(1. - s**2 - p1**2 - p2**2)
    theta = Quaternion(np.array([s, p1, p2, p3]))
    # Construct location and get r vectors.
    location = [10., 20., 2.0]
    r_vectors = tf.get_free_r_vectors(location, theta)
    
    # Because height is 2, the particles are guaranteed to be within a cutoff of
    # 4.
    force_array = np.array([
      0., 0., 
      tf.REPULSION_STRENGTH*(tf.REPULSION_CUTOFF - r_vectors[0][2]) - tf.M1,
      0., 0., 
      tf.REPULSION_STRENGTH*(tf.REPULSION_CUTOFF - r_vectors[1][2]) - tf.M2,
      0., 0., 
      tf.REPULSION_STRENGTH*(tf.REPULSION_CUTOFF - r_vectors[2][2]) - tf.M3])
    
    rot_matrix = tf.calc_free_rot_matrix(r_vectors, location)

    correct_torque = np.dot(rot_matrix.T, force_array)
    calculated_torque = tf.free_gravity_torque_calculator([location], [theta])
    for k in range(3):
      self.assertAlmostEqual(calculated_torque[k], correct_torque[k])


  def test_free_force(self):
    ''' Test that the force is correct for a semi-random configuration. '''
    # First construct any random unit quaternion. Not uniform.
    s = 2*random.random() - 1.
    p1 = (2. - 2*np.abs(s))*random.random() - (1. - np.abs(s))
    p2 = ((2. - 2.*np.abs(s) - 2.*np.abs(p1))*random.random() - 
          (1. - np.abs(s) - np.abs(p1)))
    p3 = np.sqrt(1. - s**2 - p1**2 - p2**2)
    theta = Quaternion(np.array([s, p1, p2, p3]))
    # Construct location and get r vectors.
    location = [10., 20., 2.0]
    r_vectors = tf.get_free_r_vectors(location, theta)

    correct_force = np.zeros(3)
    # Add repulsion force from top vertex and all gravity.
    correct_force[2] += (
      tf.REPULSION_STRENGTH*(tf.REPULSION_CUTOFF - location[2]) - tf.M1 -tf.M2 
      - tf.M3 - tf.M4)
    # Add repulsion force from each other particle
    for k in range(3):
      correct_force[2] += (
        tf.REPULSION_STRENGTH*(tf.REPULSION_CUTOFF - r_vectors[k][2]))

    calculated_force = tf.free_gravity_force_calculator([location], [theta])

    for k in range(3):
      self.assertAlmostEqual(calculated_force[k], correct_force[k])


    
    
    
    

if __name__ == '__main__':
  unittest.main()

    
    
    

