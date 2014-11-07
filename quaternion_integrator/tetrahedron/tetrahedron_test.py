""" Test the functions used in the tetrahedron script. """
import sys
sys.path.append('../')

import unittest
import numpy as np
import random
from quaternion import Quaternion
import tetrahedron

class TestTetrahedron(unittest.TestCase):

  def setUp(self):
    pass

  def test_get_r_vectors(self):
    ''' Test that we can get R vectors correctly for a few rotations. '''
    # Identity quaternion.
    theta = Quaternion([1., 0., 0., 0.])

    r_vectors = tetrahedron.get_r_vectors(theta)
    
    #Check r1
    self.assertAlmostEqual(r_vectors[0][0], 0.)
    self.assertAlmostEqual(r_vectors[0][1], 2./np.sqrt(3.))
    self.assertAlmostEqual(r_vectors[0][2], -2.*np.sqrt(2.)/np.sqrt(3.))

    #Check r2
    self.assertAlmostEqual(r_vectors[1][0], -1.)
    self.assertAlmostEqual(r_vectors[1][1], -1./np.sqrt(3.))
    self.assertAlmostEqual(r_vectors[1][2], -2.*np.sqrt(2.)/np.sqrt(3.))

    #Check r3
    self.assertAlmostEqual(r_vectors[2][0], 1.)
    self.assertAlmostEqual(r_vectors[2][1], -1./np.sqrt(3.))
    self.assertAlmostEqual(r_vectors[2][2], -2.*np.sqrt(2.)/np.sqrt(3.))

    # Try quaternion that flips tetrahedron 180 degrees, putting it upside down.
    theta = Quaternion([0., -1., 0., 0.])

    r_vectors = tetrahedron.get_r_vectors(theta)
    
    #Check r1
    self.assertAlmostEqual(r_vectors[0][0], 0.)
    self.assertAlmostEqual(r_vectors[0][1], -2./np.sqrt(3.))
    self.assertAlmostEqual(r_vectors[0][2], 2.*np.sqrt(2.)/np.sqrt(3.))

    #Check r2
    self.assertAlmostEqual(r_vectors[1][0], -1.)
    self.assertAlmostEqual(r_vectors[1][1], 1./np.sqrt(3.))
    self.assertAlmostEqual(r_vectors[1][2], 2.*np.sqrt(2.)/np.sqrt(3.))

    #Check r3
    self.assertAlmostEqual(r_vectors[2][0], 1.)
    self.assertAlmostEqual(r_vectors[2][1], 1./np.sqrt(3.))
    self.assertAlmostEqual(r_vectors[2][2], 2.*np.sqrt(2.)/np.sqrt(3.))


  def test_r_matrix(self):
    ''' Test that we generate the correct R matrix.'''

    # Test the r_matrix for the initial configuration.
    r_matrix = tetrahedron.get_r_vectors(Quaternion([1., 0., 0., 0.]))
    
    
  def test_stokes_doublet_e1(self):
    ''' Test stokes doublet when r = e1. '''
    r = np.array([1., 0., 0.])
    
    doublet = tetrahedron.stokes_doublet(r)
    
    actual = (1./(8*np.pi))*np.array([
        [0., 0., 1.],
        [0., 0., 0.],
        [1., 0., 0.],
        ])
    for j in range(3):
      for k in range(3):
        self.assertAlmostEqual(doublet[j, k], actual[j, k])


  def test_stokes_dipole_e1(self):
    ''' Test dipole when r = e1. '''
    r = np.array([1., 0., 0.])
    dipole = tetrahedron.potential_dipole(r)
    actual = (1./(4*np.pi))*np.array([
        [2., 0., 0.],
        [0., -1., 0.],
        [0., 0., 1.],
        ])

    for j in range(3):
      for k in range(3):
        self.assertAlmostEqual(dipole[j, k], actual[j, k])


  def test_torque_calculator(self):
    ''' Test torque for a couple different configurations. '''
    # Overwrite Masses.
    tetrahedron.M1 = 1.0
    tetrahedron.M2 = 2.0
    tetrahedron.M3 = 3.0
    
    # Identity quaternion.
    theta = Quaternion([1., 0., 0., 0.])
    torque = tetrahedron.gravity_torque_calculator([theta])
    for k in range(3):
      self.assertAlmostEqual(torque[k], 0.)
      
    # Upside down.
    theta = Quaternion([0., 1., 0., 0.])
    torque = tetrahedron.gravity_torque_calculator([theta])
    for k in range(3):
      self.assertAlmostEqual(torque[k], 0.)

    # Sideways.
    theta = Quaternion([1/np.sqrt(2.), 1./np.sqrt(2.), 0., 0.])
    torque = tetrahedron.gravity_torque_calculator([theta])
    self.assertAlmostEqual(torque[0], -2*np.sqrt(6.))
    for k in range(1, 3):
      self.assertAlmostEqual(torque[k], 0.)


  def test_mobility_spd(self):
    ''' Test that for random configurations, the mobility is SPD. '''
    def is_pos_def(x):
      return np.all(np.linalg.eigvals(x) > 0)    

    # Generate 5 random configurations
    for _ in range(5):
      # First construct any random unit quaternion. Not uniform.
      s = 2*random.random() - 1.
      p1 = (2. - 2*np.abs(s))*random.random() - (1. - np.abs(s))
      p2 = ((2. - 2.*np.abs(s) - 2.*np.abs(p1))*random.random() - 
            (1. - np.abs(s) - np.abs(p1)))
      p3 = np.sqrt(1. - s**2 - p1**2 - p2**2)
      theta = Quaternion(np.array([s, p1, p2, p3]))

      mobility = tetrahedron.tetrahedron_mobility([theta])
      self.assertEqual(len(mobility), 3)
      self.assertEqual(len(mobility[0]), 3)
      self.assertTrue(is_pos_def(mobility))


  def test_image_system_spd(self):
    ''' Test that the image system for a singular stokeslet is SPD.'''
    # First construct any random unit quaternion. Not uniform.
    s = 2*random.random() - 1.
    p1 = (2. - 2*np.abs(s))*random.random() - (1. - np.abs(s))
    p2 = ((2. - 2.*np.abs(s) - 2.*np.abs(p1))*random.random() - 
          (1. - np.abs(s) - np.abs(p1)))
    p3 = np.sqrt(1. - s**2 - p1**2 - p2**2)
    theta = Quaternion(np.array([s, p1, p2, p3]))

    r_vectors = tetrahedron.get_r_vectors(theta)
    stokeslet = tetrahedron.image_singular_stokeslet(theta, r_vectors)

    def is_pos_def(x):
      return np.all(np.linalg.eigvals(x) > 0)    
    
    self.assertEqual(len(stokeslet), 9)
    self.assertEqual(len(stokeslet[0]), 9)
    self.assertTrue(is_pos_def(stokeslet))

      
if __name__ == '__main__':
  unittest.main()
