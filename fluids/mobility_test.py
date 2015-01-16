''' Test the fluid mobilities. '''
import sys
sys.path.append('..')
import unittest
import numpy as np
import random
from quaternion_integrator.quaternion import Quaternion
from quaternion_integrator.quaternion_integrator import QuaternionIntegrator
import mobility


class TestMobility(unittest.TestCase):

  def setUp(self):
    pass


  def test_stokes_doublet_e1(self):
    ''' Test stokes doublet when r = e1. '''
    r = np.array([1., 0., 0.])
    
    doublet = mobility.stokes_doublet(r)
    
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
    dipole = mobility.potential_dipole(r)
    actual = (1./(4*np.pi))*np.array([
        [2., 0., 0.],
        [0., -1., 0.],
        [0., 0., 1.],
        ])

    for j in range(3):
      for k in range(3):
        self.assertAlmostEqual(dipole[j, k], actual[j, k])


  def test_image_system_zero_at_wall(self):
    ''' Test that the image system gives the correct 0 velocity at the wall.'''
    # Identity quaternion represents initial configuration.
    r_vectors = [np.array([0., 0., 0.]),
                 np.array([2., 2., 2.]),
                 np.array([1., 1., 1.])]
    fluid_mobility = mobility.image_singular_stokeslet(r_vectors)
    # Test particle 1 to particle 0.
    block = fluid_mobility[0:3, 3:6]
    for j in range(3):
      for k in range(3):
        self.assertAlmostEqual(block[j, k], 0.)
    # Test particle 2 to particle 0
    block = fluid_mobility[0:3, 6:9]
    for j in range(3):
      for k in range(3):
        self.assertAlmostEqual(block[j, k], 0.)


if __name__ == '__main__':
  unittest.main()
