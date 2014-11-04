""" Test the functions used in the tetrahedron script. """

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

    r_vectors = tetrahedron.GetRVectors(theta)
    
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

    r_vectors = tetrahedron.GetRVectors(theta)
    
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
    
  def test_stokes_doublet_e1(self):
    ''' Test stokes doublet when r = e1. '''
    r = np.array([1., 0., 0.])
    
    doublet = tetrahedron.StokesDoublet(r)
    
    actual = (1./(8*np.pi))*np.array([
        [0., 0., 1.],
        [0., 0., 0.],
        [1., 0., 0.],
        ])
    for j in range(3):
      for k in range(3):
        self.assertAlmostEqual(doublet[j, k], actual[j, k])

  def test_torque_calculator(self):
    ''' Test torque for a couple different configurations. '''
    # Identity quaternion.
    theta = Quaternion([1., 0., 0., 0.])
    torque = tetrahedron.TorqueCalculator([theta])
    for k in range(3):
      self.assertAlmostEqual(torque[k], 0.)
      
    # Upside down.
    theta = Quaternion([0., 1., 0., 0.])
    torque = tetrahedron.TorqueCalculator([theta])
    for k in range(3):
      self.assertAlmostEqual(torque[k], 0.)

    # Sideways.
    theta = Quaternion([1/np.sqrt(2.), 1./np.sqrt(2.), 0., 0.])
    torque = tetrahedron.TorqueCalculator([theta])
    self.assertAlmostEqual(torque[0], 2*np.sqrt(6.))
    for k in range(1, 3):
      self.assertAlmostEqual(torque[k], 0.)
      
if __name__ == '__main__':
  unittest.main()
    


