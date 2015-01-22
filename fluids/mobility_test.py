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


  def test_finite_size_limit(self):
    ''' Test that the finite size mobility goes to the oseen for small a.'''
    n_particles = 4
    a = 0.000001
    old_a = mobility.A
    mobility.A = a
    # Random configuration, all above wall.
    r_vectors = [np.random.normal(5., 1., 3) for _ in range(n_particles)]

    mobility_finite = mobility.single_wall_fluid_mobility(r_vectors, 1., a)
    mobility_point = mobility.image_singular_stokeslet(r_vectors)
    for j in range(3*n_particles):
      for k in range(3*n_particles):
        # We multiply by a to get reasonable numbers for mobility.
        self.assertAlmostEqual(a*mobility_finite[j,k], a*mobility_point[j,k])
    mobility.A = old_a


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
    stokeslet = mb.image_singular_stokeslet(r_vectors)
    def is_pos_def(x):
      return np.all(np.linalg.eigvals(x) > 0)    
    
    self.assertEqual(len(stokeslet), 9)
    self.assertEqual(len(stokeslet[0]), 9)
    self.assertTrue(is_pos_def(stokeslet))
    for j in range(9):
      for k in range(j+1, 9):
        self.assertAlmostEqual(stokeslet[j, k], stokeslet[k, j])

  def test_rpy_spd(self):
    ''' Test that the rotne prager tensor implementation gives an SPD tensor. '''
    # Number of particles
    n_particles = 4
    def is_pos_def(x):
      return np.all(np.linalg.eigvals(x) > 0)
    # Random configuration.
    r_vectors = [np.random.normal(0., 1., 3) for _ in range(n_particles)]
    
    rpy = tetrahedron.rotne_prager_tensor(r_vectors, 1., 1.)
    self.assertTrue(is_pos_def(rpy))
    for j in range(3*n_particles):
      for k in range(j+1, 3*n_particles):
        self.assertAlmostEqual(rpy[j, k], rpy[k, j])

  def test_single_wall_mobility_spd(self):
    ''' Test that single wall mobility from Swan Brady paper is SPD. '''
    # Number of particles
    n_particles = 5
    height = 10.
    def is_pos_def(x):
      return np.all(np.linalg.eigvals(x) > 0)
    # Random configuration.
    r_vectors = [np.random.normal(height, 1., 3) for _ in range(n_particles)]
    mobility = mb.single_wall_fluid_mobility(r_vectors, 1., 1.)

    self.assertTrue(is_pos_def(mobility))
    for j in range(3*n_particles):
      for k in range(j+1, 3*n_particles):
        self.assertAlmostEqual(mobility[j, k], mobility[k, j])

  def test_single_wall_mobility_zero_at_wall(self):
    ''' 
    Test that single wall mobility from Swan Brady paper is zero for very small
    particles at the wall.
    '''
    a = 0.0001
    r_vectors = [np.array([0., 0., a]),
                 np.array([1., 1., 8.])]
    mobility = mb.single_wall_fluid_mobility(r_vectors, 1., a)
    for j in range(3):
      for k in range(3):
        self.assertAlmostEqual(mobility[j, 3 + k], 0.0, places=6)


if __name__ == '__main__':
  unittest.main()
