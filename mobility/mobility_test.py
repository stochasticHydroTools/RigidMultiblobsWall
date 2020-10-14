''' Test the fluid mobilities. '''

import unittest
import numpy as np
import random
import sys
sys.path.append('..')

from quaternion_integrator.quaternion import Quaternion
from quaternion_integrator.quaternion_integrator import QuaternionIntegrator
from . import mobility as mb


class TestMobility(unittest.TestCase):

  def setUp(self):
    pass

  def test_stokes_doublet_e1(self):
    ''' Test stokes doublet when r = e1. '''
    r = np.array([1., 0., 0.])
    
    doublet = mb.stokes_doublet(r)
    
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
    dipole = mb.potential_dipole(r)
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
    a = 0.5
    fluid_mobility = mb.image_singular_stokeslet(r_vectors, 1.0, a)
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
    # Random configuration, all above wall.
    r_vectors = [np.random.normal(5., 1., 3) for _ in range(n_particles)]

    mobility_finite = mb.single_wall_fluid_mobility(r_vectors, 1., a)
    mobility_point = mb.image_singular_stokeslet(r_vectors, 1.0, a)
    for j in range(3*n_particles):
      for k in range(3*n_particles):
        # We multiply by a to get reasonable numbers for mobility.
        self.assertAlmostEqual(a*mobility_finite[j,k], a*mobility_point[j,k])


  def test_image_system_spd(self):
    ''' Test that the image system for a singular stokeslet is SPD.'''
    n_particles = 5
    a = 0.25
    r_vectors = [np.random.normal(12., 2.5, 3) for _ in range(n_particles)]
    stokeslet = mb.image_singular_stokeslet(r_vectors, 1.0, a)
    def is_pos_def(x):
      return np.all(np.linalg.eigvals(x) > 0)    
    
    self.assertEqual(len(stokeslet), 3*n_particles)
    self.assertEqual(len(stokeslet[0]), 3*n_particles)
    self.assertTrue(is_pos_def(stokeslet))
    for j in range(3*n_particles):
      for k in range(j+1, 3*n_particles):
        self.assertAlmostEqual(stokeslet[j, k], stokeslet[k, j])

  def test_rpy_spd(self):
    ''' Test that the rotne prager tensor implementation gives an SPD tensor. '''
    # Number of particles
    n_particles = 4
    def is_pos_def(x):
      return np.all(np.linalg.eigvals(x) > 0)
    # Random configuration.
    r_vectors = [np.random.normal(0., 1., 3) for _ in range(n_particles)]
    
    rpy = mb.rotne_prager_tensor(r_vectors, 1., 1.)
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


  def test_single_wall_mobility_with_rotation_spd(self):
    ''' Test that the mobility with rotation is SPD. '''
    location = np.random.normal(10., 3., 3)
    eta = 1.0
    a = 0.25
    fluid_mobility = mb.single_wall_self_mobility_with_rotation(location, eta, a)
    def is_pos_def(x):
      return np.all(np.linalg.eigvals(x) > 0)    

    self.assertTrue(is_pos_def(fluid_mobility))
    for j in range(6):
      for k in range(j+1, 6):
        self.assertAlmostEqual(fluid_mobility[j][k], fluid_mobility[k][j])
    

  def test_sphere_wall_rotation_mobility_torque(self):
    ''' 
    Test that turning a sphere clockwise around the one axis 
    causes motion in the correct direction.
    '''
    location = np.random.normal(10., 3., 3)
    eta = 1.0
    a = 0.25
    fluid_mobility = mb.single_wall_self_mobility_with_rotation(location, eta, a)
    
    # Torque about x axis.
    self.assertAlmostEqual(fluid_mobility[0, 3], 0.)
    self.assertTrue(fluid_mobility[1, 3] > 0)
    self.assertAlmostEqual(fluid_mobility[2, 3], 0.)
    #Torque about y axis.
    self.assertTrue(fluid_mobility[0, 4] < 0)
    self.assertAlmostEqual(fluid_mobility[1, 4], 0.)
    self.assertAlmostEqual(fluid_mobility[2, 4], 0.)
    # Rotate around the Z axis.
    self.assertAlmostEqual(fluid_mobility[0, 5], 0.)
    self.assertAlmostEqual(fluid_mobility[1, 5], 0.)
    self.assertAlmostEqual(fluid_mobility[2, 5], 0.)
    
    

  def test_epsilon_tensor(self):
    ''' Check that we get the right cross epsilon for a few possible indices.'''
    self.assertAlmostEqual(1.0, mb.epsilon_tensor(0, 1, 2))
    self.assertAlmostEqual(0.0, mb.epsilon_tensor(0, 1, 1))
    self.assertAlmostEqual(-1.0, mb.epsilon_tensor(1, 0, 2))
    self.assertAlmostEqual(0.0, mb.epsilon_tensor(1, 1, 2))
    self.assertAlmostEqual(-1.0, mb.epsilon_tensor(2, 1, 0))
    self.assertAlmostEqual(1.0, mb.epsilon_tensor(2, 0, 1))

  def test_boosted_v_python_agreement(self):
    ''' 
    Test that for random R vectors, the boosted and python
    versions of mobility agree.'''
    location = [np.random.normal(10., 3., 3) for _ in range(4)]
    eta = 1.0
    a = 0.25
    fluid_mobility = mb.single_wall_fluid_mobility(location, eta, a)
    fluid_mobility_boost = mb.boosted_single_wall_fluid_mobility(
      location, eta, a)
    

    for i in range(len(fluid_mobility)):
      for j in range(len(fluid_mobility[0])):
        self.assertAlmostEqual(fluid_mobility[i, j], fluid_mobility_boost[i, j])
        

    

if __name__ == '__main__':
  unittest.main()
