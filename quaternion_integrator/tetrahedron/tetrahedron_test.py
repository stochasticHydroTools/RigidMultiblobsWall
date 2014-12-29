""" Test the functions used in the tetrahedron script. """
import sys
import unittest
import numpy as np
import random
from quaternion_integrator.quaternion import Quaternion
from quaternion_integrator.quaternion_integrator import QuaternionIntegrator
import tetrahedron

# Some tests just print quantities, and
# this disables the prints if False.
PRINTOUT = False  


class MockIntegrator(object):
  ''' Mock Quaternion Integrator for Rotational MSD test. '''
  def __init__(self):
    self.position = [Quaternion([1., 0., 0., 0.])]
    self.kT = 1.0
    
  def additive_em_time_step(self, dt):
    '''
    Mock EM timestep.  Just set position to 
    (1/sqrt(2), 1/sqrt(2), 0, 0.).
    '''
    self.position = [Quaternion([1./np.sqrt(2.), 1./np.sqrt(2.), 0., 0.])]

  def rfd_time_step(self, dt):
    '''
    Mock RFD timestep.  Just set position to 
    (1/sqrt(2), 1/sqrt(2), 0, 0.).
    '''
    self.position = [Quaternion([1./np.sqrt(2.), 1./np.sqrt(2.), 0., 0.])]

  def fixman_time_step(self, dt):
    '''
    Mock Fixman timestep.  Just set position to 
    (1/sqrt(2), 1/sqrt(2), 0, 0.).
    '''
    self.position = [Quaternion([1./np.sqrt(2.), 1./np.sqrt(2.), 0., 0.])]


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
    self.assertAlmostEqual(r_vectors[0][2], tetrahedron.H
                           - 2.*np.sqrt(2.)/np.sqrt(3.))
    #Check r2
    self.assertAlmostEqual(r_vectors[1][0], -1.)
    self.assertAlmostEqual(r_vectors[1][1], -1./np.sqrt(3.))
    self.assertAlmostEqual(r_vectors[1][2], tetrahedron.H
                           - 2.*np.sqrt(2.)/np.sqrt(3.))
    #Check r3
    self.assertAlmostEqual(r_vectors[2][0], 1.)
    self.assertAlmostEqual(r_vectors[2][1], -1./np.sqrt(3.))
    self.assertAlmostEqual(r_vectors[2][2], tetrahedron.H
                           - 2.*np.sqrt(2.)/np.sqrt(3.))

    # Try quaternion that flips tetrahedron 180 degrees, putting it upside down.
    theta = Quaternion([0., -1., 0., 0.])
    r_vectors = tetrahedron.get_r_vectors(theta)
    
    #Check r1
    self.assertAlmostEqual(r_vectors[0][0], 0.)
    self.assertAlmostEqual(r_vectors[0][1], -2./np.sqrt(3.))
    self.assertAlmostEqual(r_vectors[0][2], tetrahedron.H + 
                           2.*np.sqrt(2.)/np.sqrt(3.))
    #Check r2
    self.assertAlmostEqual(r_vectors[1][0], -1.)
    self.assertAlmostEqual(r_vectors[1][1], 1./np.sqrt(3.))
    self.assertAlmostEqual(r_vectors[1][2], tetrahedron.H + 
                           2.*np.sqrt(2.)/np.sqrt(3.))
    #Check r3
    self.assertAlmostEqual(r_vectors[2][0], 1.)
    self.assertAlmostEqual(r_vectors[2][1], 1./np.sqrt(3.))
    self.assertAlmostEqual(r_vectors[2][2], tetrahedron.H + 
                           2.*np.sqrt(2.)/np.sqrt(3.))


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


  def test_image_system_zero_at_wall(self):
    ''' Test that the image system gives the correct 0 velocity at the wall.'''
    # Identity quaternion represents initial configuration.
    r_vectors = [np.array([0., 0., 0.]),
                 np.array([2., 2., 2.]),
                 np.array([1., 1., 1.])]
    mobility = tetrahedron.image_singular_stokeslet(r_vectors)
    # Test particle 1 to particle 0.
    block = mobility[0:3, 3:6]
    for j in range(3):
      for k in range(3):
        self.assertAlmostEqual(block[j, k], 0.)
    # Test particle 2 to particle 0
    block = mobility[0:3, 6:9]
    for j in range(3):
      for k in range(3):
        self.assertAlmostEqual(block[j, k], 0.)

  def test_torque_rotor_x(self):
    ''' 
    Test that we get the correct angular velocity from a torque
    applied to a rotor in the x  and y direction.  Rotor looks like:
                                       z y  
                         0             |/          
                         |              ---> x   
                     0-------0
                         |
                         0
    '''
    # Overwrite height for this test, want to ignore the wall.
    old_height = tetrahedron.H
    tetrahedron.H = 1e8
    # Set up a rod to spin with positive x torque.
    r_vectors = [np.array([0., 0., 1e8 + 2.]),
                 np.array([2., 0., 1e8]),
                 np.array([-2., 0., 1e8]),
                 np.array([0., 0., 1e8 - 2.])]
    
    # Calculate \Tau -> Omega mobility
    mobility = tetrahedron.torque_oseen_mobility(r_vectors)
    
    # Check that the first column matches the correct result. X torque.
    # r*omega = (1/(6 pi eta a) - 1/(8 pi eta 2r)) F
    # T = 2r F  =>  F = T/2R
    # r = 2
    omega_x = ((1./2./np.pi/tetrahedron.ETA)*
               (1./(6.*tetrahedron.A) - 1./(8.*4)))/4.

    self.assertAlmostEqual(mobility[0, 0], omega_x)
    self.assertAlmostEqual(mobility[1, 0], 0.)
    self.assertAlmostEqual(mobility[2, 0], 0.)


    # Check that the second column matches the correct result. Y torque.
    # r*omega = (1/(6 pi eta a) + 2/(8*pi*eta sqrt(2)r 2) - 1/(8 pi eta 2r))F
    # T = 4r F
    # r = 2
    omega_y = ((1./2./np.pi/tetrahedron.ETA)*
             (1./(6.*tetrahedron.A) + 1./(16.*np.sqrt(2)) - 1./32.)/8.)

    self.assertAlmostEqual(mobility[0, 1], 0.)
    self.assertAlmostEqual(mobility[1, 1], omega_y)
    self.assertAlmostEqual(mobility[2, 1], 0.)

    # Replace height with the tetrahedron module height.
    tetrahedron.H = old_height

  def test_torque_calculator(self):
    ''' Test torque for a couple different configurations. '''
    # Overwrite Masses.
    old_masses = [tetrahedron.M1, tetrahedron.M2, tetrahedron.M3]
    tetrahedron.M1 = 1.0
    tetrahedron.M2 = 1.0
    tetrahedron.M3 = 1.0
    
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

    tetrahedron.M1 = old_masses[0]
    tetrahedron.M2 = old_masses[1]
    tetrahedron.M3 = old_masses[2]

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
    stokeslet = tetrahedron.image_singular_stokeslet(r_vectors)
    print "stokeslet is "
    print stokeslet
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
    mobility = tetrahedron.single_wall_fluid_mobility(r_vectors, 1., 1.)

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
    mobility = tetrahedron.single_wall_fluid_mobility(r_vectors, 1., a)
    for j in range(3):
      for k in range(3):
        self.assertAlmostEqual(mobility[j, 3 + k], 0.0, places=6)

  def test_finite_size_limit(self):
    ''' Test that the finite size mobility goes to the oseen for small a.'''
    n_particles = 4
    a = 0.000001
    old_a = tetrahedron.A
    tetrahedron.A = a
    # Random configuration, all above wall.
    r_vectors = [np.random.normal(5., 1., 3) for _ in range(n_particles)]

    mobility_finite = tetrahedron.single_wall_fluid_mobility(r_vectors, 1., a)
    mobility_point = tetrahedron.image_singular_stokeslet(r_vectors)
    for j in range(3*n_particles):
      for k in range(3*n_particles):
        # We multiply by a to get reasonable numbers for mobility.
        self.assertAlmostEqual(a*mobility_finite[j,k], a*mobility_point[j,k])
        
    tetrahedron.A = old_a

  def test_print_divergence_term(self):
    ''' Just print the avg divergence term for certain orientations. '''
    # Construct any random unit quaternion to rotate. Not uniform.
    s = 2*random.random() - 1.
    p1 = (2. - 2*np.abs(s))*random.random() - (1. - np.abs(s))
    p2 = ((2. - 2.*np.abs(s) - 2.*np.abs(p1))*random.random() - 
          (1. - np.abs(s) - np.abs(p1)))
    p3 = np.sqrt(1. - s**2 - p1**2 - p2**2)
    theta = Quaternion(np.array([s, p1, p2, p3]))
    theta = Quaternion([1./np.sqrt(2.), 1./np.sqrt(2.), 0., 0.])
    integrator = QuaternionIntegrator(tetrahedron.tetrahedron_mobility,
                                      [theta],
                                      tetrahedron.gravity_torque_calculator)
    div_term = integrator.estimate_divergence()
    if PRINTOUT:
      print "\n"
      print "divergence term is ", div_term

  def test_rpy_tensor_value_diagonal(self):
    ''' Test that the free rotational mobility of the tetrahedron is diagonal. '''
    n_particles = 3
    # Random configuration.
    # Center of mass is (0, 0, -1/sqrt(2) )
    r_vectors = [np.array([0., 0., np.sqrt(3.)/np.sqrt(2.)]),
                 np.array([0., 2./np.sqrt(3.), -1./(np.sqrt(6.))]),
                 np.array([-1., -1./np.sqrt(3.), -1./(np.sqrt(6.))]),
                 np.array([1., -1./np.sqrt(3.), -1./(np.sqrt(6.))])]

    # Construct any random unit quaternion to rotate. Not uniform.
    s = 2*random.random() - 1.
    p1 = (2. - 2*np.abs(s))*random.random() - (1. - np.abs(s))
    p2 = ((2. - 2.*np.abs(s) - 2.*np.abs(p1))*random.random() - 
          (1. - np.abs(s) - np.abs(p1)))
    p3 = np.sqrt(1. - s**2 - p1**2 - p2**2)
    theta = Quaternion(np.array([s, p1, p2, p3]))

    r_vectors = [np.dot(theta.rotation_matrix(), vec) for vec in r_vectors]
    mobility = tetrahedron.rpy_torque_mobility(r_vectors)
    
    for j in range(3):
      for k in range(j+1, 3):
        self.assertAlmostEqual(mobility[j, k], 0.)
        self.assertAlmostEqual(mobility[k, j], 0.)


  def test_calc_rotational_msd(self):
    ''' 
    Test that calc_rotational_msd does the right thing for a simple mock.
    integrator.
    '''
    TOL = 5e-2
    initial_orientation = [Quaternion([1., 0., 0., 0.])]
    integrator = MockIntegrator()
    msd = tetrahedron.calc_rotational_msd(integrator, 'RFD', 1.0, 1000, initial_orientation)
    self.assertLess(abs(msd), TOL)

if __name__ == '__main__':
  unittest.main()
