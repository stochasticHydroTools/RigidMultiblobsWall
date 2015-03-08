""" Test the functions used in the tetrahedron script. """
import unittest
import numpy as np
import random
import sys
sys.path.append('..')

from quaternion_integrator.quaternion import Quaternion
from quaternion_integrator.quaternion_integrator import QuaternionIntegrator
import tetrahedron
import tetrahedron_rotational_msd
import fluids.mobility as mb

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

if __name__ == '__main__':
  unittest.main()
