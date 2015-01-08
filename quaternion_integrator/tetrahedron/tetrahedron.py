'''
Script to test a tetrahedron near a wall.  The wall is at z = 0, and
the tetrahedron's "top" vertex is fixed at (0, 0, H).

This file has the mobility and torque calculator used for any tetrahedron
test.  Running this script will run a trajectory and bin the heights of each
of the three non-fixed vertices for Fixman, RFD, and EM timestepping, as well as
for the equilibrium distribution.  
'''
import sys
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot
import argparse
import cPickle
import cProfile, pstats, StringIO
import math
import time
import logging
import logging.handlers

from quaternion_integrator.quaternion import Quaternion
from quaternion_integrator.quaternion_integrator import QuaternionIntegrator
import uniform_analyzer as ua

# TODO: Move the fluid dynamics (not tetrahedron specific)
# stuff (mobilities,etc) to a diff file.

ETA = 1.0   # Fluid viscosity.
A = 0.5     # Particle Radius.
H = 2.5     # Distance to wall.

# Masses of particles.
M1 = 0.1
M2 = 0.2
M3 = 0.3

def identity_mobility(orientation):
  ''' Simple identity mobility for testing. '''
  return np.identity(3)


def test_mobility(orientation):
  ''' Simple mobility that's not divergence free. '''
  r_vectors = get_r_vectors(orientation[0])
  total_mobility = np.array([np.zeros(3) for _ in range(3)])
  for k in range(3):
    total_mobility[k, k] = r_vectors[k][2]**2 + 1.
  return total_mobility


def tetrahedron_mobility(orientation):
  ''' 
  Wrapper for torque mobility that takes a quaternion for
  use with quaternion_integrator. 
  '''
  r_vectors = get_r_vectors(orientation[0])
  return torque_mobility(r_vectors)

def torque_oseen_mobility(r_vectors):
  '''
  Calculate the mobility, torque -> angular velocity, at orientation 
  In this case, orientation is length 1, as there is just 1 quaternion.
  The mobility is equal to R M^-1 R^t where R is 3N x 3 (9 x 3)
  Rx = r cross x
  r is the distance from the fixed vertex of the tetrahedron to
  each other vertex (a length 3N vector).
  M (3N x 3N) is the singular image stokeslet for a point force near a wall, but
  we've replaced the diagonal piece by 1/(6 pi eta a).
  '''  
  mobility = image_singular_stokeslet(r_vectors)
  rotation_matrix = calculate_rot_matrix(r_vectors)
  total_mobility = np.linalg.inv(np.dot(rotation_matrix.T,
                                        np.dot(np.linalg.inv(mobility),
                                               rotation_matrix)))
  return total_mobility


def torque_mobility(r_vectors):
  '''
  Calculate the mobility, torque -> angular velocity, at orientation 
  In this case, orientation is length 1, as there is just 1 quaternion.
  The mobility is equal to R M^-1 R^t where R is 3N x 3 (9 x 3)
  Rx = r cross x
  r is the distance from the fixed vertex of the tetrahedron to
  each other vertex (a length 3N vector).
  M (3N x 3N) is the finite size single wall mobility taken from the
  Swan and Brady paper:
   "Simulation of hydrodynamically interacting particles near a no-slip
    boundary."
  '''  
  mobility = single_wall_fluid_mobility(r_vectors, ETA, A)
  rotation_matrix = calculate_rot_matrix(r_vectors)
  total_mobility = np.linalg.inv(np.dot(rotation_matrix.T,
                                        np.dot(np.linalg.inv(mobility),
                                               rotation_matrix)))
  return total_mobility


def rpy_torque_mobility(r_vectors):
  '''
  Calculate the mobility, torque -> angular velocity, at orientation 
  In this case, orientation is length 1, as there is just 1 quaternion.
  The mobility is equal to R M^-1 R^t where R is 3N x 3 (9 x 3)
  Rx = r cross x
  r is the distance from the fixed vertex of the tetrahedron to
  each other vertex (a length 3N vector).
  M (3N x 3N) is the RPY tensor.
  '''  
  mobility = rotne_prager_tensor(r_vectors, ETA, A)
  rotation_matrix = calculate_rot_matrix_cm(r_vectors)
  total_mobility = np.linalg.inv(np.dot(rotation_matrix.T,
                                        np.dot(np.linalg.inv(mobility),
                                               rotation_matrix)))
  return total_mobility


def image_singular_stokeslet(r_vectors):
  ''' Calculate the image system for the singular stokeslet (M above).'''
  mobility = np.array([
      np.zeros(3*len(r_vectors)) for _ in range(3*len(r_vectors))])
  # Loop through particle interactions
  for j in range(len(r_vectors)):
    for k in range(len(r_vectors)):
      if j != k:  #  do particle interaction
        r_particles = r_vectors[j] - r_vectors[k]
        r_norm = np.linalg.norm(r_particles)
        wall_dist = r_vectors[k][2]
        r_reflect = r_vectors[j] - (r_vectors[k] - 2.*np.array([0., 0., wall_dist]))
        r_ref_norm = np.linalg.norm(r_reflect)
        # Loop through components.
        for l in range(3):
          for m in range(3):
            # Two stokeslets, one with negative force at image.
            mobility[j*3 + l][k*3 + m] = (
              ((l == m)*1./r_norm + r_particles[l]*r_particles[m]/(r_norm**3) -
               ((l == m)*1./r_ref_norm + r_reflect[l]*r_reflect[m]/(r_ref_norm**3)))/
              (8.*np.pi))
        # Add doublet and dipole contribution.
        mobility[(j*3):(j*3 + 3), (k*3):(k*3 + 3)] += (
          doublet_and_dipole(r_reflect, wall_dist))
        
      else:
        # j == k
        mobility[(j*3):(j*3 + 3), (k*3):(k*3 + 3)] = 1./(6*np.pi*ETA*A)*np.identity(3)
  return mobility

def stokes_doublet(r):
  ''' Calculate stokes doublet from direction, strength, and r. '''
  r_norm = np.linalg.norm(r)
  e3 = np.array([0., 0., 1.])
  doublet = (np.outer(r, e3) + np.dot(r, e3)*np.identity(3) -
             np.outer(e3, r) - 3.*np.dot(e3, r)*np.outer(r, r)/(r_norm**2))
  # Negate the first two columns for the correct forcing.
  doublet[:, 0:2] = -1.*doublet[:, 0:2]
  doublet = doublet/(8*np.pi*(r_norm**3))
  return doublet

def potential_dipole(r):
  ''' Calculate potential dipole. '''
  r_norm = np.linalg.norm(r)
  dipole = np.identity(3) - 3.*np.outer(r, r)/(r_norm**2)
  # Negate the first two columns for the correct forcing.
  dipole[:, 0:2] = -1.*dipole[:, 0:2]
  dipole = dipole/(4.*np.pi*(r_norm**3))
  return dipole


def doublet_and_dipole(r, h):
  ''' 
  Just keep the pieces of the potential dipole and the doublet
  that we need for the image system.  No point in calculating terms that will cancel.
  This function includes the prefactors of 2H and H**2.  
  Seems to be significantly faster.
  '''
  r_norm = np.linalg.norm(r)
  e3 = np.array([0., 0., 1.])
  doublet_and_dipole = 2.*h*(np.outer(r, e3) - np.outer(e3, r))/(8.*np.pi*(r_norm**3))
  doublet_and_dipole[:, 0:2] = -1.*doublet_and_dipole[:, 0:2]
  return doublet_and_dipole


def single_wall_fluid_mobility(r_vectors, eta, a):
  ''' Mobility for particles near a wall.  This uses the expression from
  the Swan and Brady paper for a finite size particle, as opposed to the 
  Blake paper point particle result. '''
  num_particles = len(r_vectors)
  # We add the corrections from the appendix of the paper to the unbounded mobility.
  mobility = rotne_prager_tensor(r_vectors, eta, a)
  for j in range(num_particles):
    for k in range(j+1, num_particles):
      # Here notation is based on appendix C of the Swan and Brady paper:
      #  'Simulation of hydrodynamically interacting particles near a no-slip
      #   boundary.'
      h = r_vectors[k][2]
      R = (r_vectors[j] - (r_vectors[k] - 2.*np.array([0., 0., h])))/a
      R_norm = np.linalg.norm(R)
      e = R/R_norm
      e_3 = np.array([0., 0., e[2]])
      h_hat = h/(a*R[2])
      # Taken from Appendix C expression for M_UF
      mobility[(j*3):(j*3 + 3), (k*3):(k*3 + 3)] += (1./(6.*np.pi*eta*a))*(
        -0.25*(3.*(1. - 6.*h_hat*(1. - h_hat)*e[2]**2)/R_norm
               - 6.*(1. - 5.*e[2]**2)/(R_norm**3)
               + 10.*(1. - 7.*e[2]**2)/(R_norm**5))*np.outer(e, e)
         - (0.25*(3.*(1. + 2.*h_hat*(1. - h_hat)*e[2]**2)/R_norm
                  + 2.*(1. - 3.*e[2]**2)/(R_norm**3)
                  - 2.*(2. - 5.*e[2]**2)/(R_norm**5)))*np.identity(3)
         + 0.5*(3.*h_hat*(1. - 6.*(1. - h_hat)*e[2]**2)/R_norm
                - 6.*(1. - 5.*e[2]**2)/(R_norm**3)
                + 10.*(2. - 7.*e[2]**2)/(R_norm**5))*np.outer(e, e_3)
         + 0.5*(3.*h_hat/R_norm - 10./(R_norm**5))*np.outer(e_3, e)
         - (3.*(h_hat**2)*(e[2]**2)/R_norm 
            + 3.*(e[2]**2)/(R_norm**3)
            + (2. - 15.*e[2]**2)/(R_norm**5))*np.outer(e_3, e_3)/(e[2]**2))
      
      mobility[(k*3):(k*3 + 3), (j*3):(j*3 + 3)] = (
        mobility[(j*3):(j*3 + 3), (k*3):(k*3 + 3)].T)

  for j in range(len(r_vectors)):
    # Diagonal blocks, self mobility.
    h = r_vectors[j][2]/a
    for l in range(3):
      for m in range(3):
        mobility[j*3 + l][j*3 + m] += (1./(6.*np.pi*eta*a))*(
          (l == m)*(l != 2)*(-1./16.)*(9./h - 2./(h**3) + 1./(h**5))
          + (l == m)*(l == 2)*(-1./8.)*(9./h - 4./(h**3) + 1./(h**5)))
  return mobility


def rotne_prager_tensor(r_vectors, eta, a):
  ''' Calculate free rotne prager tensor for particles at locations given by
  r_vectors (list of 3 dimensional locationis) of radius a.'''
  num_particles = len(r_vectors)
  mobility = np.array([np.zeros(3*num_particles) for _ in range(3*num_particles)])
  for j in range(num_particles):
    for k in range(num_particles):
      if j != k:
        # Particle interaction, rotne prager.
        r = r_vectors[j] - r_vectors[k]
        r_norm = np.linalg.norm(r)
        if r_norm > 2.*a:
          # Constants for far RPY tensor, taken from OverdampedIB paper.
          C1 = 3.*a/(4.*r_norm) + (a**3)/(2.*r_norm**3)
          C2 = 3.*a/(4.*r_norm) - (3.*a**3)/(2.*r_norm**3)
        elif r_norm <= 2.*a:
          # This is for the close interaction, 
          #  Call C3 -> C1 and C4 -> C2
          C1 = 1 - 9.*r_norm/(32.*a)
          C2 = 3*r_norm/(32.*a)
        mobility[(j*3):(j*3 + 3), (k*3):(k*3 + 3)] = (1./(6.*np.pi*eta*a)*(
            C1*np.identity(3) + C2*np.outer(r, r)/(r_norm**2)))
      elif j == k:
        # j == k, diagonal block.
        mobility[(j*3):(j*3 + 3), (k*3):(k*3 + 3)] = ((1./(6.*np.pi*eta*a))*
                                                      np.identity(3))
  return mobility
  

def calculate_rot_matrix(r_vectors):
  ''' Calculate R, 3N by 3 matrix of cross products for r_i. '''
  
  # Create the 3N x 3 matrix.  Each 3x3 block is the matrix for a cross
  # product with one of the r_vectors.  Cross is relative to (0, 0, H) the location
  # of the fixed vertex.
  
  # Adjust so we take the cross relative to (0, 0, H)
  rot_matrix = None
  for k in range(len(r_vectors)):
    r_vectors[k] = r_vectors[k] - np.array([0., 0., H])

    # Current r cross x matrix block.
    block = np.array(
        [[0.0, r_vectors[k][2], -1.*r_vectors[k][1]],
        [-1.*r_vectors[k][2], 0.0, r_vectors[k][0]],
        [r_vectors[k][1], -1.*r_vectors[k][0], 0.0]])

    if rot_matrix is None:
      rot_matrix = block
    else:
      rot_matrix = np.concatenate([rot_matrix, block], axis=0)

  return rot_matrix


def calculate_rot_matrix_cm(r_vectors):
  ''' Calculate R, 3N by 3 matrix of cross products for r_i relative to 0, 0, 0 '''
  # Create the 3N x 3 matrix.  Each 3x3 block is the matrix for a cross
  # product with one of the r_vectors.  Cross is relative to (0, 0, 0)   
  rot_matrix = None
  for k in range(len(r_vectors)):
    # Current r cross x matrix block.
    block = np.array(
        [[0.0, r_vectors[k][2], -1.*r_vectors[k][1]],
        [-1.*r_vectors[k][2], 0.0, r_vectors[k][0]],
        [r_vectors[k][1], -1.*r_vectors[k][0], 0.0]])

    if rot_matrix is None:
      rot_matrix = block
    else:
      rot_matrix = np.concatenate([rot_matrix, block], axis=0)

  return rot_matrix


def get_r_vectors(quaternion):
  ''' Calculate r_i from a given quaternion. 
  The initial configuration is hard coded here but can be changed by
  considering an initial quaternion not equal to the identity rotation.
  initial configuration (top down view, the top vertex is fixed at the origin):

                         O r_1 = (0, 2/sqrt(3), -(2 sqrt(2))/3)
                        / \
                       /   \
                      /     \
                     /   O(0, 0, H)
                    /          \
                   /            \
               -> O--------------O  r_3 = (1, -1/sqrt(3),-(2 sqrt(2))/3)
             /
           r_2 = (-1, -1/sqrt(3),-(2 sqrt(2))/3)

  Each side of the tetrahedron has length 2.
  '''
  initial_r1 = np.array([0., 2./np.sqrt(3.), -2.*np.sqrt(2.)/np.sqrt(3.)])
  initial_r2 = np.array([-1., -1./np.sqrt(3.), -2.*np.sqrt(2.)/np.sqrt(3.)])
  initial_r3 = np.array([1., -1./np.sqrt(3.), -2.*np.sqrt(2.)/np.sqrt(3.)])
  
  rotation_matrix = quaternion.rotation_matrix()

  r1 = np.dot(rotation_matrix, initial_r1) + np.array([0., 0., H])
  r2 = np.dot(rotation_matrix, initial_r2) + np.array([0., 0., H])
  r3 = np.dot(rotation_matrix, initial_r3) + np.array([0., 0., H])
  
  return [r1, r2, r3]

  
def gravity_torque_calculator(orientation):
  ''' 
  Calculate torque based on orientation, given as a length
  1 list of quaternions (1 quaternion).  This assumes the masses
  of particles 1, 2, and 3 are M1, M2, and M3 respectively.
  '''
  r_vectors = get_r_vectors(orientation[0])
  R = calculate_rot_matrix(r_vectors)
  # Gravity
  g = np.array([0., 0., -1.*M1, 0., 0., -1.*M2, 0., 0., -1.*M3])
  return np.dot(R.T, g)


def zero_torque_calculator(orientation):
  ''' Return 0 torque. '''
  # Gravity
  return np.array([0., 0., 0.])


def generate_equilibrium_sample():
  ''' 
  Generate a sample according to the equilibrium distribution, exp(-\beta U(heights)).
  Do this by generating a uniform quaternion, then accept/rejecting with probability
  exp(-U(heights))'''
  max_gibbs_term = 0.
  while True:
    # First generate a uniform quaternion on the 4-sphere.
    theta = np.random.normal(0., 1., 4)
    theta = Quaternion(theta/np.linalg.norm(theta))
  
    r_vectors = get_r_vectors(theta)
    U = M1*r_vectors[0][2] + M2*r_vectors[1][2] + M3*r_vectors[2][2]
    # Roughly the smallest height.
    smallest_height = H - 1.8
    normalization_constant = np.exp(-1.*smallest_height*(M1 + M2 + M3))
    # For now, we set the normalization to 1e-2 for masses:
    #       M1 = 0.1, M2 = 0.2, M3 = 0.3
    gibbs_term = np.exp(-1.*U)
    if gibbs_term > max_gibbs_term:
      max_gibbs_term = gibbs_term
    accept_prob = np.exp(-1.*(U))/normalization_constant
    if accept_prob > 1:
      print "Warning: acceptance probability > 1."
      print "accept_prob = ", accept_prob
    if np.random.uniform() < accept_prob:
      return theta

def bin_particle_heights(orientation, bin_width, height_histogram):
  ''' 
  Given a quaternion orientation, bin the heights of the three particles.
  '''
  r_vectors = get_r_vectors(orientation)
  for k in range(3):
    # Bin each particle height.
    idx = int(math.floor((r_vectors[k][2] - H)/bin_width)) + len(height_histogram[k])/2
    height_histogram[k][idx] += 1

def calc_rotational_msd(integrator, scheme, dt, n_steps, initial_orientation):
  ''' 
  Calculate Error in rotational MSD at initial_orientation given an
  integrator and number of steps. Return the error between this MSD and
  the theoretical msd as the 2 Norm of the matrix difference.
  '''
  #TODO: Remove this, it is no longer used.
  # TODO: Change this to accept an initial orientation.
  msd = np.array([np.zeros(3) for _ in range(3)])
  for k in range(n_steps):
    integrator.orientation = initial_orientation
    if scheme == 'EM':
      integrator.additive_em_time_step(dt)
    elif scheme == 'RFD':
      integrator.rfd_time_step(dt)
    elif scheme == 'FIXMAN':
      integrator.fixman_time_step(dt)
    else:
      raise Exception('Scheme must be FIXMAN, RFD, or EM')

    u_hat = np.zeros(3)
    rot_matrix = integrator.orientation[0].rotation_matrix()
    original_rot_matrix = initial_orientation[0].rotation_matrix()
    for i in range(3):
      e = np.zeros(3)
      e[i] = 1.
      u_hat += 0.5*np.cross(np.inner(original_rot_matrix, e),
                            np.inner(rot_matrix, e))
    msd += np.outer(u_hat, u_hat)
  msd = msd/float(n_steps)/dt

  msd_theory = 2.*integrator.kT*tetrahedron_mobility(
    initial_orientation)

  return np.linalg.norm(msd_theory - msd)


if __name__ == "__main__":
  # Get command line arguments.
  parser = argparse.ArgumentParser(description='Run Simulation of fixed '
                                   'tetrahedron with Fixman, EM, and RFD '
                                   'schemes, and bin the resulting '
                                   'height distribution.')
  parser.add_argument('-dt', dest='dt', type=float,
                      help='Timestep to use for runs.')
  parser.add_argument('-N', dest='n_steps', type=int,
                      help='Number of steps to take for runs.')
  parser.add_argument('--data-name', dest='data_name', type=str,
                      default='',
                      help='Optional name added to the end of the '
                      'data file.  Useful for multiple runs '
                      '(--data_name=run-1).')
  parser.add_argument('--profile', dest='profile', type=bool, default=False,
                      help='True or False: Do we profile this run or not.')
  

  args = parser.parse_args()
  if args.profile:
    pr = cProfile.Profile()
    pr.enable()

  # Script to run the various integrators on the quaternion.
  initial_orientation = [Quaternion([1., 0., 0., 0.])]
  fixman_integrator = QuaternionIntegrator(tetrahedron_mobility,
                                           initial_orientation, 
                                           gravity_torque_calculator)

  rfd_integrator = QuaternionIntegrator(tetrahedron_mobility, 
                                        initial_orientation, 
                                        gravity_torque_calculator)

  em_integrator = QuaternionIntegrator(tetrahedron_mobility, 
                                       initial_orientation, 
                                       gravity_torque_calculator)
  # Get command line parameters
  dt = args.dt #float(sys.argv[1])
  n_steps = args.n_steps #int(sys.argv[2])
  print_increment = max(int(n_steps/20.), 1)

  # Set up logging.
  # Make directory for logs if it doesn't exist.
  if not os.path.isdir(os.path.join(os.getcwd(), 'logs')):
    os.mkdir(os.path.join(os.getcwd(), 'logs'))

  log_filename = './logs/tetrahedron-dt-%d-N-%d-%s.log' % (
    dt, n_steps, args.data_name)
  progress_logger = logging.getLogger('progress_logger')
  progress_logger.setLevel(logging.INFO)
  # Add the log message handler to the logger
  logging.basicConfig(filename=log_filename,
                      level=logging.INFO,
                      filemode='w')

  # For now hard code bin width.  Number of bins is equal to
  # 4 over bin_width, since the particle can be in a -2, +2 range around
  # the fixed vertex.
  bin_width = 1./5.
  fixman_heights = np.array([np.zeros(int(4./bin_width)) for _ in range(3)])
  rfd_heights = np.array([np.zeros(int(4./bin_width)) for _ in range(3)])
  em_heights = np.array([np.zeros(int(4./bin_width)) for _ in range(3)])
  equilibrium_heights = np.array([np.zeros(int(4./bin_width)) for _ in range(3)])

  start_time = time.time()
  for k in range(n_steps):
    # Fixman step and bin result.
    fixman_integrator.fixman_time_step(dt)
    bin_particle_heights(fixman_integrator.orientation[0], 
                         bin_width, 
                         fixman_heights)
    # RFD step and bin result.
    rfd_integrator.rfd_time_step(dt)
    bin_particle_heights(rfd_integrator.orientation[0],
                         bin_width, 
                         rfd_heights)    
    # EM step and bin result.
    em_integrator.additive_em_time_step(dt)
    bin_particle_heights(em_integrator.orientation[0],
                         bin_width, 
                         em_heights)
    # Bin equilibrium sample.
    bin_particle_heights(generate_equilibrium_sample(), 
                         bin_width, 
                         equilibrium_heights)
    
    if k % print_increment == 0:
      elapsed_time = time.time() - start_time
      if elapsed_time < 60.:
        progress_logger.info('At step: %d. Time Taken: %.2f Seconds' % 
                             (k, float(elapsed_time)))
        if k > 0:
          progress_logger.info('Estimated Total time required: %.2f Seconds.' %
                               (elapsed_time*float(n_steps)/float(k)))
      else:
        progress_logger.info('At step: %d. Time Taken: %.2f Minutes.' %
                             (k, (float(elapsed_time)/60.)))
        if k > 0:
          progress_logger.info('Estimated Total time required: %.2f Minutes.' %
                               (elapsed_time*float(n_steps)/float(k)/60.))
      sys.stdout.flush()

  elapsed_time = time.time() - start_time
  if elapsed_time > 60:
    progress_logger.info('Finished timestepping. Total Time: %.2f minutes.' % 
                         float(elapsed_time)/60.)
  else:
    progress_logger.info('Finished timestepping. Total Time: %.2f seconds.' % 
                         float(elapsed_time))

  heights = [fixman_heights/(n_steps*bin_width),
             rfd_heights/(n_steps*bin_width),
             em_heights/(n_steps*bin_width),
             equilibrium_heights/(n_steps*bin_width)]

  # Make directory for data if it doesn't exist.
  if not os.path.isdir(os.path.join(os.getcwd(), 'data')):
    os.mkdir(os.path.join(os.getcwd(), 'data'))

  # Optional name for data provided    
  if len(args.data_name) > 0:
    data_name = './data/tetrahedron-dt-%g-N-%d-%s.pkl' % (dt, n_steps, args.data_name)
  else:
    data_name = './data/tetrahedron-dt-%g-N-%d.pkl' % (dt, n_steps)

  height_data = dict()
  height_data['params'] = {'A': A, 'ETA': ETA, 'H': H, 'M1': M1, 'M2': M2, 
                           'M3': M3}
  height_data['heights'] = heights
  height_data['names'] = ['Fixman', 'RFD', 'EM', 'Gibbs-Boltzmann']
  height_data['buckets'] = H + np.linspace(-2., 2., len(heights[0][0]))

  with open(data_name, 'wb') as f:
    cPickle.dump(height_data, f)
  
  if args.profile:
    pr.disable()
    s = StringIO.StringIO()
    sortby = 'cumulative'
    ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
    ps.print_stats()
    print s.getvalue()


