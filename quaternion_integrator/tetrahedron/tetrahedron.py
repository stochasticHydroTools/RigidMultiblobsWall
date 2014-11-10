''' 
Script to test a tetrahedron near a wall.  The wall is at z = -h, and
the tetrahedron's "top" vertex is fixed at (0, 0, 0).
'''
import sys
sys.path.append('..')
sys.path.append('../..')
import numpy as np
from matplotlib import pyplot
from quaternion import Quaternion
from quaternion_integrator import QuaternionIntegrator
import uniform_analyzer as ua
import cProfile, pstats, StringIO
# import tetrahedron_ext
#  Parameters. TODO: perhaps there's a better way to do this.  Input file?
PROFILE = False  # Do we profile this run?

ETA = 1.0   # Fluid viscosity.
A = 0.03     # Particle Radius.
H = 10.     # Distance to wall.

# Masses of particles.
M1 = 1.0
M2 = 2.0
M3 = 3.0

def identity_mobility(position):
  ''' Simple identity mobility for testing. '''
  return np.identity(3)


def tetrahedron_mobility(position):
  '''
  Calculate the mobility, torque -> angular velocity, at position 
  In this case, position is length 1, as there is just 1 quaternion.
  The mobility is equal to R M^-1 R^t where R is 3N x 3 (9 x 3)
  Rx = r cross x
  r is the distance from the fixed vertex of the tetrahedron to
  each other vertex (a length 3N vector).
  M (3N x 3N) is the singular image stokeslet for a point force near a wall, but
  we've replaced the diagonal piece by 1/(6 pi eta a).
  '''
  r_vectors = get_r_vectors(position[0])
  mobility = image_singular_stokeslet(r_vectors)
  rotation_matrix = calculate_rot_matrix(r_vectors)
  total_mobility = np.dot(rotation_matrix.T,
                          np.dot(np.linalg.inv(mobility),
                                 rotation_matrix))
  return total_mobility


def image_singular_stokeslet(r_vectors):
  ''' Calculate the image system for the singular stokeslet (M above).'''
  mobility = np.array([np.zeros(9) for _ in range(9)])
  # Loop through particle interactions
  for j in range(3):
    for k in range(3):
      if j != k:  #  do particle interaction
        r_particles = r_vectors[j] - r_vectors[k]
        r_norm = np.linalg.norm(r_particles)
        r_reflect = r_vectors[j] - (r_vectors[k] - 2.*np.array([0., 0., H])
                                       - 2.*r_vectors[k][2])
        r_ref_norm = np.linalg.norm(r_reflect)
        # Loop through components.
        for l in range(3):
          for m in range(3):
            # Two stokeslets, one with negative force at image.
            mobility[j*3 + l][k*3 + m] = (
              (l == m)*1./r_norm + r_particles[l]*r_particles[m]/(r_norm**3) -
              (l == m)*1./r_ref_norm + r_reflect[l]*r_reflect[m]/(r_ref_norm**3))
        # Add Doublet.
        mobility[(j*3):(j*3 + 3), (k*3):(k*3 + 3)] += 2.*H*stokes_doublet(r_reflect)
        # Add Potential Dipole.
        mobility[(j*3):(j*3 + 3), (k*3):(k*3 + 3)] -= H*H*potential_dipole(r_reflect)
      else:
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

  
def calculate_rot_matrix(r_vectors):
  ''' Calculate R, 9 by 3 matrix of cross products for r_i. '''
  
  # Create the 9 x 3 matrix.  Each 3x3 block is the matrix for a cross
  # product with one of the r_vectors.
  return np.array([
      #Block 1, cross r_1 
      [0.0, r_vectors[0][2], -1.*r_vectors[0][1]],
      [-1.*r_vectors[0][2], 0.0, r_vectors[0][0]],
      [r_vectors[0][1], -1.*r_vectors[0][0],0.0],
      # Block 2, cross r_2
      [0.0, r_vectors[1][2], -1.*r_vectors[1][1]],
      [-1.*r_vectors[1][2], 0.0, r_vectors[1][0]],
      [r_vectors[1][1], -1.*r_vectors[1][0],0.0],
      # Block 3, cross r_3
      [0.0, r_vectors[2][2], -1.*r_vectors[2][1]],
      [-1.*r_vectors[2][2], 0.0, r_vectors[2][0]],
      [r_vectors[2][1], -1.*r_vectors[2][0],0.0],
      ])


def get_r_vectors(quaternion):
  ''' Calculate r_i from a given quaternion. 
  The initial configuration is hard coded here but can be changed by
  considering an initial quaternion not equal to the identity rotation.
  initial configuration (top down view, the top vertex is fixed at the origin):

                         O r_1 = (0, 2/sqrt(3), -(2 sqrt(2))/3)
                        / \
                       /   \
                      /     \
                     /   O(0, 0, 0)
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

  r1 = np.dot(rotation_matrix, initial_r1)
  r2 = np.dot(rotation_matrix, initial_r2)
  r3 = np.dot(rotation_matrix, initial_r3)
  
  return [r1, r2, r3]

  
def gravity_torque_calculator(position):
  ''' 
  Calculate torque based on position, given as a length
  1 list of quaternions (1 quaternion).  This assumes the masses
  of particles 1, 2, and 3 are M1, M2, and M3 respectively.
  '''
  r_vectors = get_r_vectors(position[0])
  R = calculate_rot_matrix(r_vectors)
  
  # Gravity
  g = np.array([0., 0., -1.*M1, 0., 0., -1.*M2, 0., 0., -1.*M3])
  return np.dot(R.T, g)


def zero_torque_calculator(position):
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
    # 22500 is roughly the value of exp(-beta U) for the most likely config.
    # when M1 = 1, M2 = 2, M3 = 3.
    # 137 is roughly the value when M1 = M2 = M3 = 1.
    gibbs_term = np.exp(-1.*(U))
    if gibbs_term > max_gibbs_term:
      max_gibbs_term = gibbs_term
    accept_prob = np.exp(-1.*(U))/22500.
    if accept_prob > 1:
      print "Warning: acceptance probability > 1."
      print "accept_prob = ", accept_prob
    if np.random.uniform() < accept_prob:
      return theta
    

def distribution_height_particle(particle, path, equilibrium_samples):
  ''' 
  Given the path of the quaternion, make a historgram of the 
  height of particle <particle> and compare to equilibrium. 
  '''
  pyplot.figure()

  hist_bins = np.linspace(-1.8, 1.8, 40)
  heights = []
  for pos in path:
    # TODO: do this a faster way perhaps with a special function.
    r_vectors = get_r_vectors(pos[0])
    heights.append(r_vectors[particle][2])

  height_hist = np.histogram(heights, density=True, bins=hist_bins)
  buckets = (height_hist[1][:-1] + height_hist[1][1:])/2.
  pyplot.plot(buckets, height_hist[0], 'b-',  label='Fixman')


  heights = []
  for sample in equilibrium_samples:
    # TODO: do this a faster way perhaps with a special function, since we only need
    #  the z coordinate.
    r_vectors = get_r_vectors(sample)
    heights.append(r_vectors[particle][2])

  height_hist = np.histogram(heights, density=True, bins=hist_bins)
  buckets = (height_hist[1][:-1] + height_hist[1][1:])/2.
  pyplot.plot(buckets, height_hist[0], 'k--',  label='Gibbs-Boltzmann')
  pyplot.legend(loc='best', prop={'size': 9})
  pyplot.title('Location of particle %d' % particle)
  pyplot.ylabel('Probability Density')
  pyplot.xlabel('Height')
  pyplot.savefig('./plots/Height%d_Distribution.pdf' % particle)


if __name__ == "__main__":
  if PROFILE:
    pr = cProfile.Profile()
    pr.enable()

  # Script to run the Fixman integrator on the quaternion.
  initial_position = [Quaternion([1., 0., 0., 0.])]
  fixman_integrator = QuaternionIntegrator(tetrahedron_mobility, 
                                           initial_position, 
                                           gravity_torque_calculator)
  # Get command line parameters
  dt = float(sys.argv[1])
  n_steps = int(sys.argv[2])

  equilibrium_samples = []  
  for k in range(n_steps):
    fixman_integrator.fixman_time_step(dt)
    equilibrium_samples.append(generate_equilibrium_sample())

  distribution_height_particle(0, fixman_integrator.path, equilibrium_samples)
  distribution_height_particle(1, fixman_integrator.path, equilibrium_samples)
  distribution_height_particle(2, fixman_integrator.path, equilibrium_samples)
  
  if PROFILE:
    pr.disable()
    s = StringIO.StringIO()
    sortby = 'cumulative'
    ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
    ps.print_stats()
    print s.getvalue()


