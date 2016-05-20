''' Fluid Mobilities near a wall, from Swan and Brady's paper.'''
import numpy as np
import sys
sys.path.append('..')
import time
import imp

# Try to import the mobility boost implementation
try:
  import mobility_ext as me
except ImportError:
  pass
# If pycuda is installed import mobility_pycuda
try: 
  imp.find_module('pycuda')
  found_pycuda = True
except ImportError:
  found_pycyda = False
if found_pycuda:
  import mobility_pycuda

ETA = 1.0 # Viscosity

def image_singular_stokeslet(r_vectors, a):
  ''' Calculate the image system for the singular stokeslet (M above).'''
  fluid_mobility = np.array([
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
            fluid_mobility[j*3 + l][k*3 + m] = (
              ((l == m)*1./r_norm + r_particles[l]*r_particles[m]/(r_norm**3) -
               ((l == m)*1./r_ref_norm + r_reflect[l]*r_reflect[m]/(r_ref_norm**3)))/
              (8.*np.pi))
        # Add doublet and dipole contribution.
        fluid_mobility[(j*3):(j*3 + 3), (k*3):(k*3 + 3)] += (
          doublet_and_dipole(r_reflect, wall_dist))
        
      else:
        # j == k
        fluid_mobility[(j*3):(j*3 + 3), (k*3):(k*3 + 3)] = 1./(6*np.pi*ETA*a)*np.identity(3)
  return fluid_mobility

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


def boosted_single_wall_fluid_mobility(r_vectors, eta, a):
  ''' 
  Same as single wall fluid mobility, but boosted into C++ for 
  a speedup. Must compile mobility_ext.cc before this will work 
  (use Makefile).
  ''' 
  num_particles = r_vectors.size / 3
  fluid_mobility = np.zeros( (num_particles*3, num_particles*3) )
  me.RPY_single_wall_fluid_mobility(np.reshape(r_vectors, (num_particles, 3)), eta, a, num_particles, fluid_mobility)
  return fluid_mobility

def boosted_infinite_fluid_mobility(r_vectors, eta, a):
  ''' 
  Same as rotne_prager_tensor, but boosted into C++ for 
  a speedup. Must compile mobility_ext.cc before this will work 
  (use Makefile).
  '''
  num_particles = len(r_vectors)
  fluid_mobility = np.array([np.zeros(3*num_particles) for _ in range(3*num_particles)])
  me.RPY_infinite_fluid_mobility(r_vectors, eta, a, num_particles, fluid_mobility)
  return fluid_mobility

   
def boosted_mobility_vector_product(r_vectors, vector, eta, a):
  ''' 
  Compute a mobility * vector product boosted in C++ for a
  speedup. It includes wall corrections.
  Must compile mobility_ext.cc before this will work 
  (use Makefile).
  '''
  ## THE USE OF VECTOR_RES AS THE RESULT OF THE MATRIX VECTOR PRODUCT IS 
  ## TEMPORARY: I NEED TO FIGURE OUT HOW TO CONVERT A DOUBLE TO A NUMPY ARRAY
  ## WITH BOOST
  num_particles = r_vectors.size / 3
  vector_res = np.zeros(r_vectors.size)
  r_vec_for_mob = np.reshape(r_vectors, (r_vectors.size / 3, 3))  
  me.mobility_vector_product(r_vec_for_mob, eta, a, num_particles, vector, vector_res)
  return vector_res

def single_wall_mobility_trans_times_force_pycuda(r_vectors, force, eta, a):
  ''' 
  Returns the product of the mobility at the blob level by the force 
  on the blobs.
  Mobility for particles near a wall.  This uses the expression from
  the Swan and Brady paper for a finite size particle, as opposed to the 
  Blake paper point particle result. 
  
  This function makes use of pycuda.
  '''
  velocities = mobility_pycuda.single_wall_mobility_trans_times_force_pycuda(r_vectors, force, eta, a) 
  return velocities

def single_wall_mobility_rot_times_force_pycuda(r_vectors, force, eta, a):
  ''' 
  Returns the product of the mobility at the blob level to the force 
  on the blobs.
  Mobility for particles near a wall.  This uses the expression from
  the Swan and Brady paper for a finite size particle, as opposed to the 
  Blake paper point particle result. 
  
  This function makes use of pycuda.
  '''

  rot = mobility_pycuda.single_wall_mobility_rot_times_force_pycuda(r_vectors, force, eta, a)
  
  return rot

def no_wall_mobility_rot_times_force_pycuda(r_vectors, force, eta, a):
  ''' 
  Returns the product of the mobility at the blob level to the force 
  on the blobs.
  Mobility for particles near a wall.  This uses the expression from
  the Swan and Brady paper for a finite size particle, as opposed to the 
  Blake paper point particle result. 
  
  This function makes use of pycuda.
  '''

  rot = mobility_pycuda.no_wall_mobility_rot_times_force_pycuda(r_vectors, force, eta, a)
  
  return rot

def single_wall_mobility_rot_times_torque_pycuda(r_vectors, torque, eta, a):
  ''' 
  Returns the product of the mobility at the blob level to the force 
  on the blobs.
  Mobility for particles near a wall.  This uses the expression from
  the Swan and Brady paper for a finite size particle, as opposed to the 
  Blake paper point particle result. 
  
  This function makes use of pycuda.
  '''

  rot = mobility_pycuda.single_wall_mobility_rot_times_torque_pycuda(r_vectors, torque, eta, a)
  
  return rot

def no_wall_mobility_rot_times_torque_pycuda(r_vectors, torque, eta, a):
  ''' 
  Returns the product of the mobility at the blob level to the force 
  on the blobs.
  Mobility for particles near a wall.  This uses the expression from
  the Swan and Brady paper for a finite size particle, as opposed to the 
  Blake paper point particle result. 
  
  This function makes use of pycuda.
  '''

  rot = mobility_pycuda.no_wall_mobility_rot_times_torque_pycuda(r_vectors, torque, eta, a)
  
  return rot

def single_wall_mobility_trans_times_force_torque_pycuda(r_vectors, force, torque, eta, a):
  ''' 
  Returns the product of the mobility at the blob level to the force 
  on the blobs.
  Mobility for particles near a wall.  This uses the expression from
  the Swan and Brady paper for a finite size particle, as opposed to the 
  Blake paper point particle result. 
  
  This function makes use of pycuda.
  '''

  velocities = mobility_pycuda.single_wall_mobility_trans_times_force_torque_pycuda(r_vectors, force, torque, eta, a)
  
  return velocities


def no_wall_mobility_trans_times_force_torque_pycuda(r_vectors, force, torque, eta, a):
  ''' 
  Returns the product of the mobility at the blob level to the force 
  on the blobs.
  Mobility for particles near a wall.  This uses the expression from
  the Swan and Brady paper for a finite size particle, as opposed to the 
  Blake paper point particle result. 
  
  This function makes use of pycuda.
  '''

  velocities = mobility_pycuda.no_wall_mobility_trans_times_force_torque_pycuda(r_vectors, force, torque, eta, a)
  
  return velocities

def single_wall_mobility_trans_times_force_pycuda_single(r_vectors, force, eta, a):
  ''' 
  Returns the product of the mobility at the blob level to the force 
  on the blobs.
  Mobility for particles near a wall.  This uses the expression from
  the Swan and Brady paper for a finite size particle, as opposed to the 
  Blake paper point particle result. 
  
  This function makes use of pycuda.
  '''
  velocities = mobility_pycuda.single_wall_mobility_trans_times_force_pycuda_single(r_vectors, force, eta, a)
  
  return velocities



  
def boosted_mobility_vector_product_one_particle(r_vectors, eta, a, vector, \
                                                 index_particle):
  ''' 
  Compute a mobility * vector product for only one particle. Return the 
  velocity of of the desired particle. It includes wall corrections.
  Boosted in C++ for a speedup. Must compile mobility_ext.cc before this 
  will work (use Makefile).
  '''
  num_particles = len(r_vectors)
  ## THE USE OF VECTOR_RES AS THE RESULT OF THE MATRIX VECTOR PRODUCT IS 
  ## TEMPORARY: I NEED TO FIGURE OUT HOW TO CONVERT A DOUBLE TO A NUMPY ARRAY
  ## WITH BOOST
  vector_res = np.zeros(3)
  me.mobility_vector_product_one_particle(r_vectors, eta, a, \
					  num_particles, vector, \
					  vector_res, index_particle)
  return vector_res
  


def single_wall_mobility_times_force_pycuda(r_vectors, force, eta, a):
  ''' 
  Returns the product of the mobility at the blob level to the force 
  on the blobs.
  Mobility for particles near a wall.  This uses the expression from
  the Swan and Brady paper for a finite size particle, as opposed to the 
  Blake paper point particle result. 
  
  This function makes use of pycuda.
  '''

  velocities = mobility_pycuda.single_wall_mobility_times_force_pycuda(r_vectors, force, eta, a)
  
  return velocities


def single_wall_fluid_mobility(r_vectors, eta, a):
  ''' Mobility for particles near a wall.  This uses the expression from
  the Swan and Brady paper for a finite size particle, as opposed to the 
  Blake paper point particle result. '''
  num_particles = len(r_vectors)
  # We add the corrections from the appendix of the paper to the unbounded mobility.
  fluid_mobility = rotne_prager_tensor(r_vectors, eta, a)
  for j in range(num_particles):
    for k in range(j+1, num_particles):
      # Here notation is based on appendix C of the Swan and Brady paper:
      # 'Simulation of hydrodynamically interacting particles near a no-slip
      # boundary.'
      h = r_vectors[k][2]
      R = (r_vectors[j] - (r_vectors[k] - 2.*np.array([0., 0., h])))/a
      R_norm = np.linalg.norm(R)
      e = R/R_norm
      e_3 = np.array([0., 0., e[2]])
      h_hat = h/(a*R[2])
      # Taken from Appendix C expression for M_UF
      fluid_mobility[(j*3):(j*3 + 3), (k*3):(k*3 + 3)] += (1./(6.*np.pi*eta*a))*(
        -0.25*(3.*(1. - 6.*h_hat*(1. - h_hat)*e[2]**2)/R_norm
               - 6.*(1. - 5.*e[2]**2)/(R_norm**3)
               + 10.*(1. - 7.*e[2]**2)/(R_norm**5))*np.outer(e, e)
         - (0.25*(3.*(1. + 2.*h_hat*(1. - h_hat)*e[2]**2)/R_norm
                  + 2.*(1. - 3.*e[2]**2)/(R_norm**3)
                  - 2.*(1. - 5.*e[2]**2)/(R_norm**5)))*np.identity(3)
         + 0.5*(3.*h_hat*(1. - 6.*(1. - h_hat)*e[2]**2)/R_norm
                - 6.*(1. - 5.*e[2]**2)/(R_norm**3)
                + 10.*(2. - 7.*e[2]**2)/(R_norm**5))*np.outer(e, e_3)
         + 0.5*(3.*h_hat/R_norm - 10./(R_norm**5))*np.outer(e_3, e)
         - (3.*(h_hat**2)*(e[2]**2)/R_norm 
            + 3.*(e[2]**2)/(R_norm**3)
            + (2. - 15.*e[2]**2)/(R_norm**5))*np.outer(e_3, e_3)/(e[2]**2))
      
      fluid_mobility[(k*3):(k*3 + 3), (j*3):(j*3 + 3)] = (
        fluid_mobility[(j*3):(j*3 + 3), (k*3):(k*3 + 3)].T)

  for j in range(len(r_vectors)):
    # Diagonal blocks, self mobility.
    h = r_vectors[j][2]/a
    for l in range(3):
      fluid_mobility[j*3 + l][j*3 + l] += (1./(6.*np.pi*eta*a))*(
        (l != 2)*(-1./16.)*(9./h - 2./(h**3) + 1./(h**5))
        + (l == 2)*(-1./8.)*(9./h - 4./(h**3) + 1./(h**5)))
  return fluid_mobility


def rotne_prager_tensor(r_vectors, eta, a):
  ''' Calculate free rotne prager tensor for particles at locations given by
  r_vectors (list of 3 dimensional locations) of radius a.'''
  num_particles = len(r_vectors)
  fluid_mobility = np.array([np.zeros(3*num_particles) 
                             for _ in range(3*num_particles)])
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
          # Call C3 -> C1 and C4 -> C2
          C1 = 1 - 9.*r_norm/(32.*a)
          C2 = 3*r_norm/(32.*a)
        fluid_mobility[(j*3):(j*3 + 3), (k*3):(k*3 + 3)] = (1./(6.*np.pi*eta*a)*(
          C1*np.identity(3) + C2*np.outer(r, r)/(r_norm**2)))

      elif j == k:
        # j == k, diagonal block.
        fluid_mobility[(j*3):(j*3 + 3), (k*3):(k*3 + 3)] = ((1./(6.*np.pi*eta*a))*
                                                      np.identity(3))
  return fluid_mobility


def single_wall_fluid_mobility_product(r_vectors, vector, eta, a):
  ''' Product (Mobility * vector). Mobility for particles near a wall.  
  This uses the expression from the Swan and Brady paper for a finite 
  size particle, as opposed to the Blake paper point particle result. 
  '''
  r = np.reshape(r_vectors, (r_vectors.size / 3, 3))
  mobility = single_wall_fluid_mobility(r, eta, a)
  return np.dot(mobility, vector)


def single_wall_self_mobility_with_rotation(location, eta, a):
  ''' 
  Self mobility for a single sphere of radius a with translation rotation
  coupling.  Returns the 6x6 matrix taking force and torque to 
  velocity and angular velocity.
  This expression is taken from Swan and Brady's paper:
  '''
  h = location[2]/a
  fluid_mobility = (1./(6.*np.pi*eta*a))*np.identity(3)
  zero_matrix = np.zeros([3, 3])
  fluid_mobility = np.concatenate([fluid_mobility, zero_matrix])
  zero_matrix = np.zeros([6, 3])
  fluid_mobility = np.concatenate([fluid_mobility, zero_matrix], axis=1)
  # First the translation-translation block.
  for l in range(3):
    for m in range(3):
      fluid_mobility[l][m] += (1./(6.*np.pi*eta*a))*(
        (l == m)*(l != 2)*(-1./16.)*(9./h - 2./(h**3) + 1./(h**5))
        + (l == m)*(l == 2)*(-1./8.)*(9./h - 4./(h**3) + 1./(h**5)))
  # Translation-Rotation blocks.
  for l in range(3):
    for m in range(3):
      fluid_mobility[3 + l][m] += (1./(6.*np.pi*eta*a*a))*((3./32.)*
                                     (h**(-4))*epsilon_tensor(2, l, m))
      fluid_mobility[m][3 + l] += fluid_mobility[3 + l][m]
  
  # Rotation-Rotation block.
  for l in range(3):
    for m in range(3):
      fluid_mobility[3 + l][3 + m] += (
        (1./(8.*np.pi*eta*(a**3)))*(l == m) - ((1./(6.*np.pi*eta*(a**3)))*(
                                      (15./64.)*(h**(-3))*(l == m)*(l != 2)
                                      + (3./32.)*(h**(-3))*(m == 2)*(l == 2))))
  return fluid_mobility

  
def epsilon_tensor(i, j, k):
  ''' 
  Epsilon tensor (cross product).  Only works for arguments
  between 0 and 2.
  '''
  if j == ((i + 1) % 3) and k == ((j+1) % 3):
    return 1.
  elif i == ((j + 1) % 3) and j == ((k + 1) % 3):
    return -1.
  else:
    return 0.
  
  
if __name__ == '__main__':
  # Example of using single wall mobility
  r_vectors = np.array([[5., 0., 3.],
                       [2., 0., 2.]])

  a = 0.2
  eta = 1.0
  
  start = time.time()
  for k in range(10000):
    mobility = single_wall_fluid_mobility(r_vectors, eta, a)
  elapsed = time.time() - start
  print "mobility is ", mobility
  print 'elapsed python is ', elapsed
  
  start = time.time()
  for k in range(10000):
    mobility = me.single_wall_fluid_mobility(r_vectors, eta, a)
  elapsed = time.time() - start
  print "mobility is ", mobility
  print 'elapsed cython is ', elapsed


