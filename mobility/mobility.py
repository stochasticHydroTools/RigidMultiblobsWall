''' Fluid Mobilities near a wall, from Swan and Brady's paper.'''

import numpy as np
import scipy.sparse
import sys
import time
import imp

# If pycuda is installed import mobility_pycuda
try:
  imp.find_module('pycuda')
  found_pycuda = True
except ImportError:
  found_pycuda = False
if found_pycuda:
  try:
    import pycuda.autoinit
    autoinit_pycuda = True
  except:
    autoinit_pycuda = False
  if autoinit_pycuda:
    try:
      from . import mobility_pycuda
    except ImportError:
      from .mobility import mobility_pycuda

# If numba is installed import mobility_numba
try: 
  imp.find_module('numba')
  found_numba = True
except ImportError:
  found_numba = False
if found_numba:
  try:
    from . import mobility_numba
  except ImportError:
    import mobility_numba

# Try to import the mobility fmm implementation
try:
  import mobility_fmm as fmm
except ImportError:
  pass
try:
  import mobility_cpp
except ImportError:
  try:
    from . import mobility_cpp
  except ImportError:
    pass

def shift_heights(r_vectors, blob_radius, *args, **kwargs):
  '''
  Return an array with the blobs' height

  z_effective = maximum(z, blob_radius)

  This function is used to compute positive
  definite mobilites for blobs close to the wall.
  '''
  r_effective = np.copy(r_vectors)
  r_effective[r_vectors[:,2] <= blob_radius, 2] = blob_radius

  return r_effective


def damping_matrix_B(r_vectors, blob_radius, *args, **kwargs):
  '''
  Return sparse diagonal matrix with components
  B_ii = 1.0               if z_i >= blob_radius
  B_ii = z_i / blob_radius if z_i < blob_radius

  It is used to compute positive definite mobilities
  close to the wall.
  '''
  B = np.ones(r_vectors.size)
  overlap = False
  for k, r in enumerate(r_vectors):
    if r[2] < blob_radius:
      B[k*3]     = r[2] / blob_radius
      B[k*3 + 1] = r[2] / blob_radius
      B[k*3 + 2] = r[2] / blob_radius
      overlap = True
  return (scipy.sparse.dia_matrix((B, 0), shape=(B.size, B.size)), overlap)


def shift_heights_different_radius(r_vectors, blob_radius, *args, **kwargs):
  '''
  Return an array with the blobs' height

  z_effective = maximum(z, blob_radius)

  This function is used to compute positive
  definite mobilites for blobs close to the wall.
  '''
  r_effective = np.copy(r_vectors)
  for k, r in enumerate(r_effective):
    r[2] = r[2] if r[2] > blob_radius[k] else blob_radius[k]
  return r_effective


def damping_matrix_B_different_radius(r_vectors, blob_radius, *args, **kwargs):
  '''
  Return sparse diagonal matrix with components
  B_ii = 1.0               if z_i >= blob_radius
  B_ii = z_i / blob_radius if z_i < blob_radius

  It is used to compute positive definite mobilities
  close to the wall.
  '''
  B = np.ones(r_vectors.size)
  overlap = False
  for k, r in enumerate(r_vectors):
    if r[2] < blob_radius[k]:
      B[k*3]     = r[2] / blob_radius[k]
      B[k*3 + 1] = r[2] / blob_radius[k]
      B[k*3 + 2] = r[2] / blob_radius[k]
      overlap = True
  return (scipy.sparse.dia_matrix((B, 0), shape=(B.size, B.size)), overlap)


def image_singular_stokeslet(r_vectors, eta, a, *args, **kwargs):
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
        fluid_mobility[(j*3):(j*3 + 3), (k*3):(k*3 + 3)] = 1./(6*np.pi*eta*a)*np.identity(3)
  return fluid_mobility

def stokes_doublet(r, *args, **kwargs):
  ''' Calculate stokes doublet from direction, strength, and r. '''
  r_norm = np.linalg.norm(r)
  e3 = np.array([0., 0., 1.])
  doublet = (np.outer(r, e3) + np.dot(r, e3)*np.identity(3) -
             np.outer(e3, r) - 3.*np.dot(e3, r)*np.outer(r, r)/(r_norm**2))
  # Negate the first two columns for the correct forcing.
  doublet[:, 0:2] = -1.*doublet[:, 0:2]
  doublet = doublet/(8*np.pi*(r_norm**3))
  return doublet

def potential_dipole(r, *args, **kwargs):
  ''' Calculate potential dipole. '''
  r_norm = np.linalg.norm(r)
  dipole = np.identity(3) - 3.*np.outer(r, r)/(r_norm**2)
  # Negate the first two columns for the correct forcing.
  dipole[:, 0:2] = -1.*dipole[:, 0:2]
  dipole = dipole/(4.*np.pi*(r_norm**3))
  return dipole


def doublet_and_dipole(r, h, *args, **kwargs):
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


def single_wall_mobility_trans_times_force_cpp(r_vectors, force, eta, a, *args, **kwargs):
  '''
  Returns the product of the mobility at the blob level by the force
  on the blobs.
  Mobility for particles near a wall.  This uses the expression from
  the Swan and Brady paper for a finite size particle, as opposed to the
  Blake paper point particle result.

  If a component of periodic_length is larger than zero the
  space is assume to be pseudo-periodic in that direction. In that case
  the code will compute the interactions M*f between particles in
  the minimal image convection and also in the first neighbor boxes.

  For blobs overlaping the wall we use
  Compute M = B^T * M_tilde(z_effective) * B.

  This function uses mobility_cpp, a separate extention from mobility_ext that must be built.
  '''
  L = kwargs.get('periodic_length', np.array([0.0, 0.0, 0.0]))
  # Get effective height
  r_vectors_effective = shift_heights(r_vectors, a)
  # Compute damping matrix B
  B, overlap = mobility_cpp.damping_matrix_B(r_vectors, a)

  # Compute B * force
  if overlap is True:
    force = B.dot(force.flatten())
  # Compute M_tilde * B * force
  velocities = mobility_cpp.single_wall_mobility_trans_times_force(r_vectors_effective, force, eta, a, L)
  # Compute B.T * M * B * vector
  if overlap is True:
    velocities = B.dot(velocities)
  return velocities


def single_wall_mobility_trans_times_force_pycuda(r_vectors, force, eta, a, *args, **kwargs):
  '''
  Returns the product of the mobility at the blob level by the force
  on the blobs.
  Mobility for particles near a wall.  This uses the expression from
  the Swan and Brady paper for a finite size particle, as opposed to the
  Blake paper point particle result.

  If a component of periodic_length is larger than zero the
  space is assume to be pseudo-periodic in that direction. In that case
  the code will compute the interactions M*f between particles in
  the minimal image convection and also in the first neighbor boxes.

  For blobs overlaping the wall we use
  Compute M = B^T * M_tilde(z_effective) * B.

  This function uses pycuda.
  '''
  # Get effective height
  r_vectors_effective = shift_heights(r_vectors, a)
  # Compute damping matrix B
  B, overlap = damping_matrix_B(r_vectors, a, *args, **kwargs)
  # Compute B * force
  if overlap is True:
    force = B.dot(force.flatten())
  # Compute M_tilde * B * force
  velocities = mobility_pycuda.single_wall_mobility_trans_times_force_pycuda(r_vectors_effective, force, eta, a, *args, **kwargs)
  # Compute B.T * M * B * vector
  if overlap is True:
    velocities = B.dot(velocities)
  return velocities

def in_plane_mobility_trans_times_force_pycuda(r_vectors, force, eta, a, *args, **kwargs):
  '''
  Returns the product of the mobility at the blob level by the force
  on the blobs.
  Mobility for particles near a wall.  This uses the expression from
  the Swan and Brady paper for a finite size particle, as opposed to the
  Blake paper point particle result.

  If a component of periodic_length is larger than zero the
  space is assume to be pseudo-periodic in that direction. In that case
  the code will compute the interactions M*f between particles in
  the minimal image convection and also in the first neighbor boxes.

  For blobs overlaping the wall we use
  Compute M = B^T * M_tilde(z_effective) * B.

  This function makes use of pycuda.
  '''
  # Get effective height
  r_vectors_effective = shift_heights(r_vectors, a)
  # Compute damping matrix B
  B, overlap = damping_matrix_B(r_vectors, a, *args, **kwargs)
  # Compute B * force
  if overlap is True:
    force = B.dot(force)
  # Compute M_tilde * B * force
  velocities = mobility_pycuda.in_plane_mobility_trans_times_force_pycuda(r_vectors_effective, force, eta, a, *args, **kwargs)
  # Compute B.T * M * B * vector
  if overlap is True:
    velocities = B.dot(velocities)
  return velocities


def no_wall_mobility_trans_times_force_pycuda(r_vectors, force, eta, a, *args, **kwargs):
  '''
  Returns the product of the mobility at the blob level to the force
  on the blobs. Mobility for particles in an unbounded domain, it uses
  the standard RPY tensor.  
  
  This function uses pycuda.
  '''
  vel = mobility_pycuda.no_wall_mobility_trans_times_force_pycuda(r_vectors, force, eta, a, *args, **kwargs)
  return vel


def single_wall_mobility_rot_times_force_pycuda(r_vectors, force, eta, a, *args, **kwargs):
  '''
  Returns the product of the mobility at the blob level to the force
  on the blobs.
  Mobility for particles near a wall.  This uses the expression from
  the Swan and Brady paper for a finite size particle, as opposed to the
  Blake paper point particle result.

  For blobs overlaping the wall we use
  Compute M = B^T * M_tilde(z_effective) * B.
  
  This function uses pycuda.
  '''
  # Get effective height
  r_vectors_effective = shift_heights(r_vectors, a)
  # Compute damping matrix B
  B, overlap = damping_matrix_B(r_vectors, a, *args, **kwargs)
  # Compute B * force
  if overlap is True:
    force = B.dot(force.flatten())
  # Compute M_tilde * B * force
  rot = mobility_pycuda.single_wall_mobility_rot_times_force_pycuda(r_vectors_effective, force, eta, a, *args, **kwargs)
  # Compute B.T * M * B * force
  if overlap is True:
    rot = B.dot(rot)
  return rot


def no_wall_mobility_rot_times_force_pycuda(r_vectors, force, eta, a, *args, **kwargs):
  '''
  Returns the product of the mobility at the blob level to the force
  on the blobs.
  Mobility for particles near a wall.  This uses the expression from
  the Swan and Brady paper for a finite size particle, as opposed to the 
  Blake paper point particle result. 
  
  This function uses pycuda.
  '''
  rot = mobility_pycuda.no_wall_mobility_rot_times_force_pycuda(r_vectors, force, eta, a, *args, **kwargs)
  return rot


def single_wall_mobility_rot_times_torque_pycuda(r_vectors, torque, eta, a, *args, **kwargs):
  '''
  Returns the product of the mobility at the blob level to the force
  on the blobs.
  Mobility for particles near a wall.  This uses the expression from
  the Swan and Brady paper for a finite size particle, as opposed to the
  Blake paper point particle result.

  For blobs overlaping the wall we use
  Compute M = B^T * M_tilde(z_effective) * B.

  This function uses pycuda.
  '''
  # Get effective height
  r_vectors_effective = shift_heights(r_vectors, a)
  # Compute damping matrix B
  B, overlap = damping_matrix_B(r_vectors, a, *args, **kwargs)
  # Compute B * vector
  if overlap is True:
    torque = B.dot(torque.flatten())
  # Compute M_tilde * B * torque
  rot = mobility_pycuda.single_wall_mobility_rot_times_torque_pycuda(r_vectors_effective, torque, eta, a, *args, **kwargs)
  # Compute B.T * M * B * torque
  if overlap is True:
    rot = B.dot(rot)
  return rot


def no_wall_mobility_rot_times_torque_pycuda(r_vectors, torque, eta, a, *args, **kwargs):
  '''
  Returns the product of the mobility at the blob level to the force
  on the blobs.
  Mobility for particles near a wall.  This uses the expression from
  the Swan and Brady paper for a finite size particle, as opposed to the 
  Blake paper point particle result. 
  
  This function uses pycuda.
  '''
  rot = mobility_pycuda.no_wall_mobility_rot_times_torque_pycuda(r_vectors, torque, eta, a, *args, **kwargs)
  return rot


def single_wall_mobility_trans_times_force_torque_pycuda(r_vectors, force, torque, eta, a, *args, **kwargs):
  '''
  Returns the product of the mobility at the blob level to the force
  on the blobs.
  Mobility for particles near a wall.  This uses the expression from
  the Swan and Brady paper for a finite size particle, as opposed to the
  Blake paper point particle result.

  For blobs overlaping the wall we use
  Compute M = B^T * M_tilde(z_effective) * B.

  This function uses pycuda.
  '''
  # Get effective height
  r_vectors_effective = shift_heights(r_vectors, a)
  # Compute damping matrix B
  B, overlap = damping_matrix_B(r_vectors, a, *args, **kwargs)
  # Compute B * force, B * torque
  if overlap is True:
    force = B.dot(force)
    torque = B.dot(torque)
  # Compute M_tilde * B * (force + torque)
  velocities = mobility_pycuda.single_wall_mobility_trans_times_force_torque_pycuda(r_vectors_effective, force, torque, eta, a, *args, **kwargs)
  # Compute B.T * M * B * (force + torque)
  if overlap is True:
    velocities = B.dot(velocities)
  return velocities


def no_wall_mobility_trans_times_force_torque_pycuda(r_vectors, force, torque, eta, a, *args, **kwargs):
  '''
  Returns the product of the mobility at the blob level to the force
  on the blobs.
  Mobility for particles near a wall.  This uses the expression from
  the Swan and Brady paper for a finite size particle, as opposed to the 
  Blake paper point particle result. 
  
  This function uses pycuda.
  '''
  velocities = mobility_pycuda.no_wall_mobility_trans_times_force_torque_pycuda(r_vectors, force, torque, eta, a, *args, **kwargs)
  return velocities


def single_wall_mobility_trans_times_torque_pycuda(r_vectors, torque, eta, a, *args, **kwargs):
  '''
  Returns the product of the mobility at the blob level to the force
  on the blobs.
  Mobility for particles near a wall.  This uses the expression from
  the Swan and Brady paper for a finite size particle, as opposed to the
  Blake paper point particle result.

  For blobs overlaping the wall we use
  Compute M = B^T * M_tilde(z_effective) * B.

  This function uses pycuda.
  '''
  # Get effective height
  r_vectors_effective = shift_heights(r_vectors, a)
  # Compute damping matrix B
  B, overlap = damping_matrix_B(r_vectors, a, *args, **kwargs)
  # Compute B * torque
  if overlap is True:
    torque = B.dot(torque.flatten())
  # Compute M_tilde * B * torque
  velocities = mobility_pycuda.single_wall_mobility_trans_times_torque_pycuda(r_vectors_effective, torque, eta, a, *args, **kwargs)
  # Compute B.T * M * B * torque
  if overlap is True:
    velocities = B.dot(velocities)
  return velocities

def in_plane_mobility_trans_times_torque_pycuda(r_vectors, torque, eta, a, *args, **kwargs):
  '''
  Returns the product of the mobility at the blob level to the force
  on the blobs.
  Mobility for particles near a wall.  This uses the expression from
  the Swan and Brady paper for a finite size particle, as opposed to the
  Blake paper point particle result.

  For blobs overlaping the wall we use
  Compute M = B^T * M_tilde(z_effective) * B.

  This function makes use of pycuda.
  '''
  # Get effective height
  r_vectors_effective = shift_heights(r_vectors, a)
  # Compute damping matrix B
  B, overlap = damping_matrix_B(r_vectors, a, *args, **kwargs)
  # Compute B * torque
  if overlap is True:
    torque = B.dot(torque)
  # Compute M_tilde * B * torque
  velocities = mobility_pycuda.in_plane_mobility_trans_times_torque_pycuda(r_vectors_effective, torque, eta, a, *args, **kwargs)
  # Compute B.T * M * B * torque
  if overlap is True:
    velocities = B.dot(velocities)
  return velocities


def no_wall_mobility_trans_times_torque_pycuda(r_vectors, force, eta, a, *args, **kwargs):
  '''
  Returns the product of the mobility at the blob level to the force
  on the blobs. Mobility for particles in an unbounded domain, it uses
  the standard RPY tensor.  
  
  This function uses pycuda.
  '''
  vel = mobility_pycuda.no_wall_mobility_trans_times_torque_pycuda(r_vectors, force, eta, a, *args, **kwargs)
  return vel


def single_wall_mobility_trans_times_force_source_target_pycuda(source, target, force, radius_source, radius_target, eta, *args, **kwargs):
  '''
  Returns the product of the mobility at the blob level by the force
  on the blobs.
  Mobility for particles near a wall.  This uses the expression from
  the Swan and Brady paper for a finite size particle, as opposed to the
  Blake paper point particle result.

  If a component of periodic_length is larger than zero the
  space is assume to be pseudo-periodic in that direction. In that case
  the code will compute the interactions M*f between particles in
  the minimal image convection and also in the first neighbor boxes.

  For blobs overlaping the wall we use
  Compute M = B^T * M_tilde(z_effective) * B.

  This function uses pycuda.
  '''
  # Compute effective heights
  x = shift_heights_different_radius(target, radius_target)
  y = shift_heights_different_radius(source, radius_source)

  # Compute dumping matrices
  B_target, overlap_target = damping_matrix_B_different_radius(target, radius_target, *args, **kwargs)
  B_source, overlap_source = damping_matrix_B_different_radius(source, radius_source, *args, **kwargs)

  # Compute B * force
  if overlap_source is True:
    force = B_source.dot(force.flatten())

  # Compute M_tilde * B * force
  velocities = mobility_pycuda.single_wall_mobility_trans_times_force_source_target_pycuda(y, x, force, radius_source, radius_target, eta, *args, **kwargs)

  # Compute B.T * M * B * vector
  if overlap_target is True:
    velocities = B_target.dot(velocities)
  return velocities


def single_wall_mobility_trans_times_force_source_target_numba(source, target, force, radius_source, radius_target, eta, *args, **kwargs):
  '''
  Returns the product of the mobility at the blob level by the force
  on the blobs.
  Mobility for particles near a wall.  This uses the expression from
  the Swan and Brady paper for a finite size particle, as opposed to the
  Blake paper point particle result.

  If a component of periodic_length is larger than zero the
  space is assume to be pseudo-periodic in that direction. In that case
  the code will compute the interactions M*f between particles in
  the minimal image convection and also in the first neighbor boxes.

  For blobs overlaping the wall we use
  Compute M = B^T * M_tilde(z_effective) * B.

  This function uses numba.
  '''
  # Get domain size for Pseudo-PBC
  L = kwargs.get('periodic_length', np.array([0.0, 0.0, 0.0]))
  
  # Compute effective heights
  x = shift_heights_different_radius(target, radius_target)
  y = shift_heights_different_radius(source, radius_source)

  # Compute dumping matrices
  B_target, overlap_target = damping_matrix_B_different_radius(target, radius_target, *args, **kwargs)
  B_source, overlap_source = damping_matrix_B_different_radius(source, radius_source, *args, **kwargs)

  # Compute B * force
  if overlap_source is True:
    force = B_source.dot(force.flatten())

  # Compute M_tilde * B * force
  velocities = mobility_numba.mobility_trans_times_force_source_target_numba(y, x, force, radius_source, radius_target, eta, L=L, wall=1)

  # Compute B.T * M * B * vector
  if overlap_target is True:
    velocities = B_target.dot(velocities)
  return velocities


def no_wall_mobility_trans_times_force_source_target_numba(source, target, force, radius_source, radius_target, eta, *args, **kwargs):
  '''
  Returns the product of the mobility at the blob level by the force
  on the blobs.
  Mobility for particles in unbounded domain.  

  If a component of periodic_length is larger than zero the
  space is assume to be pseudo-periodic in that direction. In that case
  the code will compute the interactions M*f between particles in
  the minimal image convection and also in the first neighbor boxes.

  For blobs overlaping the wall we use
  Compute M = B^T * M_tilde(z_effective) * B.

  This function uses numba.
  '''
  # Get domain size for Pseudo-PBC
  L = kwargs.get('periodic_length', np.array([0.0, 0.0, 0.0]))

  # Compute M_tilde * B * force
  velocities = mobility_numba.mobility_trans_times_force_source_target_numba(source, target, force, radius_source, radius_target, eta, L=L, wall=0)

  return velocities


def single_wall_fluid_mobility_loops(r_vectors, eta, a, *args, **kwargs):
  ''' 
  Mobility for particles near a wall.  This uses the expression from
  the Swan and Brady paper for a finite size particle, as opposed to the
  Blake paper point particle result.

  For blobs overlaping the wall we use
  Compute M = B^T * M_tilde(z_effective) * B.
  '''
  # Get effective height
  r_vectors_effective = shift_heights(r_vectors, a)
  # Compute damping matrix B
  B, overlap = damping_matrix_B(r_vectors, a, *args, **kwargs)
  num_particles = len(r_vectors_effective)
  # We add the corrections from the appendix of the paper to the unbounded mobility.
  fluid_mobility = rotne_prager_tensor_loops(r_vectors_effective, eta, a)
  for j in range(num_particles):
    for k in range(j+1, num_particles):
      # Here notation is based on appendix C of the Swan and Brady paper:
      # 'Simulation of hydrodynamically interacting particles near a no-slip
      # boundary.'
      h = r_vectors_effective[k][2]
      R = (r_vectors_effective[j] - (r_vectors_effective[k] - 2.*np.array([0., 0., h])))/a
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

  for j in range(len(r_vectors_effective)):
    # Diagonal blocks, self mobility.
    h = r_vectors_effective[j][2]/a
    for l in range(3):
      fluid_mobility[j*3 + l][j*3 + l] += (1./(6.*np.pi*eta*a))*(
        (l != 2)*(-1./16.)*(9./h - 2./(h**3) + 1./(h**5))
        + (l == 2)*(-1./8.)*(9./h - 4./(h**3) + 1./(h**5)))

  # Compute M = B^T * M_tilde * B
  if overlap is True:
    return B.dot( (B.dot(fluid_mobility.T)).T )
  else:
    return fluid_mobility


def rotne_prager_tensor_loops(r_vectors, eta, a, *args, **kwargs):
  ''' 
  Calculate free rotne prager tensor for particles at locations given by
  r_vectors (list of 3 dimensional locations) of radius a.
  '''
  num_particles = len(r_vectors)
  fluid_mobility = np.array([np.zeros(3*num_particles) for _ in range(3*num_particles)])
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
          C1*np.identity(3) + C2*np.outer(r, r)/(np.maximum(r_norm, np.finfo(float).eps)**2)))

      elif j == k:
        # j == k, diagonal block.
        fluid_mobility[(j*3):(j*3 + 3), (k*3):(k*3 + 3)] = ((1./(6.*np.pi*eta*a)) * np.identity(3))
  return fluid_mobility


def single_wall_fluid_mobility_product(r_vectors, vector, eta, a, *args, **kwargs):
  '''
  WARNING: pseudo-PBC are not implemented for this function.

  Product (Mobility * vector). Mobility for particles near a wall.
  This uses the expression from the Swan and Brady paper for a finite
  size particle, as opposed to the Blake paper point particle result.

  For blobs overlaping the wall we use
  Compute M = B^T * M_tilde(z_effective) * B.
  '''
  mobility = single_wall_fluid_mobility(np.reshape(r_vectors, (r_vectors.size // 3, 3)), eta, a)
  velocities = np.dot(mobility, vector)
  return velocities


def no_wall_fluid_mobility_product(r_vectors, vector, eta, a, *args, **kwargs):
  '''
  WARNING: pseudo-PBC are not implemented for this function.

  Product (Mobility * vector). Mobility for particles in an unbounded domain.
  This uses the standard Rotne-Prager-Yamakawa expression.
  '''
  mobility = rotne_prager_tensor(np.reshape(r_vectors, (r_vectors.size // 3, 3)), eta, a)
  velocities = np.dot(mobility, vector)
  return velocities


def single_wall_self_mobility_with_rotation(location, eta, a, *args, **kwargs):
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


def fmm_single_wall_stokeslet(r_vectors, force, eta, a, *args, **kwargs):
  '''
  WARNING: pseudo-PBC are not implemented for this function.

  Compute the Stokeslet interaction plus self mobility
  II/(6*pi*eta*a) in the presence of a wall at z=0.
  It uses the fmm implemented in the library stfmm3d.
  Must compile mobility_fmm.f90 before this will work
  (see Makefile).

  For blobs overlaping the wall we use
  Compute M = B^T * M_tilde(z_effective) * B.
  '''
  # Get effective height
  r_vectors_effective = shift_heights(r_vectors, a)
  # Compute damping matrix B
  B, overlap = damping_matrix_B(r_vectors, a, *args, **kwargs)
  # Compute B * force
  if overlap is True:
    force = B.dot(force)
  # Compute M_tilde * B * vector
  num_particles = r_vectors.size // 3
  ier = 0
  iprec = 5
  r_vectors_fortran = np.copy(r_vectors_effective.T, order='F')
  force_fortran = np.copy(np.reshape(force, (num_particles, 3)).T, order='F')
  u_fortran = np.empty_like(r_vectors_fortran, order='F')
  fmm.fmm_stokeslet_half(r_vectors_fortran, force_fortran, u_fortran, ier, iprec, a, eta, num_particles)
  # Compute B.T * M * B * force
  if overlap is True:
    return B.dot(np.reshape(u_fortran.T, u_fortran.size))
  else:
    return np.reshape(u_fortran.T, u_fortran.size)


def fmm_rpy(r_vectors, force, eta, a, *args, **kwargs):
  '''
  WARNING: pseudo-PBC are not implemented for this function.

  Compute the Stokes interaction using the Rotner-Prager
  tensor. Here there is no wall.
  It uses the fmm implemented in the library rpyfmm.
  Must compile mobility_fmm.f90 before this will work
  (see Makefile).
  '''
  num_particles = r_vectors.size // 3
  ier = 0
  iprec = 1
  r_vectors_fortran = np.copy(r_vectors.T, order='F')
  force_fortran = np.copy(np.reshape(force, (num_particles, 3)).T, order='F')
  u_fortran = np.empty_like(r_vectors_fortran, order='F')
  fmm.fmm_rpy(r_vectors_fortran, force_fortran, u_fortran, ier, iprec, a, eta, num_particles)
  return np.reshape(u_fortran.T, u_fortran.size)


def mobility_vector_product_source_target_one_wall(source, target, force, radius_source, radius_target, eta, *args, **kwargs):
  '''
  WARNING: pseudo-PBC are not implemented for this function.

  Compute velocity of targets of radius radius_target due
  to forces on sources of radius source_target in half-space.

  That is, compute the matrix vector product
  velocities_target = M_tt * forces_sources
  where M_tt has dimensions (target, source)
  '''
  # Compute effective heights
  x = shift_heights_different_radius(target, radius_target)
  y = shift_heights_different_radius(source, radius_source)

  # Compute dumping matrices
  B_target, overlap_target = damping_matrix_B_different_radius(target, radius_target, *args, **kwargs)
  B_source, overlap_source = damping_matrix_B_different_radius(source, radius_source, *args, **kwargs)

  # Compute B * vector
  if overlap_source is True:
    force = B_source.dot(force.flatten())

  # Compute unbounded contribution
  force = np.reshape(force, (force.size // 3, 3))
  velocity = mobility_vector_product_source_target_unbounded(y, x, force, radius_source, radius_target, eta, *args, **kwargs)
  y_image = np.copy(y)
  y_image[:,2] = -y[:,2]

  # Compute wall correction
  prefactor = 1.0 / (8 * np.pi * eta)
  b2 = radius_target**2
  a2 = radius_source**2
  I = np.eye(3)
  J = np.zeros((3,3))
  J[2,2] = 1.0
  P = np.eye(3)
  P[2,2] = -1.0
  delta_3 = np.zeros(3)
  delta_3[2] = 1.0
  # Loop over targets
  for i, r_target in enumerate(x):
    # Distance between target and image sources
    r_source_to_target = r_target - y_image
    x3 = np.zeros(3)
    x3[2] = r_target[2]
    # Loop over sources
    for j, r in enumerate(r_source_to_target):
      y3 = np.zeros(3)
      y3[2] = y[j,2]
      r2 = np.dot(r,r)
      r_norm  = np.sqrt(r2)
      r3 = r_norm * r2
      r5 = r3 * r2
      r7 = r5 * r2
      r9 = r7 * r2
      RR = np.outer(r,r)
      R3 = delta_3 * r[2]

      # Compute 3x3 block mobility
      Mij = ((1+(b2[i]+a2[j])/(3.0*r2)) * I + (1-(b2[i]+a2[j])/r2) * RR / r2) / r_norm
      Mij += 2*(-J/r_norm - np.outer(r,x3)/r3 - np.outer(y3,r)/r3 + x3[2]*y3[2]*(I/r3 - 3*RR/r5))
      Mij += (2*b2[i]/3.0) * (-J/r3 + 3*np.outer(r,R3)/r5 - y3[2]*(3*R3[2]*I/r5 + 3*np.outer(delta_3,r)/r5 + 3*np.outer(r,delta_3)/r5 - 15*R3[2]*RR/r7))
      Mij += (2*a2[j]/3.0) * (-J/r3 + 3*np.outer(R3,r)/r5 - x3[2]*(3*R3[2]*I/r5 + 3*np.outer(delta_3,r)/r5 + 3*np.outer(r,delta_3)/r5 - 15*R3[2]*RR/r7))
      Mij += (2*b2[i]*a2[j]/3.0) * (-I/r5 + 5*R3[2]*R3[2]*I/r7 - J/r5 + 5*np.outer(R3,r)/r7 - J/r5 + 5*np.outer(r,R3)/r7 + 5*np.outer(R3,r)/r7 + 5*RR/r7 + 5*np.outer(r,R3)/r7 -35 * R3[2]*R3[2]*RR/r9)
      Mij = -prefactor * np.dot(Mij, P)
      velocity[i] += np.dot(Mij, force[j])

  # Compute B.T * M * B * vector
  if overlap_target is True:
    velocity = B_target.dot(np.reshape(velocity, velocity.size))

  return velocity


def mobility_vector_product_source_target_unbounded(source, target, force, radius_source, radius_target, eta, *args, **kwargs):
  '''
  WARNING: pseudo-PBC are not implemented for this function.

  Compute velocity of targets of radius radius_target due
  to forces on sources of radius source_targer in unbounded domain.

  That is, compute the matrix vector product
  velocities_target = M_tt * forces_sources
  where M_tt has dimensions (target, source)

  See Reference P. J. Zuk et al. J. Fluid Mech. (2014), vol. 741, R5, doi:10.1017/jfm.2013.668
  '''
  force = np.reshape(force, (force.size // 3, 3))
  velocity = np.zeros((target.size // 3, 3))
  prefactor = 1.0 / (8 * np.pi * eta)
  b2 = radius_target**2
  a2 = radius_source**2
  # Loop over targets
  for i, r_target in enumerate(target):
    # Distance between target and sources
    r_source_to_target = r_target - source
    # Loop over sources
    for j, r in enumerate(r_source_to_target):
      r2 = np.dot(r,r)
      r_norm  = np.sqrt(r2)
      # Compute 3x3 block mobility
      if r_norm >= (radius_target[i] + radius_source[j]):
        Mij = (1 + (b2[i]+a2[j]) / (3 * r2)) * np.eye(3) + (1 - (b2[i]+a2[j]) / r2) * np.outer(r,r) / r2
        Mij = (prefactor / r_norm) * Mij
      elif r_norm > np.absolute(radius_target[i]-radius_source[j]):
        r3 = r_norm * r2
        Mij = ((16*(radius_target[i]+radius_source[j])*r3 - ((radius_target[i]-radius_source[j])**2 + 3*r2)**2) / (32*r3)) * np.eye(3) +\
            ((3*((radius_target[i]-radius_source[j])**2-r2)**2) / (32*r3)) * np.outer(r,r) / r2
        Mij = Mij / (6 * np.pi * eta * radius_target[i] * radius_source[j])
      else:
        largest_radius = radius_target[i] if radius_target[i] > radius_source[j] else radius_source[j]
        Mij = (1.0 / (6 * np.pi * eta * largest_radius)) * np.eye(3)
      velocity[i] += np.dot(Mij, force[j])

  return velocity


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

def rotne_prager_tensor_cpp(r_vectors, eta, a, *args, **kwargs):
  ''' 
  Calculate free rotne prager tensor for particles at locations given by
  r_vectors of radius a.
  '''
  return mobility_cpp.rotne_prager_tensor(r_vectors, eta, a)

def rotne_prager_tensor(r_vectors, eta, a, *args, **kwargs):
  ''' 
  Calculate free rotne prager tensor for particles at locations given by
  r_vectors of radius a.
  '''
  # Extract variables
  r_vectors = r_vectors.reshape((r_vectors.size // 3, 3))
  x = r_vectors[:,0]
  y = r_vectors[:,1]
  z = r_vectors[:,2]
  
  # Compute distances between blobs
  dx = x - x[:, None]
  dy = y - y[:, None]
  dz = z - z[:, None]
  dr = np.sqrt(dx**2 + dy**2 + dz**2)

  # Compute scalar functions f(r) and g(r)
  factor = 1.0 / (6.0 * np.pi * eta)
  fr = np.zeros_like(dr)
  gr = np.zeros_like(dr)
  sel = dr > 2.0 * a
  nsel = np.logical_not(sel)
  sel_zero = dr == 0.
  nsel[sel_zero] = False

  fr[sel] = factor * (0.75 / dr[sel] + a**2 / (2.0 * dr[sel]**3))
  gr[sel] = factor * (0.75 / dr[sel]**3 - 1.5 * a**2 / dr[sel]**5)

  fr[sel_zero] = (factor / a)
  fr[nsel] = factor * (1.0 / a - 0.28125 * dr[nsel] / a**2)
  gr[nsel] = factor * (3.0 / (32.0 * a**2 * dr[nsel]))

  # Build mobility matrix of size 3N \times 3N
  M = np.zeros((r_vectors.size, r_vectors.size))
  M[0::3, 0::3] = fr + gr * dx * dx
  M[0::3, 1::3] =      gr * dx * dy
  M[0::3, 2::3] =      gr * dx * dz

  M[1::3, 0::3] =      gr * dy * dx
  M[1::3, 1::3] = fr + gr * dy * dy
  M[1::3, 2::3] =      gr * dy * dz

  M[2::3, 0::3] =      gr * dz * dx
  M[2::3, 1::3] =      gr * dz * dy
  M[2::3, 2::3] = fr + gr * dz * dz
  return M

def single_wall_fluid_mobility_cpp(r_vectors, eta, a):
  return mobility_cpp.single_wall_fluid_mobility(r_vectors, eta, a)

def single_wall_fluid_mobility(r_vectors, eta, a, *args, **kwargs):
  ''' 
  Mobility for particles near a wall.  This uses the expression from
  the Swan and Brady paper for a finite size particle, as opposed to the 
  Blake paper point particle result. 

  For blobs overlaping the wall we use
  Compute M = B^T * M_tilde(z_effective) * B.
  '''
  # Get effective height
  r_vectors_effective = shift_heights(r_vectors, a)
  # Compute damping matrix B
  B_damp, overlap = damping_matrix_B(r_vectors, a, *args, **kwargs)
  num_particles = len(r_vectors_effective)
  # We add the corrections from the appendix of the paper to the unbounded mobility.
  if 'mobility_cpp' in sys.modules:
    fluid_mobility = mobility_cpp.rotne_prager_tensor(r_vectors_effective, eta, a)
  else:
    fluid_mobility = rotne_prager_tensor(r_vectors_effective, eta, a)


  # Extract variables
  r_vectors_effective = r_vectors_effective.reshape((r_vectors_effective.size // 3, 3))
  N = r_vectors.size // 3
  x = r_vectors_effective[:,0]
  y = r_vectors_effective[:,1]
  z = r_vectors_effective[:,2]
  
  # Compute distances between blobs
  dx = (x[:, None] - x) / a
  dy = (y[:, None] - y) / a
  dz = (z[:, None] + z) / a
  dr = np.sqrt(dx**2 + dy**2 + dz**2)
  h_hat = z[:, None] / (a * dz)
  h = z / a
  ex = dx / dr
  ey = dy / dr
  ez = dz / dr

  # Compute scalar functions, the mobility is
  # M = A*delta_ij + B*e_i*e_j + C*e_i*delta_3j + D*delta_i3*e_j + E*delta_i3*delta_3j 
  factor = 1.0 / (6.0 * np.pi * eta * a)  
  sel = dr > 1e-12 * a
  rows, columns = np.diag_indices(N)
  sel[rows, columns] = False

  # Allocate memory
  M = np.zeros((r_vectors.size, r_vectors.size)) 
  rows, columns = np.diag_indices(r_vectors.size) 
  
  A = np.zeros_like(dr)
  B = np.zeros_like(dr)
  C = np.zeros_like(dr)
  D = np.zeros_like(dr)
  E = np.zeros_like(dr)


  # Self-mobility terms
  A_vec = -0.0625 * (9.0 / h - 2.0 / h**3 + 1.0 / h**5)
  E_vec = -A_vec - 0.125 * (9.0 / h - 4.0 / h**3 + 1.0 / h**5)
  M[rows[0::3], columns[0::3]] += A_vec
  M[rows[1::3], columns[1::3]] += A_vec
  M[rows[2::3], columns[2::3]] += A_vec + E_vec

  # Particle-particle terms
  A[sel] = -0.25 * (3.0 * (1.0 + 2*h_hat[sel]*(1.0-h_hat[sel])*ez[sel]**2) / dr[sel] + 
                    2.0*(1-3.0*ez[sel]**2) / dr[sel]**3 - 2.0*(1-5.0*ez[sel]**2) / dr[sel]**5)

  B[sel] = -0.25 * (3.0 * (1.0 - 6.0*h_hat[sel]*(1.0-h_hat[sel])*ez[sel]**2) / dr[sel] - 
                    6.0 * (1.0 - 5.0*ez[sel]**2) / dr[sel]**3 + 10.0 * (1.0 - 7.0*ez[sel]**2) / dr[sel]**5)

  C[sel] = 0.5 * ez[sel] * (3.0 * h_hat[sel]*(1.0 - 6.0*(1.0-h_hat[sel])*ez[sel]**2) / dr[sel] - 
                            6.0 * (1.0 - 5.0*ez[sel]**2) / dr[sel]**3 + 10.0 * (2.0 - 7.0*ez[sel]**2) / dr[sel]**5)

  D[sel] = 0.5 * ez[sel] * (3.0 * h_hat[sel] / dr[sel] - 10.0 / dr[sel]**5)

  E[sel] = -(3.0 * h_hat[sel]**2 * ez[sel]**2 / dr[sel] + 3.0 * ez[sel]**2 / dr[sel]**3 + (2.0 - 15.0*ez[sel]**2) / dr[sel]**5)
  
  # Build mobility matrix of size 3N \times 3N
  M[0::3, 0::3] += A + B * ex * ex 
  M[0::3, 1::3] +=     B * ex * ey
  M[0::3, 2::3] +=     B * ex * ez + C.T * ex

  M[1::3, 0::3] +=     B * ey * ex
  M[1::3, 1::3] += A + B * ey * ey
  M[1::3, 2::3] +=     B * ey * ez + C.T * ey 

  M[2::3, 0::3] +=     B * ez * ex            + D.T * ex 
  M[2::3, 1::3] +=     B * ez * ey            + D.T * ey 
  M[2::3, 2::3] += A + B * ez * ez + C.T * ez + D.T * ez + E.T

  M *= factor
  M += fluid_mobility

  # Compute M = B^T * M_tilde * B
  if overlap is True:
    return B_damp.dot( (B_damp.dot(M.T)).T )
  else:
    return M


def no_wall_mobility_trans_times_force_numba(r_vectors, force, eta, a, *args, **kwargs):
  ''' 
  Returns the product of the mobility at the blob level to the force 
  on the blobs. Mobility for particles in an unbounded domain, it uses
  the standard RPY tensor.  
  
  This function uses numba.
  '''
  L = kwargs.get('periodic_length', np.array([0.0, 0.0, 0.0]))
  vel = mobility_numba.no_wall_mobility_trans_times_force_numba(r_vectors, force, eta, a, L)
  return vel


def single_wall_mobility_trans_times_force_numba(r_vectors, force, eta, a, *args, **kwargs):
  ''' 
  Returns the product of the mobility at the blob level by the force 
  on the blobs.
  Mobility for particles near a wall.  This uses the expression from
  the Swan and Brady paper for a finite size particle, as opposed to the 
  Blake paper point particle result. 
   
  If a component of periodic_length is larger than zero the
  space is assume to be pseudo-periodic in that direction. In that case
  the code will compute the interactions M*f between particles in
  the minimal image convection and also in the first neighbor boxes. 

  For blobs overlaping the wall we use
  Compute M = B^T * M_tilde(z_effective) * B.

  This function uses numba.
  '''
  L = kwargs.get('periodic_length', np.array([0.0, 0.0, 0.0]))
  # Get effective height
  r_vectors_effective = shift_heights(r_vectors, a)
  # Compute damping matrix B
  B, overlap = damping_matrix_B(r_vectors, a, *args, **kwargs)
  # Compute B * force
  if overlap is True:
    force = B.dot(force.flatten())
  # Compute M_tilde * B * force
  velocities = mobility_numba.single_wall_mobility_trans_times_force_numba(r_vectors_effective, force, eta, a, L)
  # Compute B.T * M * B * vector
  if overlap is True:
    velocities = B.dot(velocities)
  return velocities


def in_plane_mobility_trans_times_force_numba(r_vectors, force, eta, a, *args, **kwargs):
  ''' 
  Returns the product of the mobility at the blob level by the force 
  on the blobs.
  Mobility for particles near a wall, fixed in a plane.  This uses the expression from
  the Swan and Brady paper for a finite size particle, as opposed to the 
  Blake paper point particle result. 
   
  If a component of periodic_length is larger than zero the
  space is assume to be pseudo-periodic in that direction. In that case
  the code will compute the interactions M*f between particles in
  the minimal image convection and also in the first neighbor boxes. 

  For blobs overlaping the wall we use
  Compute M = B^T * M_tilde(z_effective) * B.

  This function uses numba.
  '''
  L = kwargs.get('periodic_length', np.array([0.0, 0.0, 0.0]))
  # Get effective height
  r_vectors_effective = shift_heights(r_vectors, a)
  # Compute damping matrix B
  B, overlap = damping_matrix_B(r_vectors, a, *args, **kwargs)
  # Compute B * force
  if overlap is True:
    force = B.dot(force.flatten())
  # Compute M_tilde * B * force
  velocities = mobility_numba.in_plane_mobility_trans_times_force_numba(r_vectors_effective, force, eta, a, L)
  # Compute B.T * M * B * vector
  if overlap is True:
    velocities = B.dot(velocities)
  return velocities


def no_wall_mobility_trans_times_torque_numba(r_vectors, torque, eta, a, *args, **kwargs):
  ''' 
  Returns the product of the mobility at the blob level to the force 
  on the blobs. Mobility for particles in an unbounded domain, it uses
  the standard RPY tensor.  
  
  This function uses numba.
  '''
  L = kwargs.get('periodic_length', np.array([0.0, 0.0, 0.0]))
  vel = mobility_numba.no_wall_mobility_trans_times_torque_numba(r_vectors, torque, eta, a, L)
  return vel


def single_wall_mobility_trans_times_torque_numba(r_vectors, torque, eta, a, *args, **kwargs):
  ''' 
  Returns the product of the mobility at the blob level to the force 
  on the blobs. Mobility for particles on top of an infinite wall.
  
  This function uses numba.
  '''
  L = kwargs.get('periodic_length', np.array([0.0, 0.0, 0.0]))
  # Get effective height
  r_vectors_effective = shift_heights(r_vectors, a)
  # Compute damping matrix B
  B, overlap = damping_matrix_B(r_vectors, a, *args, **kwargs)
  # Compute B * force
  if overlap is True:
    torque = B.dot(torque.flatten())
  # Compute M_tilde * B * force
  velocities = mobility_numba.single_wall_mobility_trans_times_torque_numba(r_vectors_effective, torque, eta, a, L)
  # Compute B.T * M * B * vector
  if overlap is True:
    velocities = B.dot(velocities)
  return velocities

def in_plane_mobility_trans_times_torque_numba(r_vectors, torque, eta, a, *args, **kwargs):
  ''' 
  Returns the product of the mobility at the blob level to the force 
  on the blobs. Mobility for particles on top of an infinite wall.
  
  This function uses numba.
  '''
  L = kwargs.get('periodic_length', np.array([0.0, 0.0, 0.0]))
  # Get effective height
  r_vectors_effective = shift_heights(r_vectors, a)
  # Compute damping matrix B
  B, overlap = damping_matrix_B(r_vectors, a, *args, **kwargs)
  # Compute B * force
  if overlap is True:
    torque = B.dot(torque.flatten())
  # Compute M_tilde * B * force
  velocities = mobility_numba.in_plane_mobility_trans_times_torque_numba(r_vectors_effective, torque, eta, a, L)
  # Compute B.T * M * B * vector
  if overlap is True:
    velocities = B.dot(velocities)
  return velocities


def no_wall_mobility_rot_times_force_numba(r_vectors, force, eta, a, *args, **kwargs):
  ''' 
  Returns the product of the mobility rotational-translation at the blob level to the force 
  on the blobs.
  Mobility for particles near a wall.  This uses the expression from
  the Swan and Brady paper for a finite size particle, as opposed to the 
  Blake paper point particle result. 
  
  This function uses numba.
  '''
  L = kwargs.get('periodic_length', np.array([0.0, 0.0, 0.0]))
  rot = mobility_numba.no_wall_mobility_rot_times_force_numba(r_vectors, force, eta, a, L)
  return rot


def single_wall_mobility_rot_times_force_numba(r_vectors, force, eta, a, *args, **kwargs):
  ''' 
  Returns the product of the mobility at the blob level to the force 
  on the blobs.
  Mobility for particles near a wall.  This uses the expression from
  the Swan and Brady paper for a finite size particle, as opposed to the 
  Blake paper point particle result. 

  For blobs overlaping the wall we use
  Compute M = B^T * M_tilde(z_effective) * B.
  
  This function uses pycuda.
  '''
  L = kwargs.get('periodic_length', np.array([0.0, 0.0, 0.0]))
  # Get effective height
  r_vectors_effective = shift_heights(r_vectors, a)
  # Compute damping matrix B
  B, overlap = damping_matrix_B(r_vectors, a, *args, **kwargs)
  # Compute B * force
  if overlap is True:
    force = B.dot(force.flatten())
  # Compute M_tilde * B * force
  rot = mobility_numba.single_wall_mobility_rot_times_force_numba(r_vectors_effective, force, eta, a, L)
  # Compute B.T * M * B * force
  if overlap is True:
    rot = B.dot(rot)
  return rot


def no_wall_mobility_rot_times_torque_numba(r_vectors, torque, eta, a, *args, **kwargs):
  ''' 
  Returns the product of the mobility at the blob level to the force 
  on the blobs.
  Mobility for particles near a wall.  This uses the expression from
  the Swan and Brady paper for a finite size particle, as opposed to the 
  Blake paper point particle result. 
  
  This function uses numba.
  '''
  L = kwargs.get('periodic_length', np.array([0.0, 0.0, 0.0]))
  rot = mobility_numba.no_wall_mobility_rot_times_torque_numba(r_vectors, torque, eta, a, L)
  return rot


def single_wall_mobility_rot_times_torque_numba(r_vectors, torque, eta, a, *args, **kwargs):
  ''' 
  Returns the product of the mobility at the blob level to the force 
  on the blobs.
  Mobility for particles near a wall.  This uses the expression from
  the Swan and Brady paper for a finite size particle, as opposed to the 
  Blake paper point particle result. 
  
  For blobs overlaping the wall we use
  Compute M = B^T * M_tilde(z_effective) * B.

  This function uses pycuda.
  '''
  L = kwargs.get('periodic_length', np.array([0.0, 0.0, 0.0]))
  # Get effective height
  r_vectors_effective = shift_heights(r_vectors, a)
  # Compute damping matrix B
  B, overlap = damping_matrix_B(r_vectors, a, *args, **kwargs)
  # Compute B * vector
  if overlap is True:
    torque = B.dot(torque.flatten())
  # Compute M_tilde * B * torque
  rot = mobility_numba.single_wall_mobility_rot_times_torque_numba(r_vectors_effective, torque, eta, a, L)
  # Compute B.T * M * B * torque
  if overlap is True:
    rot = B.dot(rot)
  return rot



def no_wall_pressure_Stokeslet_numba(source, target, force, *args, **kwargs):
  ''' 
  Returns the pressure created by Stokeslets located at source in the positions
  of the targets. The space is unbounded.
  
  This function uses numba.
  '''
  L = kwargs.get('periodic_length', np.array([0.0, 0.0, 0.0]))
  p = mobility_numba.no_wall_pressure_Stokeslet_numba(source, target, force, L)
  return p


def single_wall_pressure_Stokeslet_numba(source, target, force, *args, **kwargs):
  ''' 
  Returns the pressure created by Stokeslets located at source in the positions
  of the targets. Stokeslets above an infinite no-slip wall.

  This function uses numba.
  '''
  L = kwargs.get('periodic_length', np.array([0.0, 0.0, 0.0]))
  p = mobility_numba.single_wall_pressure_Stokeslet_numba(source, target, force, L)
  return p


def mobility_radii_trans_times_force(r_vectors, force, eta, a, radius_blobs, function, *args, **kwargs): 
  '''
  Mobility vector product M*f with blobs with different radii.
  function should provide the appropiate implementation (python, numba, pycuda, above a wall or unbounded...).
  '''
  return function(r_vectors, r_vectors, force, radius_blobs, radius_blobs, eta, *args, **kwargs)

