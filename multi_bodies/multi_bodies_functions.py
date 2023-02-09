'''
In this module the user can define functions that modified the
code multi_blobs.py. For example, functions to define the
blobs-blobs interactions, the forces and torques on the rigid
bodies or the slip on the blobs.
'''

import numpy as np
import sys
import imp
import os.path
from functools import partial

import general_application_utils as utils
from quaternion_integrator.quaternion import Quaternion

# If pycuda is installed import forces_pycuda
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
      from . import forces_pycuda
    except ImportError:
      from .multi_bodies import forces_pycuda
# If numba is installed import forces_numba
try: 
  imp.find_module('numba')
  found_numba = True
except ImportError:
  found_numba = False
if found_numba:
  import forces_numba
try:
  import forces_cpp
except ImportError:
  pass

# Override forces_pycuda with user defined functions.
# If forces_pycuda_user_defined does not exists nothing happens.
if found_pycuda:
  forces_pycuda_user_defined = False
  if os.path.isfile('forces_pycuda_user_defined.py'):
    forces_pycuda_user_defined = True
  if forces_pycuda_user_defined:
    del sys.modules['forces_pycuda']
    sys.modules['forces_pycuda'] = __import__('forces_pycuda_user_defined')
    from . import forces_pycuda

# Override forces_numba with user defined functions.
# If forces_pycuda_user_defined does not exists nothing happens.
if found_numba:
  forces_numba_user_defined = False
  if os.path.isfile('forces_numba_user_defined.py'):
    forces_numba_user_defined = True
  if forces_numba_user_defined:
    del sys.modules['forces_numba']
    sys.modules['forces_numba'] = __import__('forces_numba_user_defined')
    import forces_numba
    

def project_to_periodic_image(r, L):
  '''
  Project a vector r to the minimal image representation
  centered around (0,0,0) and of size L=(Lx, Ly, Lz). If 
  any dimension of L is equal or smaller than zero the 
  box is assumed to be infinite in that direction.
  '''
  if L is not None:
    for i in range(3):
      if(L[i] > 0):
        r[i] = r[i] - int(r[i] / L[i] + 0.5 * (int(r[i]>0) - int(r[i]<0))) * L[i]
  return r

def default_zero_r_vectors(r_vectors, *args, **kwargs):
  return np.zeros((r_vectors.size // 3, 3))


def default_zero_blobs(body, *args, **kwargs):
  ''' 
  Return a zero array of shape (body.Nblobs, 3)
  '''
  return np.zeros((body.Nblobs, 3))


def default_zero_bodies(bodies, *args, **kwargs):
  ''' 
  Return a zero array of shape (2*len(bodies), 3)
  '''
  return np.zeros((2*len(bodies), 3))
  

def set_slip_by_ID(body, slip, *args, **kwargs):
  '''
  This function assign a slip function to each body.
  If the body has an associated slip file the function
  "active_body_slip" is assigned (see function below).
  Otherwise the slip is set to zero.

  This function can be override to assign other slip
  functions based on the body ID, (ID of a structure
  is the name of the clones file (without .clones)).
  See the example in
  "examples/pair_active_rods/".
  '''
  if slip is not None:
    active_body_slip_partial = partial(active_body_slip, slip = slip)
    body.function_slip = active_body_slip_partial
  else:
    body.function_slip = default_zero_blobs
  return


def active_body_slip(body, slip):
  '''
  This function set the slip read from the *.slip file to the
  blobs. The slip on the file is given in the body reference 
  configuration (quaternion = (1,0,0,0)) therefore this
  function rotates the slip to the current body orientation.
  
  This function can be used, for example, to model active rods
  that propel along their axes. 
  '''
  # Get rotation matrix
  rotation_matrix = body.orientation.rotation_matrix()

  # Rotate  slip on each blob
  slip_rotated = np.empty((body.Nblobs, 3))
  for i in range(body.Nblobs):
    slip_rotated[i] = np.dot(rotation_matrix, slip[i])
  return slip_rotated


def bodies_external_force_torque(bodies, r_vectors, *args, **kwargs):
  '''
  This function returns the external force-torques acting on the bodies.
  It returns an array with shape (2*len(bodies), 3)
  
  In this is example we just set it to zero.
  '''
  return np.zeros((2*len(bodies), 3))
  

def blob_external_forces(r_vectors, *args, **kwargs):
  '''
  This function compute the external force acting on a
  single blob. It returns an array with shape (3).

  In this example we add gravity and a repulsion with the wall;
  the interaction with the wall is derived from the potential

  U(z) = U0 + U0 * (a-z)/b   if z<a
  U(z) = U0 * exp(-(z-a)/b)  iz z>=a

  with
  e = repulsion_strength_wall
  a = blob_radius
  h = distance to the wall
  b = debye_length_wall
  '''
  N = r_vectors.size // 3
  f = np.zeros((N, 3))

  # Get parameters from arguments
  blob_mass = kwargs.get('blob_mass')
  blob_radius = kwargs.get('blob_radius')
  g = kwargs.get('g')
  repulsion_strength_wall = kwargs.get('repulsion_strength_wall')
  debye_length_wall = kwargs.get('debye_length_wall')
  # Add gravity
  f[:,2] = -g * blob_mass

  # Add wall interaction
  h = r_vectors[:,2]
  lr_mask = h > blob_radius
  sr_mask = h <= blob_radius
  f[lr_mask,2] += (repulsion_strength_wall / debye_length_wall) * np.exp(-(h[lr_mask]-blob_radius)/debye_length_wall)
  f[sr_mask,2] += (repulsion_strength_wall / debye_length_wall)

  return f

def blob_external_force(r_vectors, *args, **kwargs):
  '''
  This function compute the external force acting on a
  single blob. It returns an array with shape (3).
  
  In this example we add gravity and a repulsion with the wall;
  the interaction with the wall is derived from the potential

  U(z) = U0 + U0 * (a-z)/b   if z<a
  U(z) = U0 * exp(-(z-a)/b)  iz z>=a

  with 
  e = repulsion_strength_wall
  a = blob_radius
  h = distance to the wall
  b = debye_length_wall
  '''
  f = np.zeros(3)

  # Get parameters from arguments
  blob_mass = kwargs.get('blob_mass')
  blob_radius = kwargs.get('blob_radius')
  g = kwargs.get('g')
  repulsion_strength_wall = kwargs.get('repulsion_strength_wall') 
  debye_length_wall = kwargs.get('debye_length_wall')
  # Add gravity
  f += -g * blob_mass * np.array([0., 0., 1.0])

  # Add wall interaction
  h = r_vectors[2]
  if h > blob_radius:
    f[2] += (repulsion_strength_wall / debye_length_wall) * np.exp(-(h-blob_radius)/debye_length_wall)
  else:
    f[2] += (repulsion_strength_wall / debye_length_wall)
  return f


def calc_one_blob_forces(r_vectors, *args, **kwargs):
  '''
  Compute one-blob forces. It returns an array with shape (Nblobs, 3).
  '''
  Nblobs = r_vectors.size // 3
  force_blobs = np.zeros((Nblobs, 3))
  r_vectors = np.reshape(r_vectors, (Nblobs, 3))
  
  # Loop over blobs
  force_blobs = blob_external_forces(r_vectors, *args, **kwargs)
  return force_blobs


def calc_one_blob_torques(r_vectors, *args, **kwargs):
  ''' 
  Compute one-blob torques. It returns an array with shape (Nblobs, 3).
  '''
  Nblobs = r_vectors.size // 3
  return np.zeros((Nblobs, 3)) 


def set_blob_blob_forces(implementation, *args, **kwargs):
  '''
  Set the function to compute the blob-blob forces
  to the right function.
  The implementations in numba, pycuda and C++ are much faster than the
  implimentation in python.
  To use the pycuda implementation is necessary to have installed pycuda and a GPU 
  with CUDA capabilities. To use the C++ implementation the user has to compile 
  the file blob_blob_forces.cpp.
  '''
  if implementation == 'None':
    return default_zero_r_vectors
  elif implementation == 'python':
    return calc_blob_blob_forces_python
  elif implementation == 'C++':
    return calc_blob_blob_forces_cpp
  elif implementation == 'pycuda':
    return forces_pycuda.calc_blob_blob_forces_pycuda
  elif implementation == 'numba':
    return forces_numba.calc_blob_blob_forces_numba
  elif implementation == 'tree_numba':
    return forces_numba.calc_blob_blob_forces_tree_numba
  elif implementation == 'radii_numba':
    # Get blobs radii
    bodies = kwargs.get('bodies')
    radius_blobs = []
    for k, b in enumerate(bodies):
      radius_blobs.append(b.blobs_radius)
    radius_blobs = np.concatenate(radius_blobs, axis=0)    
    return partial(forces_numba.calc_blob_blob_forces_radii_numba, radius_blobs=radius_blobs)


def calc_blob_blob_forces_cpp(r_vectors, *args, **kwargs):
  '''
  This function computes the blob-blob forces and returns
  an array with shape (Nblobs, 3).
  '''

  # Get parameters from arguments
  L = kwargs.get('periodic_length')
  eps = kwargs.get('repulsion_strength')
  b = kwargs.get('debye_length')
  a = kwargs.get('blob_radius')

  forces = forces_cpp.blob_blob_force(r_vectors, L, eps, b, a)
  return np.reshape(forces, (forces.size // 3, 3))


def blob_blob_force(r, *args, **kwargs):
  '''
  This function compute the force between two blobs
  with vector between blob centers r.

  In this example the force is derived from the potential
  
  U(r) = U0 + U0 * (2*a-r)/b   if z<2*a
  U(r) = U0 * exp(-(r-2*a)/b)  iz z>=2*a
  
  with
  eps = potential strength
  r_norm = distance between blobs
  b = Debye length
  a = blob_radius
  '''
  # Get parameters from arguments
  L = kwargs.get('periodic_length')
  eps = kwargs.get('repulsion_strength')
  b = kwargs.get('debye_length')
  a = kwargs.get('blob_radius')

  # Compute force
  project_to_periodic_image(r, L)
  r_norm = np.linalg.norm(r)
  if r_norm > 2*a:
    return -((eps / b) * np.exp(-(r_norm-2*a) / b) / np.maximum(r_norm, np.finfo(float).eps)) * r 
  else:
    return -((eps / b) / np.maximum(r_norm, np.finfo(float).eps)) * r;
  

def calc_blob_blob_forces_python(r_vectors, *args, **kwargs):
  '''
  This function computes the blob-blob forces and returns
  an array with shape (Nblobs, 3).
  '''
  Nblobs = r_vectors.size // 3
  force_blobs = np.zeros((Nblobs, 3))

  # Double loop over blobs to compute forces
  for i in range(Nblobs-1):
    for j in range(i+1, Nblobs):
      # Compute vector from j to u
      r = r_vectors[j] - r_vectors[i]
      force = blob_blob_force(r, *args, **kwargs)
      force_blobs[i] += force
      force_blobs[j] -= force

  return force_blobs


def set_body_body_forces_torques(implementation):
  '''
  Set the function to compute the body-body forces
  to the right function. 
  '''
  if implementation == 'None':
    return default_zero_bodies
  elif implementation == 'python':
    return calc_body_body_forces_torques_python


def body_body_force_torque(r, quaternion_i, quaternion_j, *args, **kwargs):
  '''
  This function compute the force between two bodies
  with vector between locations r.
  In this example the torque is zero and the force 
  is derived from a Yukawa potential
  
  U = eps * exp(-r_norm / b) / r_norm
  
  with
  eps = potential strength
  r_norm = distance between bodies' location
  b = Debye length
  '''
  force_torque = np.zeros((2, 3))

  # Get parameters from arguments
  L = kwargs.get('periodic_length')
  eps = kwargs.get('repulsion_strength')
  b = kwargs.get('debye_length')
  
  # Compute force
  project_to_periodic_image(r, L)
  r_norm = np.linalg.norm(r)
  force_torque[0] = -((eps / b) + (eps / r_norm)) * np.exp(-r_norm / b) * r / r_norm**2  
  return force_torque


def calc_body_body_forces_torques_python(bodies, r_vectors, *args, **kwargs):
  '''
  This function computes the body-body forces and torques and returns
  an array with shape (2*Nblobs, 3).
  '''
  Nbodies = len(bodies)
  force_torque_bodies = np.zeros((2*len(bodies), 3))
  
  # Double loop over bodies to compute forces
  for i in range(Nbodies-1):
    for j in range(i+1, Nbodies):
      # Compute vector from j to u
      r = bodies[j].location - bodies[i].location
      force_torque = body_body_force_torque(r, bodies[i].orientation, bodies[j].orientation, *args, **kwargs)
      # Add forces
      force_torque_bodies[2*i] += force_torque[0]
      force_torque_bodies[2*j] -= force_torque[0]
      # Add torques
      force_torque_bodies[2*i+1] += force_torque[1]
      force_torque_bodies[2*j+1] -= force_torque[1]

  return force_torque_bodies


def force_torque_calculator_sort_by_bodies(bodies, r_vectors, *args, **kwargs):
  '''
  Return the forces and torque in each body with
  format [f_1, t_1, f_2, t_2, ...] and shape (2*Nbodies, 3),
  where f_i and t_i are the force and torque on the body i.
  '''
  # Create auxiliar variables
  Nblobs = r_vectors.size // 3
  force_torque_bodies = np.zeros((2*len(bodies), 3))
  force_blobs = np.zeros((Nblobs, 3))
  blob_mass = 1.0
  blob_radius = bodies[0].blob_radius

  # Compute one-blob forces (same function for all blobs)
  force_blobs += calc_one_blob_forces(r_vectors, blob_radius = blob_radius, blob_mass = blob_mass, *args, **kwargs)

  # Compute blob-blob forces (same function for all pair of blobs)
  force_blobs += calc_blob_blob_forces(r_vectors, blob_radius = blob_radius, *args, **kwargs)  
  # Compute body force-torque forces from blob forces
  offset = 0
  for k, b in enumerate(bodies):
    # Add force to the body
    force_torque_bodies[2*k:(2*k+1)] += sum(force_blobs[offset:(offset+b.Nblobs)])
    # Add torque to the body
    R = b.calc_rot_matrix()  
    force_torque_bodies[2*k+1:2*k+2] += np.dot(R.T, np.reshape(force_blobs[offset:(offset+b.Nblobs)], 3*b.Nblobs))
    offset += b.Nblobs

  # Add one-body external force-torque
  force_torque_bodies += bodies_external_force_torque(bodies, r_vectors, *args, **kwargs)

  # Add body-body forces (same for all pair of bodies)
  force_torque_bodies += calc_body_body_forces_torques(bodies, r_vectors, *args, **kwargs)

  return force_torque_bodies


def preprocess(bodies, *args, **kwargs):
  '''
  This function is call at the start of the schemes.
  The default version do nothing, it should be modify by
  the user if he wants to change the schemes.
  '''
  return

def postprocess(bodies, *args, **kwargs):
  '''
  This function is call at the end of the schemes but
  before checking if the postions are a valid configuration.
  The default version do nothing, it should be modify by
  the user if he wants to change the schemes.
  '''
  return


# Override force interactions by user defined functions.
# This only override the functions implemented in python.
# If user_defined_functions is empty or does not exists
# this import does nothing.
user_defined_functions_found = False
if os.path.isfile('user_defined_functions.py'):
  user_defined_functions_found = True
if user_defined_functions_found:
  import user_defined_functions

