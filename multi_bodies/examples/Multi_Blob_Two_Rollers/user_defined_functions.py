'''
Use this module to override forces interactions defined in 
multi_body_functions.py. See an example in the file
RigidMultiblobsWall/multi_bodies/examples/user_defined_functions.py



In this module we override the default blob-blob, blob-wall and
body-body interactions used by the code. To use this implementation 
copy this file to 
RigidMultiblobsWall/multi_bodies/user_defined_functions.py


This module defines (and override) the following interactions:

'''

import multi_bodies_functions
from multi_bodies_functions import *



# bodies_external_force_torque 
def bodies_external_force_torque_new(bodies, r_vectors, *args, **kwargs):
  '''
  This function compute the external force acting on a
  single blob. It returns an array with shape (3).
  
  In this example we add gravity and a repulsion with the wall;
  the interaction with the wall is derived from a Yukawa-like
  potential
  U = e * a * exp(-(h-a) / b) / (h - a)
  with 
  e = repulsion_strength_wall
  a = blob_radius
  h = distance to the wall
  b = debye_length_wall
  '''
  FT = np.zeros((2*len(bodies), 3))
  # Get parameters from arguments
  particle_radius = kwargs.get('particle_radius')
  g = kwargs.get('g')
  eta = kwargs.get('eta')
  repulsion_strength_wall = kwargs.get('repulsion_strength_wall')
  debye_length_wall = kwargs.get('debye_length_wall')
  omega_one_roller = kwargs.get('omega_one_roller')
  for k, b in enumerate(bodies):
    # Add gravity and wall interaction
    FT[2*k,2] += -g
    FT[2*k+1,:] += 8.0*np.pi*eta*(particle_radius**3)*omega_one_roller
    h = b.location[2]
    if h > particle_radius:
        FT[2*k,2] += (repulsion_strength_wall / debye_length_wall) * np.exp(-(h-particle_radius)/debye_length_wall)
    else:
        FT[2*k,2] += (repulsion_strength_wall / debye_length_wall)
  return FT
multi_bodies_functions.bodies_external_force_torque = bodies_external_force_torque_new


# body_body_force_torque
def body_body_force_torque_new(r, quaternion_i, quaternion_j, *args, **kwargs):
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
  particle_radius = kwargs.get('particle_radius')
  
  # Compute force
  project_to_periodic_image(r, L)
  r_norm = np.linalg.norm(r)
  if r_norm > 2*particle_radius:
    force_torque[0] = -((eps / b) * np.exp(-(r_norm-2*particle_radius) / b) / np.maximum(r_norm, np.finfo(float).eps)) * r 
  else:
    force_torque[0] = -((eps / b) / np.maximum(r_norm, np.finfo(float).eps)) * r;
  return force_torque  
multi_bodies_functions.body_body_force_torque = body_body_force_torque_new


# Override blob_external_force 
def blob_external_force_new(r_vectors, *args, **kwargs):
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
  return f
multi_bodies_functions.blob_external_force =  blob_external_force_new

def blob_blob_force_new(r, *args, **kwargs):
  '''
  This function compute the force between two blobs
  with vector between blob centers r.
  '''
  # Get parameters from arguments
  return 0 * r 
multi_bodies_functions.blob_blob_force =  blob_blob_force_new







