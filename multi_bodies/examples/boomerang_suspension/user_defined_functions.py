'''
Use this module to override forces interactions defined in 
multi_body_functions.py. See an example in the file
RigidMultiblobsWall/multi_bodies/examples/user_defined_functions.py



In this module we override the default blob-blob, blob-wall and
body-body interactions used by the code. To use this implementation 
copy this file to 
RigidMultiblobsWall/multi_bodies/user_defined_functions.py


This module defines (and override) the following interactions:

1. blob-wall forces, they are derived from the potential:
  U = e * a * exp(-(h-a) / b) / (h - a)
  with 
  e = repulsion_strength_wall
  a = blob_radius
  h = distance to the wall
  b = debye_length_wall

2. blob-blob forces, they are derived from the potential:
  U = eps * exp(-r_norm / b) / r_norm 
  with
  eps = potential strength
  r_norm = distance between blobs
  b = Debye length

3. body-body forces and torques. The torques are zero and the forces 
  are derived from the potential:
  U = 0.5 * eps * (r_norm - 1.0)**2
  with
  eps = potential strength
  r_norm = distance between bodies' location
'''

import multi_bodies_functions
from multi_bodies_functions import *



# Override blob_external_force 
def blob_external_force_new(r_vectors, *args, **kwargs):
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
  f += np.array([0., 0., (blob_radius * repulsion_strength_wall * \
                          ((h - blob_radius) / debye_length_wall + 1.0) * \
                          np.exp(-1.0 * (h - blob_radius) / debye_length_wall) / \
                          ((h - blob_radius)**2))])
  return f
multi_bodies_functions.blob_external_force = blob_external_force_new


# Override blob_blob_force
def blob_blob_force_new(r, *args, **kwargs):
  '''
  This function compute the force between two blobs
  with vector between blob centers r.
  In this example the force is derived from a Yukawa potential
  
  U = eps * exp(-r_norm / b) / r_norm
  
  with
  eps = potential strength
  r_norm = distance between blobs
  b = Debye length
  '''
  # Get parameters from arguments
  L = kwargs.get('periodic_length')
  eps = kwargs.get('repulsion_strength')
  b = kwargs.get('debye_length')
  
  # Compute force
  project_to_periodic_image(r, L)
  r_norm = np.linalg.norm(r)
  return -((eps / b) + (eps / r_norm)) * np.exp(-r_norm / b) * r / r_norm**2   
multi_bodies_functions.blob_blob_force = blob_blob_force_new


# Override body_body_force_torque
def body_body_force_torque_new(r, quaternion_i, quaternion_j, *args, **kwargs):
  '''
  This function compute the force between two bodies
  with vector between locations r.
  In this example the torque is zero and the force 
  is derived from an harmonic potential
  
  U = 0.5 * eps * (r_norm - 1.0)**2
  
  with
  eps = potential strength
  r_norm = distance between bodies' location
  '''
  force_torque = np.zeros((2, 3))

  # Get parameters from arguments
  L = kwargs.get('periodic_length')
  eps = kwargs.get('repulsion_strength')
  b = 1.0
  # Compute force
  project_to_periodic_image(r, L)
  r_norm = np.linalg.norm(r)
  force_torque[0] = eps * (r_norm - b) * (r / r_norm) 
  return force_torque
multi_bodies_functions.body_body_force_torque =  body_body_force_torque_new






