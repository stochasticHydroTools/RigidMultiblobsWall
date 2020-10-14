'''
Use this module to override forces interactions defined in 
multi_body_functions.py. See an example in the file
'''

import multi_bodies_functions
from multi_bodies_functions import *


def bodies_external_force_torque_new(bodies, r_vectors, *args, **kwargs):
  '''
  This function returns the external force-torques acting on the bodies.
  It returns an array with shape (2*len(bodies), 3)
  
  '''
  force_torque_bodies = np.zeros((2*len(bodies), 3))
  

  # Get parameters from arguments
  blob_mass = 1.0
  blob_radius = bodies[0].blob_radius
  g = kwargs.get('g')
  repulsion_strength_wall = kwargs.get('repulsion_strength_wall') 
  debye_length_wall = kwargs.get('debye_length_wall')

  # Loop over bodies
  for k, b in enumerate(bodies):
    f = np.zeros(3)
    t = np.zeros(3)
    # Add gravity
    f -= g * blob_mass * np.array([0., 0., 1.0])

    # Add wall interaction
    h = b.location[2]
    if h > blob_radius:
      f[2] += (repulsion_strength_wall / debye_length_wall) * np.exp(-(h-blob_radius)/debye_length_wall)
    else:
      f[2] += (repulsion_strength_wall / debye_length_wall)
    force_torque_bodies[2*k:(2*k+1)] = f
 
  return force_torque_bodies
multi_bodies_functions.bodies_external_force_torque = bodies_external_force_torque_new

# We set the blob external force to zero
def blob_external_force_new(r_vectors, *args, **kwargs):
  f = np.zeros(3)
  return f
multi_bodies_functions.blob_external_force = blob_external_force_new

# We define the external torque on the micrcollers 
def blob_external_torque_new(r_vectors, *args, **kwargs):
  blob_mass = 1.0
  blob_radius = kwargs.get('blob_radius') 
  g = kwargs.get('g')
  t = 75.0 * g * blob_mass * blob_radius * np.array([0., 1., 0.0])
  return t
multi_bodies_functions.blob_external_torque = blob_external_torque_new

def calc_one_blob_torques_new(r_vectors, *args, **kwargs):
  ''' 
  Compute one-blob torques. It returns an array with shape (Nblobs, 3).
  '''
  Nblobs = r_vectors.size // 3
  torque_blobs = np.zeros((Nblobs, 3)) 
  r_vectors = np.reshape(r_vectors, (Nblobs, 3)) 
  
  # Loop over blobs
  for blob in range(Nblobs):
    torque_blobs[blob] += blob_external_torque(r_vectors[blob], *args, **kwargs)   

  return torque_blobs
multi_bodies_functions.calc_one_blob_torques = calc_one_blob_torques_new
