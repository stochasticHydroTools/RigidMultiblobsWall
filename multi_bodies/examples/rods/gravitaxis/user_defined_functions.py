'''
Use this module to override forces interactions defined in 
multi_body_functions.py. See an example in the file
RigidMultiblobsWall/multi_bodies/examples/user_defined_functions.py



In this module we override the default blob-blob, blob-wall and
body-body interactions used by the code. To use this implementation 
copy this file to 
RigidMultiblobsWall/multi_bodies/user_defined_functions.py


This module defines (and override) the slip function:

  def set_slip_by_ID_new(body)

and it defines the new slip function slip_extensile_rod, 
see below.
'''

import multi_bodies_functions
from multi_bodies_functions import *


def calc_one_blob_forces_new(r_vectors, *args, **kwargs):
  '''
  Compute one-blob forces. It returns an array with shape (Nblobs, 3).
  '''
  Nblobs = r_vectors.size / 3
  force_blobs = np.zeros((Nblobs, 3))
  r_vectors = np.reshape(r_vectors, (Nblobs, 3))
  mass_options = kwargs.get('slip_options')
  mass = np.ones(Nblobs) * mass_options[0]
  mass[mass_options[1]:mass_options[2]] = mass_options[2]
  mass[mass_options[4]:mass_options[5]] = mass_options[6]

  # Loop over blobs
  for blob in range(Nblobs):
    kwargs['blob_mass'] = mass[blob]
    force_blobs[blob] += blob_external_force(r_vectors[blob], *args, **kwargs)   

  return force_blobs
multi_bodies_functions.calc_one_blob_forces = calc_one_blob_forces_new



def set_slip_by_ID_new(body, slip, *args, **kwargs):
  '''
  This functions assing a slip function to each
  body depending on his ID. The ID of a structure
  is the name of the clones file (without .clones)
  given in the input file.
  As an example we give a default function which sets the
  slip to zero and a function for active rods with an
  slip along its axis. The user can create new functions
  for other kind of active bodies.
  '''
  if 'rod_resolved' in body.ID:
    body.function_slip = partial(slip_rod_resolved, *args, **kwargs)
  else:
    body.function_slip = default_zero_blobs
  return
multi_bodies_functions.set_slip_by_ID = set_slip_by_ID_new


def slip_rod_resolved(body, *args, **kwargs):
  '''
  Creates slip for a extensile rod. The slip is along the
  axis pointing to the closest end. 
  '''
  # Distance to center along x-axis to determine active blobs
  # x_distance_to_center = 64
  # x_distance_to_center_2 = 0.4

  # Get slip options, offset start, offset end and speed
  slip_options = kwargs.get('slip_options')
  offset_start_0 = slip_options[0]
  offset_end_0 = slip_options[1]
  speed = slip_options[2]
  if len(slip_options) >= 3:
    shear = slip_options[3]
  else:
    shear = 0.0
  if len(slip_options) >= 6:
    offset_start_1 = slip_options[4]
    offset_end_1 = slip_options[5]
  else:
    offset_start_1 = 0
    offset_end_1 = 0

  # Get rotation matrix
  rotation_matrix = body.orientation.rotation_matrix()

  # Get blobs vectors
  r_configuration = body.get_r_vectors()
  r_reference = body.reference_configuration

  slip_rotated = np.zeros((body.Nblobs, 3))
  for i in range(body.Nblobs):
    if i >= offset_start_0 and i < offset_end_0:
      # Compute slip and rotate
      slip_reference = np.array([speed, 0.0, 0.0])
      slip_rotated[i] = np.dot(rotation_matrix, slip_reference)
     
    elif i >= offset_start_1 and i < offset_end_1:
      # Compute slip and rotate
      slip_reference = np.array([-speed, 0.0, 0.0])
      slip_rotated[i] = np.dot(rotation_matrix, slip_reference)
      
    # Add shear (flow along x, gradient along z)
    slip_rotated[i,0] -= shear * r_configuration[i,2]


  return slip_rotated
