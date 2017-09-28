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



def set_slip_by_ID_new(body, slip):
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
  if body.ID == 'active_body':
    body.function_slip = active_body_slip
  elif body.ID == 'rod_minimal':
    body.function_slip = slip_minimal
  elif body.ID == 'rod_resolved':
    body.function_slip = slip_rod_resolved
  else:
    body.function_slip = default_zero_blobs
  return
multi_bodies_functions.set_slip_by_ID = set_slip_by_ID_new



def slip_minimal(body):
  '''
  Creates slip for a extensile rod. The slip is along the
  axis pointing to the closest end. There is slip in only the last blob
  '''
  # Choose number of blobs covered by slip
  num_active_blobs = 16

  # Slip speed
  speed = -1.0
  
  # Set slip to zero
  slip = np.zeros((body.Nblobs, 3))
        
  # Get rod orientation
  r_vectors = body.get_r_vectors()
    
  # Compute unit end-to-end vector 
  axis = r_vectors[-1] - r_vectors[0] 
  axis = axis / np.linalg.norm(axis) 
  
  for i in range(num_active_blobs):
    # if i > 3 and i < 12:
    if i <= 3 or i >= 12:
    # if i > 7:
    # if i <= 7:
    # if True:      
      slip[i] = axis * speed
  return slip




def slip_rod_resolved(body):
  '''
  Creates slip for a extensile rod. The slip is along the
  axis pointing to the closest end. 
  '''
  # Distance to center along x-axis to determine active blobs
  x_distance_to_center = 0.5
  x_distance_to_center_2 = 0.4

  # Speed slip
  speed = -1.0

  # Get rotation matrix
  rotation_matrix = body.orientation.rotation_matrix()

  # Get blobs vectors
  r_reference = body.reference_configuration

  slip_rotated = np.zeros((body.Nblobs, 3))
  for i in range(body.Nblobs):
    # Slip on tail
    # if r_reference[i, 0] < x_distance_to_center:
    #   # Compute slip and rotate
    #   slip_reference = np.array([speed, 0.0, 0.0])
    #   slip_rotated[i] = np.dot(rotation_matrix, slip_reference)

    # Slip on head
    # if r_reference[i, 0] > x_distance_to_center:
    #   # Compute slip and rotate
    #   slip_reference = np.array([speed, 0.0, 0.0])
    #   slip_rotated[i] = np.dot(rotation_matrix, slip_reference)

    # Slip on the middle
    # if abs(r_reference[i, 0]) < x_distance_to_center:
    #   # Compute slip and rotate
    #   slip_reference = np.array([speed, 0.0, 0.0])
    #   slip_rotated[i] = np.dot(rotation_matrix, slip_reference)
    # elif abs(r_reference[i, 0]) < x_distance_to_center_2:
    #   # Compute slip and rotate
    #   slip_reference = np.array([0.5 * speed, 0.0, 0.0])
    #   slip_rotated[i] = np.dot(rotation_matrix, slip_reference)

    # Slip on the sides
    if abs(r_reference[i, 0]) > x_distance_to_center:
      # Compute slip and rotate
      slip_reference = np.array([speed, 0.0, 0.0])
      slip_rotated[i] = np.dot(rotation_matrix, slip_reference)
    elif abs(r_reference[i, 0]) > x_distance_to_center_2:
      # Compute slip and rotate
      slip_reference = np.array([0.5 * speed, 0.0, 0.0])
      slip_rotated[i] = np.dot(rotation_matrix, slip_reference)
    pass

  # slip_rotated[-1,:] = np.array([speed, 0.0, 0.0])
  slip_rotated[-12:,:] = 0

  # print slip_rotated
  return slip_rotated
