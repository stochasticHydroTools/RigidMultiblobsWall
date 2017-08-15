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
  elif (body.ID == 'Cylinder_N_14_Lg_1_9295_Rg_0_18323') \
       or (body.ID == 'Cylinder_N_86_Lg_1_9384_Rg_0_1484') \
       or (body.ID == 'Cylinder_N_324_Lg_2_0299_Rg_0_1554'):
    body.function_slip = slip_extensile_rod  
  else:
    body.function_slip = default_zero_blobs
  return
multi_bodies_functions.set_slip_by_ID = set_slip_by_ID_new

def slip_extensile_rod(body):
  '''
  Creates slip for a extensile rod. The slip is along the
  axis pointing to the closest end. We assume the blobs
  are equispaced.
  In this version the slip is constant. 
  '''

  # Slip speed
  speed = -20.0
  
  # Identify blobs at the extremities depending on the resolution
  if body.Nblobs == 14:
   Nblobs_covering_ends = 0
   Nlobs_perimeter = 0
  elif body.Nblobs == 86:
   Nblobs_covering_ends = 1
   Nlobs_perimeter = 6 
  elif body.Nblobs == 324:
   Nblobs_covering_ends = 6
   Nlobs_perimeter = 12 
  
  slip = np.empty((body.Nblobs, 3))

  # Get rod orientation
  r_vectors = body.get_r_vectors()
    
  # Compute end-to-end vector  
  if body.Nblobs>14:
    axis = r_vectors[body.Nblobs- 2*Nblobs_covering_ends-2] - r_vectors[Nlobs_perimeter-2]
  else:
    axis = r_vectors[body.Nblobs-1] - r_vectors[0]
  
  length_rod = np.linalg.norm(axis)+2.0*body.blob_radius
  
  # axis = orientation vector
  axis = axis / np.sqrt(np.dot(axis, axis))
  
  # Choose the portion of the surface covered with a tangential slip
  length_covered = 0.8
  lower_bound = length_rod/2.0 - length_covered
  upper_bound = length_rod/2.0

  # Create slip  
  slip_blob = []
  for i in range(body.Nblobs):
    # Blobs at the extremities are passive
    if (Nblobs_covering_ends>0) and (i>=body.Nblobs-2*Nblobs_covering_ends):
      slip_blob = [0., 0., 0.]
    else:
      dist_COM_along_axis = np.dot(r_vectors[i]-body.location, axis)
      if (dist_COM_along_axis >lower_bound) and (dist_COM_along_axis <=upper_bound):
        slip_blob = -speed * axis
      elif (dist_COM_along_axis <-lower_bound) and (dist_COM_along_axis >=-upper_bound):
        slip_blob = speed * axis
      else:
        slip_blob = [0., 0., 0.]        
    slip[i] = slip_blob     
  return slip

