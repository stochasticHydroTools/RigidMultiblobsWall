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
from __future__ import division, print_function
import multi_bodies_functions
from multi_bodies_functions import *


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

  # Get parameters from arguments
  # blob_mass = kwargs.get('blob_mass')
  blob_radius = kwargs.get('blob_radius')
  # g = kwargs.get('g')
  repulsion_strength_wall = kwargs.get('repulsion_strength_wall') 
  debye_length_wall = kwargs.get('debye_length_wall')
  # alpha = kwargs.get('alpha')
  # Add gravity
  # f += -g * blob_mass * np.array([np.sin(alpha), 0., np.cos(alpha)])

  # Add wall interaction
  h = r_vectors[2]
  if h > blob_radius:
    f[2] += (repulsion_strength_wall / debye_length_wall) * np.exp(-(h-blob_radius)/debye_length_wall)
  else:
    f[2] += (repulsion_strength_wall / debye_length_wall)
  return f
multi_bodies_functions.blob_external_force = blob_external_force_new


def calc_one_blob_forces_new(r_vectors, *args, **kwargs):
  '''
  Compute one-blob forces. It returns an array with shape (Nblobs, 3).
  '''
  Nblobs = r_vectors.size // 3
  force_blobs = np.zeros((Nblobs, 3))
  r_vectors = np.reshape(r_vectors, (Nblobs, 3))
  mass_options = kwargs.get('mass_options')
  mass = np.ones(Nblobs) * mass_options[0]
  mass[int(mass_options[1]):int(mass_options[2])] = mass_options[3]
  mass[int(mass_options[4]):int(mass_options[5])] = mass_options[6]
  alpha = mass_options[7]

  # Loop over blobs
  for blob in range(Nblobs):
    kwargs['blob_mass'] = mass[blob]
    force_blobs[blob] += multi_bodies_functions.blob_external_force(r_vectors[blob], alpha=alpha, *args, **kwargs)   

  return force_blobs
# multi_bodies_functions.calc_one_blob_forces = calc_one_blob_forces_new


def bodies_external_force_torque_new(bodies, r_vectors, *args, **kwargs):
  '''
  The force is F = m * g * (sin(alpha), 0, cos(alpha))
  m = 1.0
  g = from input file, adjusted to have the correct m*g

  The torque is T = r' \cross F
  with r' = rotation_matrix * r_from_geometric_center_to_center_of_mass

  r_from_geometric_center_to_center_of_mass = from input file, for homogeneus rods is (0,0,0)
  '''
  force_torque = np.zeros((2*len(bodies), 3))

  # Get parameters
  g = kwargs.get('g')
  r_gc_to_com = kwargs.get('r_gc_to_com')
  alpha = kwargs.get('alpha')

  # Compute force
  F = -g * np.array([np.sin(alpha), 0, np.cos(alpha)])
  print('F = ', F)
  force_torque[0::2,:] = F

  # Rotate r_gc_to_com and compute torque
  for k, b in enumerate(bodies):
    rotation_matrix = b.orientation.rotation_matrix()
    r = np.dot(rotation_matrix, r_gc_to_com)
    force_torque[2*k+1,:] = np.cross(r, F)
  return force_torque
multi_bodies_functions.bodies_external_force_torque = bodies_external_force_torque_new


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
