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



def bodies_external_force_torque_new(bodies, r_vectors, *args, **kwargs):
  '''
  This function returns the external force-torques acting on the bodies.
  It returns an array with shape (2*len(bodies), 3)
  
  In this is example we just set it to zero.
  '''
  force_torque_bodies = np.zeros((2*len(bodies), 3))
  ghost_mass = 1.0
  ghost_radius = 0.15
  Nghost = 100
  
  # Loop over bodies
  for k, b in enumerate(bodies):
    # Create force-torque vector
    F = np.zeros((2*len(bodies), 3))

    # Create ghost blobs along the rod axis in the reference configuration
    r_ghost_reference = np.zeros((100, 3))
    r_ghost_reference[:,0] = np.linspace(-1.0, 1.0, r_ghost_reference.shape[0])

    # Get rotation matrix
    rotation_matrix = b.orientation.rotation_matrix()

    def calc_rot_matrix(r_vectors, location, orientation, Nblobs):
      ''' 
      Calculate the matrix R, where the i-th 3x3 block of R gives
      (R_i x) = -1 (r_i cross x).
      R has shape (3*Nblobs, 3).
      '''
      # r_vectors = self.get_r_vectors(location, orientation) - (self.location if location is None else location)    
      r = r_vectors - location
      rot_matrix = np.array([[[0.0,    vec[2], -vec[1]],
                             [-vec[2], 0.0,    vec[0]],
                             [vec[1], -vec[0], 0.0]] for vec in r])
      return np.reshape(rot_matrix, (3*Nblobs, 3))


    # Get ghost blob in the current configuration
    r_ghost = np.array([np.dot(rotation_matrix, vec) for vec in r_ghost_reference])
    r_ghost += b.location

    # Calc rotation matrix
    R = calc_rot_matrix(r_ghost, b.location, b.orientation, Nghost)  
    
    # Compute one-blob forces (same function for all blobs)
    force_blobs = calc_one_blob_forces_ghost(r_ghost, 
                                             blob_radius = ghost_radius, 
                                             blob_mass = ghost_mass,
                                             *args, **kwargs) 
                                             
    # Compute force and torque on the body
    force_torque_bodies[2*k:(2*k+1)] += sum(force_blobs)
    force_torque_bodies[2*k+1:2*k+2] += np.dot(R.T, np.reshape(force_blobs, 3*Nghost))

    # if False:
    #   # Set parameters
    #   k_spring = 4.0
    #   h_eq = 4.199231342006282119e-01
    #   cosTheta_eq = -0.100088053776
    #   # Add harmonic for to keep height constant
    #   force_torque_bodies[2*k,2] += -k_spring * (b.location[2] - h_eq)
    #   # Get rotation matrix
    #   rotation_matrix = b.orientation.rotation_matrix()
    #   # Get axis along rod
    #   e_x = np.array([1.0, 0.0, 0.0])
    #   axis = np.dot(rotation_matrix, e_x)
    #   # Get vector normal to z and axis
    #   e_z = np.array([0.0, 0.0, 1.0])
    #   n = np.cross(e_z, axis) / np.linalg.norm(np.cross(e_z, axis))
    #   # Angle with z-axis
    #   theta = np.arccos(np.dot(e_z, axis))
    #   # Harmonic torque
    #   theta_eq = np.arccos(cosTheta_eq)
    #   magnitude = k_spring * (theta - theta_eq)
    #   force_torque_bodies[2*k + 1] += -magnitude * n
  return force_torque_bodies
multi_bodies_functions.bodies_external_force_torque = bodies_external_force_torque_new


def calc_one_blob_forces_ghost(r_vectors, *args, **kwargs):
  '''
  Compute one-blob forces. It returns an array with shape (Nblobs, 3).
  '''
  Nblobs = r_vectors.size / 3
  force_blobs = np.zeros((Nblobs, 3))
  r_vectors = np.reshape(r_vectors, (Nblobs, 3))
  
  # Loop over blobs
  for blob in range(Nblobs):
    force_blobs[blob] += blob_external_force_ghost(r_vectors[blob], *args, **kwargs)   

  return force_blobs


def blob_external_force_ghost(r_vectors, *args, **kwargs):
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
  if h > (blob_radius + 20 * debye_length_wall):
    pass
  elif h > blob_radius:
    f[2] += (repulsion_strength_wall / debye_length_wall) * np.exp(-(h-blob_radius)/debye_length_wall)
  else:
    f[2] += (repulsion_strength_wall / debye_length_wall)
  return f


def blob_external_force_new(r_vectors, *args, **kwargs):
  return np.zeros(3)
multi_bodies_functions.blob_external_force = blob_external_force_new

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
  if body.ID == 'active_body':
    body.function_slip = active_body_slip
  elif body.ID == 'rod_minimal':
    body.function_slip = partial(slip_minimal, *args, **kwargs)
  elif body.ID == 'rod_resolved':
    body.function_slip = partial(slip_rod_resolved, *args, **kwargs)
  elif body.ID == 'sheet':
    body.function_slip = partial(slip_rod_resolved, *args, **kwargs)
  else:
    body.function_slip = default_zero_blobs
  return
multi_bodies_functions.set_slip_by_ID = set_slip_by_ID_new


def slip_minimal(body, *args, **kwargs):
  '''
  Creates slip for a extensile rod. The slip is along the
  axis pointing to the closest end. There is slip in only the last blob
  '''
  # Get slip options, offset start, offset end and speed
  slip_options = kwargs.get('slip_options')
  offset_start = slip_options[0]
  offset_end = slip_options[1]
  speed = slip_options[2]
  
  # Set slip to zero
  slip = np.zeros((body.Nblobs, 3))
        
  # Get rod orientation
  r_vectors = body.get_r_vectors()
    
  # Compute unit end-to-end vector 
  axis = r_vectors[-1] - r_vectors[0] 
  axis = axis / np.linalg.norm(axis) 
  
  for i in range(body.Nblobs):
    if i >= offset_start and i < offset_end:
      slip[i] = axis * speed
  return slip




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
