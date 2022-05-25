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
import scipy.spatial as spatial

import general_application_utils
from quaternion_integrator.quaternion import Quaternion


def default_zero_r_vectors(r_vectors, *args, **kwargs):
  return np.zeros((r_vectors.size / 3, 3))


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
  

def set_slip_by_ID(body, slip):
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

def put_r_vecs_in_periodic_box(r_vecs, L):
    for r_vec in r_vecs:
        for i in range(3):
            if L[i] > 0:
                while r_vec[i] < 0:
                    r_vec[i] += L[i]
                while r_vec[i] > L[i]:
                    r_vec[i] -= L[i]


# Override blob_external_force 
def blob_external_forces(r_vectors, *args, **kwargs):
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
  Nblobs = r_vectors.size // 3
  f = np.zeros((Nblobs, 3))

  # Get parameters from arguments
  blob_radius = kwargs.get('blob_radius')
  g = kwargs.get('g')
  repulsion_strength_wall = kwargs.get('repulsion_strength_wall')
  debye_length_wall = kwargs.get('debye_length_wall')
  z_max = kwargs.get('z_max')
  domType = kwargs.get('domType')
  
  # Add gravity
  f[:,2] += -(g)

  # Add wall interaction
  h = r_vectors[:,2]
  
  if(domType == 'DPSC'):
    H_chan = (z_max-blob_radius)
    lr_mask = h < H_chan
    sr_mask = h >= H_chan
    f[lr_mask,2] -= (repulsion_strength_wall / debye_length_wall) * np.exp(-(H_chan-h[lr_mask])/debye_length_wall)
    f[sr_mask,2] -= (repulsion_strength_wall / debye_length_wall)
  if((domType == 'DPBW') or (domType == 'RPB') or (domType == 'DPSC')):
    lr_mask = h > blob_radius
    sr_mask = h <= blob_radius
    f[lr_mask,2] += (repulsion_strength_wall / debye_length_wall) * np.exp(-(h[lr_mask]-blob_radius)/debye_length_wall)
    f[sr_mask,2] += (repulsion_strength_wall / debye_length_wall)
  
  return f

# Override blob_blob_force
def blob_blob_force(r, j, k, *args, **kwargs):
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
  #L = kwargs.get('periodic_length')
  #eps = kwargs.get('repulsion_strength')
  #b = kwargs.get('debye_length')
  
  ## Compute force
  #project_to_periodic_image(r, L)
  #r_norm = np.linalg.norm(r)
  #return -((eps / b) + (eps / r_norm)) * np.exp(-r_norm / b) * r / r_norm**2   
    
  # Get parameters from arguments
  L = kwargs.get('periodic_length')
  eps = kwargs.get('repulsion_strength')
  b = kwargs.get('debye_length')
  a = kwargs.get('blob_radius')
  
  project_to_periodic_image(r, L)
  r_norm = np.linalg.norm(r)
  f = 0*r

  # Compute force
  if r_norm > 2*a:
    f += -((eps / b) * np.exp(-(r_norm-2*a) / b) / np.maximum(r_norm, np.finfo(float).eps)) * r 
  else:
    f += -((eps / b) / np.maximum(r_norm, np.finfo(float).eps)) * r
  return f


# Override body_body_force_torque
def body_body_force_torque(r, *args, **kwargs):
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
  #L = kwargs.get('periodic_length')
  #eps = kwargs.get('repulsion_strength_spring')
  #b = 1.0
  ## Compute force
  #project_to_periodic_image(r, L)
  #r_norm = np.linalg.norm(r)
  #force_torque[0] = eps * (r_norm - b) * (r / r_norm) 
  return force_torque

def calc_one_blob_forces(r_vectors, *args, **kwargs):
  '''
  Compute one-blob forces. It returns an array with shape (Nblobs, 3).
  '''
  Nblobs = r_vectors.size // 3
  force_blobs = np.zeros((Nblobs, 3))
  
  # Loop over blobs
  force_blobs = blob_external_forces(r_vectors, *args, **kwargs)
  return force_blobs

def calc_blob_blob_forces(r_vectors, *args, **kwargs):
    '''
    This function computes the blob-blob forces and returns
    an array with shape (Nblobs, 3).
    '''
    Nblobs = round(r_vectors.size / 3)
    force_blobs = np.zeros((Nblobs, 3))

    a = kwargs.get('blob_radius')
    L = kwargs.get('periodic_length')    
    cut_pot = 3.5*a #3.5*a 
    put_r_vecs_in_periodic_box(r_vectors, L)
    r_tree = spatial.cKDTree(r_vectors,boxsize=L)
    

    for j in range(Nblobs):
        s1 = r_vectors[j]
        idx = r_tree.query_ball_point(s1,r=cut_pot) #2.2*a
        idx_trim = [i for i in idx if i > j]
        for k in idx_trim:
            s2 = r_vectors[k]
            r = s2-s1
            force = blob_blob_force(r, j, k, *args, **kwargs)
            force_blobs[j] += force
            force_blobs[k] -= force
    return force_blobs


def calc_body_body_forces_torques(X_0, r_vectors, *args, **kwargs):
  '''
  This function computes the body-body forces and torques and returns
  an array with shape (2*Nblobs, 3).
  '''
  bodies = X_0.reshape((-1,3))
  Nbodies = np.shape(bodies)[0]
  force_torque_bodies = np.zeros((2*Nbodies, 3))
  
  
  # Double loop over bodies to compute forces
  for i in range(Nbodies-1):
    for j in range(i+1, Nbodies):
      # Compute vector from j to i
      r = bodies[j,:] - bodies[i,:]
      force_torque = body_body_force_torque(r, *args, **kwargs)
      # Add forces
      force_torque_bodies[2*i] += force_torque[0]
      force_torque_bodies[2*j] -= force_torque[0]
      # Add torques
      force_torque_bodies[2*i+1] += force_torque[1]
      force_torque_bodies[2*j+1] -= force_torque[1]
      
  return force_torque_bodies

def bodies_external_force_torque(X_0, *args, **kwargs):
  '''
  This function returns the external force-torques acting on the bodies.
  It returns an array with shape (2*len(bodies), 3)
  
  In this is example we just set it to zero.
  '''
  bodies = X_0.reshape((-1,3))
  Nbodies = np.shape(bodies)[0]
  force_torque_bodies = np.zeros((2*Nbodies, 3))
  
  eta = kwargs.get('eta')
  Lrod = kwargs.get('Lrod')
  wall_fact = 0.2
  trq = ((np.pi/3.0)*eta*(Lrod**3))/(1.212*wall_fact)
  
  force_torque_bodies[1::,2] = trq*(2.0*np.pi*1.0)
  
  return force_torque_bodies


def calc_rot_matrix(r_vectors, location, Nblobs):
  ''' 
  Calculate the matrix R, where the i-th 3x3 block of R gives
  (R_i x) = -1 (r_i cross x).
  R has shape (3*Nblobs, 3).
  '''
  r_vecs = r_vectors - location   
  rot_matrix = np.zeros((r_vecs.shape[0], 3, 3))
  rot_matrix[:,0,1] = r_vecs[:,2]
  rot_matrix[:,0,2] = -r_vecs[:,1]
  rot_matrix[:,1,0] = -r_vecs[:,2]
  rot_matrix[:,1,2] = r_vecs[:,0]
  rot_matrix[:,2,0] = r_vecs[:,1]
  rot_matrix[:,2,1] = -r_vecs[:,0]

  return np.reshape(rot_matrix, (3*Nblobs, 3))

def force_torque_calculator_all_bodies(X_0, Quats, r_vectors, *args, **kwargs):
  '''
  Return the forces and torque in each body with
  format [f_1, t_1, f_2, t_2, ...] and shape (2*Nbodies, 3),
  where f_i and t_i are the force and torque on the body i.
  '''
  # Create auxiliar variables
  Nblobs = r_vectors.size // 3
  Nbodies = X_0.size // 3
  
  NperB = round(Nblobs/Nbodies)
  
  r_vectors = np.reshape(r_vectors, (Nblobs, 3))
  
  force_torque_bodies = np.zeros((2*Nbodies, 3))
  force_blobs = np.zeros((Nblobs, 3))

  # Compute one-blob forces (same function for all blobs)
  force_blobs += calc_one_blob_forces(r_vectors, *args, **kwargs)

  # Compute blob-blob forces (same function for all pair of blobs)
  force_blobs += calc_blob_blob_forces(r_vectors, *args, **kwargs)  

  # Compute body force-torque forces from blob forces
  offset = 0
  for k in range(Nbodies):
    # Add force to the body
    force_torque_bodies[2*k:(2*k+1)] += sum(force_blobs[offset:(offset+NperB)])
    #print(force_torque_bodies)
    # Add torque to the body
    R = calc_rot_matrix(r_vectors[offset:(offset+NperB),:], X_0[3*k:3*k+3], NperB)
    force_torque_bodies[2*k+1:2*k+2] = np.dot(R.T, np.reshape(force_blobs[offset:(offset+NperB)], 3*NperB))
    offset += NperB

  # Add one-body external force-torque
  force_torque_bodies += bodies_external_force_torque(X_0, *args, **kwargs)

  # Add body-body forces (same for all pair of bodies)
  force_torque_bodies += calc_body_body_forces_torques(X_0, r_vectors, *args, **kwargs)
  return force_torque_bodies
