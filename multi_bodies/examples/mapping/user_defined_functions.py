import numpy as np
import scipy.linalg as scla
from mobility import mobility as mob
import general_application_utils as utils
import multi_bodies_functions
from multi_bodies_functions import *

def calc_blob_forces_swimmer(slip, N, K, L, lower):
  ''' 
  This function works for phoretic swimmers. 
  '''
  lambda_tilde = scla.cho_solve((L, lower), slip.flatten(), check_finite=False)
  u_hat = np.dot(K, np.dot(N, np.dot(K.T, lambda_tilde)))
  lambda_hat = scla.cho_solve((L, lower), u_hat, check_finite=False)
  lambda_blobs = lambda_tilde - lambda_hat  
  return lambda_blobs

def calc_velocity_swimmer(slip, N, K, L, lower):
  ''' 
  This function works for phoretic swimmers.
  '''
  U = -np.dot(N, np.dot(K.T, scla.cho_solve((L, lower), slip.flatten(), check_finite=False)))
  return U


def gauss_weights(N):
  '''
  Compute Legendre points and weights for Gauss quadrature.
  From Spectral Methods in MATLAB, Trefethen 2000, Program Gauss.m (page 129).
  '''
  s = np.arange(1, N)
  beta = 0.5 / np.sqrt(1.0 - 1.0 / (2 * s)**2)
  T = np.diag(beta, k=1) + np.diag(beta, k=-1)
  eig_values, eig_vectors = np.linalg.eigh(T)
  w = 2 * eig_vectors[0,:]**2
  return eig_values, w


def parametrization(p):
  '''
  Set parametrization, (u,v), with p+1 points along u and (2*p+2) along v.
  In total 2*p**2 points because at the poles we only have one point.

  Return parametrization and weights.
  '''
  # Precomputation
  Nu = p + 1
  Nv = 2 * (p + 1)
  N = Nu * Nv
  t, w_gauss = gauss_weights(Nu)
  u = np.arccos(t)
  v = np.linspace(0, 2*np.pi, Nv, endpoint=False)
  uu, vv = np.meshgrid(u, v, indexing = 'ij')

  # Parametrization
  uv = np.zeros((N,2))
  uv[:,0] = uu.flatten()
  uv[:,1] = vv.flatten()

  # Weights precomputation
  uw = w_gauss # / np.sin(u)
  vw = np.ones(v.size) * 2 * np.pi / Nv  
  uuw, vvw = np.meshgrid(uw, vw, indexing = 'ij')
  
  # Weights
  w = uuw.flatten() * vvw.flatten()
  
  return uv, w


def sphere(a, uv):
  '''
  Return the points on a sphere of radius a parametrized by uv.
  '''
  # Generate coordinates
  x = np.zeros((uv.shape[0], 3))
  x[:,0] = a * np.cos(uv[:,1]) * np.sin(uv[:,0])
  x[:,1] = a * np.sin(uv[:,1]) * np.sin(uv[:,0])
  x[:,2] = a * np.cos(uv[:,0]) 
  return x


def plot_velocity_field_shell(r_vectors, lambda_blobs, blob_radius, eta, sphere_radius, p, output, radius_source=None, frame_body=None, *args, **kwargs):
  '''
  This function plots the velocity field to a Chebyshev-Fourier spherical grid of radius "sphere_radius".
  The grid is defined in the body frame of reference of body "frame_body".
  If frame_body < 0 the grid is defined in the laboratory frame of reference.
  '''

  # Create sphere
  uv, uv_weights = parametrization(p)
  grid_coor_ref = sphere(sphere_radius, uv)

  # Transform grid to the body frame of reference
  if frame_body is not None:
    grid_coor = utils.get_vectors_frame_body(grid_coor_ref, body=frame_body)
  else:
    grid_coor = grid_coor_ref
    
  # Set radius of blobs (= a) and grid nodes (= 0)
  if radius_source is None:
    radius_source = np.ones(r_vectors.size // 3) * blob_radius
  radius_target = np.zeros(grid_coor.size // 3) 
  
  # Compute velocity field 
  mobility_vector_prod_implementation = kwargs.get('mobility_vector_prod_implementation')
  if mobility_vector_prod_implementation == 'python':
    grid_velocity = mob.mobility_vector_product_source_target_one_wall(r_vectors, 
                                                                       grid_coor, 
                                                                       lambda_blobs, 
                                                                       radius_source, 
                                                                       radius_target, 
                                                                       eta, 
                                                                       *args, 
                                                                       **kwargs) 
  elif mobility_vector_prod_implementation == 'C++':
    grid_velocity = mob.boosted_mobility_vector_product_source_target(r_vectors, 
                                                                      grid_coor, 
                                                                      lambda_blobs, 
                                                                      radius_source, 
                                                                      radius_target, 
                                                                      eta, 
                                                                      *args, 
                                                                      **kwargs)
  elif mobility_vector_prod_implementation == 'numba_no_wall':
    grid_velocity = mob.no_wall_mobility_trans_times_force_source_target_numba(r_vectors, 
                                                                               grid_coor, 
                                                                               lambda_blobs, 
                                                                               radius_source, 
                                                                               radius_target, 
                                                                               eta, 
                                                                               *args, 
                                                                               **kwargs) 
  else:
    grid_velocity = mob.single_wall_mobility_trans_times_force_source_target_pycuda(r_vectors, 
                                                                                    grid_coor, 
                                                                                    lambda_blobs, 
                                                                                    radius_source, 
                                                                                    radius_target, 
                                                                                    eta, 
                                                                                    *args, 
                                                                                    **kwargs)
    
  # Tranform velocity to the body frame of reference
  if frame_body is not None:
    grid_velocity = utils.get_vectors_frame_body(grid_velocity, body=frame_body, translate=False, transpose=True)
 
  # Write velocity field.
  header = 'R=' + str(sphere_radius) + ', p=' + str(p) + ', N=' + str(uv_weights.size) + ', centered body=' + str(frame_body) + ', 7 Columns: grid point (x,y,z), quadrature weight, velocity (vx,vy,vz)'
  result = np.zeros((grid_coor.shape[0], 7))
  result[:,0:3] = grid_coor_ref
  result[:,3] = uv_weights
  grid_velocity = grid_velocity.reshape((grid_velocity.size // 3, 3)) 
  result[:,4:] = grid_velocity
  np.savetxt(output, result, header=header) 
  return
multi_bodies_functions.plot_velocity_field_shell = plot_velocity_field_shell


def calc_body_body_forces_torques_python_new(bodies, r_vectors, *args, **kwargs):
  '''
  Apply constant torque in the body frame of reference to a bacteria body and
  flagellum. The total torque is zero.

  This torque only applies to bodies with body ID "bacteria_constant_torque".
  '''
  Nbodies = len(bodies)
  force_torque_bodies = np.zeros((2*len(bodies), 3))

  # Get constant torque in the body frame of reference
  torque = kwargs.get('omega_one_roller')
  
  # Loop over bodies and apply torque in the laboratory frame of reference
  constant_torque_counter = 0
  for i in range(Nbodies):
    if bodies[i].ID == 'bacteria_constant_torque':
      rotation_matrix = bodies[i].orientation.rotation_matrix()
      if constant_torque_counter == 0:
        force_torque_bodies[2*i + 1] = np.dot(rotation_matrix, torque)
        constant_torque_counter = 1
      else:
        force_torque_bodies[2*i + 1] = -np.dot(rotation_matrix, torque)
        constant_torque_counter = 0
  return force_torque_bodies
multi_bodies_functions.calc_body_body_forces_torques_python = calc_body_body_forces_torques_python_new
