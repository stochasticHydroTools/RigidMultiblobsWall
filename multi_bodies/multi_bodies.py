
import argparse
import numpy as np
import scipy.linalg
import subprocess
from shutil import copyfile
from functools import partial
import sys
import time
try:
  import pickle as cpickle
except:
  try:
    import cpickle
  except:
    import _pickle as cpickle

# Add path to HydroGrid and import module
# sys.path.append('../../HydroGrid/src/')


# Find project functions
found_functions = False
path_to_append = ''
while found_functions is False:
  try:
    import multi_bodies_functions
    from mobility import mobility as mb
    from quaternion_integrator.quaternion import Quaternion
    from quaternion_integrator.quaternion_integrator_multi_bodies import QuaternionIntegrator
    from quaternion_integrator.quaternion_integrator_rollers import QuaternionIntegratorRollers
    from body import body 
    from read_input import read_input
    from read_input import read_vertex_file
    from read_input import read_clones_file
    from read_input import read_slip_file
    from read_input import read_velocity_file
    from read_input import read_constraints_file
    from read_input import read_vertex_file_list      
    from constraint.constraint import Constraint
    from articulated.articulated import Articulated
    import general_application_utils as utils
    try:
      import libCallHydroGrid as cc
      found_HydroGrid = True
    except ImportError:
      found_HydroGrid = False
    found_functions = True
  except ImportError as exc:
    sys.stderr.write('Error: failed to import settings module ({})\n'.format(exc))
    path_to_append += '../'
    print('searching functions in path ', path_to_append)
    sys.path.append(path_to_append)
    if len(path_to_append) > 21:
      print('\nProjected functions not found. Edit path in multi_bodies.py')
      sys.exit()
def calc_slip(bodies, Nblobs, *args, **kwargs):
  '''
  Function to calculate the slip in all the blobs.
  '''
  slip = np.zeros((Nblobs, 3))
  a = kwargs.get('blob_radius')
  eta = kwargs.get('eta')
  g = kwargs.get('g')
  r_vectors = get_blobs_r_vectors(bodies, Nblobs)

  #1) Compute slip due to external torques on bodies with single blobs only
  torque_blobs = multi_bodies_functions.calc_one_blob_torques(r_vectors, blob_radius = a, g = g) 

  if np.amax(np.absolute(torque_blobs))>0:
    implementation = kwargs.get('implementation')
    offset = 0
    for b in bodies:
      if b.Nblobs>1:
        torque_blobs[offset:offset+b.Nblobs] = 0.0  
      offset += b.Nblobs
    if implementation == 'pycuda':
      slip_blobs = mb.single_wall_mobility_trans_times_torque_pycuda(r_vectors, torque_blobs, eta, a) 
    elif implementation == 'pycuda_no_wall':
      slip_blobs = mb.no_wall_mobility_trans_times_torque_pycuda(r_vectors, torque_blobs, eta, a) 
    slip = np.reshape(-slip_blobs, (Nblobs, 3) ) 
 
  #2) Add prescribed slip 
  offset = 0
  for b in bodies:
    slip_b = b.calc_slip()
    slip[offset:offset+b.Nblobs] += slip_b
    offset += b.Nblobs
  return slip


def get_blobs_r_vectors(bodies, Nblobs):
  '''
  Return coordinates of all the blobs with shape (Nblobs, 3).
  '''
  r_vectors = np.empty((Nblobs, 3))
  offset = 0
  for b in bodies:
    num_blobs = b.Nblobs
    r_vectors[offset:(offset+num_blobs)] = b.get_r_vectors()
    offset += num_blobs
  return r_vectors


def set_mobility_blobs(implementation):
  '''
  Set the function to compute the dense mobility
  at the blob level to the right implementation.
  The implementation in C++ is somewhat faster than 
  the python one; to use it the user should compile 
  the file mobility/mobility.cpp

  These functions return an array with shape 
  (3*Nblobs, 3*Nblobs).
  '''
  # Implementations without wall
  if implementation == 'python_no_wall':
    return mb.rotne_prager_tensor
  if implementation == 'C++_no_wall':
    return mb.rotne_prager_tensor_cpp
  # Implementations with wall
  elif implementation == 'python':
    return mb.single_wall_fluid_mobility
  elif implementation == 'C++':
    return mb.single_wall_fluid_mobility_cpp
  # Implementation free surface
  elif implementation == 'C++_free_surface':
    return  mb.boosted_free_surface_mobility


def set_mobility_vector_prod(implementation, *args, **kwargs):
  '''
  Set the function to compute the matrix-vector
  product (M*F) with the mobility defined at the blob 
  level to the right implementation.
  
  The implementations in numba, pycuda and C++ are much faster than the
  python implementation. 
  Depending on the computer the fastest implementation will be the C++ or the pycuda codes.
  To use the pycuda implementation is necessary to have installed pycuda and a GPU with CUDA capabilities. 
  To use the C++ implementation the user has to compile the file mobility/mobility.cpp.  
  ''' 
  # Implementations without wall
  if implementation == 'python_no_wall':
    return mb.no_wall_fluid_mobility_product
  elif implementation == 'pycuda_no_wall':
    return mb.no_wall_mobility_trans_times_force_pycuda
  elif implementation == 'numba_no_wall':
    return mb.no_wall_mobility_trans_times_force_numba
  # Implementations with wall
  elif implementation == 'python':
    return mb.single_wall_fluid_mobility_product
  elif implementation == 'C++':
    return mb.single_wall_mobility_trans_times_force_cpp
  elif implementation == 'pycuda':
    return mb.single_wall_mobility_trans_times_force_pycuda
  elif implementation == 'numba':
    return mb.single_wall_mobility_trans_times_force_numba
  # Implementations free surface
  elif implementation == 'pycuda_free_surface':
    return mb.free_surface_mobility_trans_times_force_pycuda
  # Implementations different radii
  elif implementation.find('radii') > -1:
    # Get right function
    if implementation == 'radii_numba':
      function = mb.single_wall_mobility_trans_times_force_source_target_numba
    elif implementation == 'radii_numba_no_wall':
      function = mb.no_wall_mobility_trans_times_force_source_target_numba
    elif implementation == 'radii_pycuda':
      function = mb.single_wall_mobility_trans_times_force_source_target_pycuda
    elif implementation == 'radii':
      function = mb.mobility_vector_product_source_target_one_wall
    elif implementation == 'radii_no_wall':
      function = mb.mobility_vector_product_source_target_unbounded
    # Get blobs radii
    bodies = kwargs.get('bodies')
    radius_blobs = []
    for k, b in enumerate(bodies):
      radius_blobs.append(b.blobs_radius)
    radius_blobs = np.concatenate(radius_blobs, axis=0)    
    return partial(mb.mobility_radii_trans_times_force, radius_blobs=radius_blobs, function=function)


def calc_K_matrix(bodies, Nblobs):
  '''
  Calculate the geometric block-diagonal matrix K.
  Shape (3*Nblobs, 6*Nbodies).
  '''
  K = np.zeros((3*Nblobs, 6*len(bodies)))
  offset = 0
  for k, b in enumerate(bodies):
    K_body = b.calc_K_matrix()
    K[3*offset:3*(offset+b.Nblobs), 6*k:6*k+6] = K_body
    offset += b.Nblobs
  return K


def calc_K_matrix_bodies(bodies, Nblobs):
  '''
  Calculate the geometric matrix K for
  each body. List of shape (3*Nblobs, 6*Nbodies).
  '''
  K = []
  for k, b in enumerate(bodies):
    K_body = b.calc_K_matrix()
    K.append(K_body)
  return K


def calc_C_matrix_constraints(constraints):
  '''
  Calculate the geometric matrix C for
  each constraint. List of shape (3*Nconstraints, 6*Nbodies).
  '''
  C = []
  for k, c in enumerate(constraints):
    C_constraint = c.calc_C_matrix()
    C.append(C_constraint)
  return C


def K_matrix_vector_prod(bodies, vector, Nblobs, K_bodies = None):
  '''
  Compute the matrix vector product K*vector where
  K is the geometrix matrix that transport the information from the 
  level of describtion of the body to the level of describtion of the blobs.
  ''' 
  # Prepare variables
  result = np.empty((Nblobs, 3))
  v = np.reshape(vector, (len(bodies) * 6))

  # Loop over bodies
  offset = 0
  for k, b in enumerate(bodies):
    if K_bodies is None:
      K = b.calc_K_matrix()
    else:
      K = K_bodies[k] 
    result[offset : offset+b.Nblobs] = np.reshape(np.dot(K, v[6*k : 6*(k+1)]), (b.Nblobs, 3))
    offset += b.Nblobs    
  return result


def K_matrix_T_vector_prod(bodies, vector, Nblobs, K_bodies = None):
  '''
  Compute the matrix vector product K^T*vector where
  K is the geometrix matrix that transport the information from the 
  level of describtion of the body to the level of describtion of the blobs.
  ''' 
  # Prepare variables
  result = np.empty((len(bodies), 6))
  v = np.reshape(vector, (Nblobs * 3))

  # Loop over bodies
  offset = 0
  for k, b in enumerate(bodies):
    if K_bodies is None:
      K = b.calc_K_matrix()
    else:
      K = K_bodies[k] 
    result[k : k+1] = np.dot(K.T, v[3*offset : 3*(offset+b.Nblobs)])
    offset += b.Nblobs    

  result = np.reshape(result, (2*len(bodies), 3))
  return result

def C_matrix_vector_prod(bodies, constraints, vector, Nconstraints, C_constraints = None):
  '''
  Compute the matrix vector product C*vector where
  C is the Jacobian of the velocity constraints.
  ''' 
  # Prepare variables
  result = np.empty((Nconstraints, 3))
  v = np.reshape(vector, (len(bodies) * 6))

  # Loop over bodies
  for k, c in enumerate(constraints):
    if C_constraints is None:
      C = c.calc_C_matrix()
    else:
      C = C_constraints[k] 
    
    ind1 = c.ind_bodies[0]
    vbody1 = v[6*ind1 : 6*(ind1+1)]  
    ind2 = c.ind_bodies[1]
    vbody2 = v[6*ind2 : 6*(ind2+1)] 
    result[k] = np.dot(C, np.concatenate([vbody1, vbody2], axis = 0))
  return result


def C_matrix_T_vector_prod(bodies, constraints, vector, Nconstraints, C_constraints = None):
  '''
  Compute the matrix vector product C^T*vector where
  C is the Jacobian of the velocity constraints.
  ''' 
  # Prepare variables
  result = np.zeros((len(bodies), 6))
  v = np.reshape(vector, (Nconstraints * 3))

  # Loop over bodies
  for k, c in enumerate(constraints):
    if C_constraints is None:
      C = c.calc_C_matrix()
    else:
      C = C_constraints[k] 
    
    ind1 = c.ind_bodies[0]
    ind2 = c.ind_bodies[1]
    C1 = C[:,0:6]
    C2 = C[:,6:12]
    result[ind1] += np.dot(C1.T, v[3*k:3*(k+1)])
    result[ind2] += np.dot(C2.T, v[3*k:3*(k+1)])

  result = np.reshape(result, (2*len(bodies), 3))
  return result


def linear_operator_rigid(vector, bodies, constraints, r_vectors, eta, a, K_bodies = None, C_constraints = None, *args, **kwargs):
  '''
  RetC_matrix_vector_produrn the action of the linear operator of the articulated rigid bodies on vector v.
  The linear operator is
  |  M   -K  0  ||lambda| = | slip + noise_1|
  | -K^T  0  C^T||  U   |   | -F   + noise_2|
  |  0    C  0  || phi  |   |  B            |
  ''' 
  # Reserve memory for the solution and create some variables
  L = kwargs.get('periodic_length')
  Ncomp_blobs = r_vectors.size
  Nblobs = r_vectors.size // 3
  Nbodies = len(bodies)
  Nconstraints = len(constraints)
  Ncomp_bodies = 6 * Nbodies
  Ncomp_phi = 3 * Nconstraints
  Ncomp_tot = Ncomp_blobs + Ncomp_bodies + Ncomp_phi
  res = np.empty((Ncomp_tot))
  v = np.reshape(vector, (vector.size//3, 3))
  
  # Compute the "slip" part
  res[0:Ncomp_blobs] = mobility_vector_prod(r_vectors, vector[0:Ncomp_blobs], eta, a, *args, **kwargs) 
  K_times_U = K_matrix_vector_prod(bodies, v[Nblobs : Nblobs+2*Nbodies], Nblobs, K_bodies = K_bodies) 
  res[0:Ncomp_blobs] -= np.reshape(K_times_U , (3*Nblobs))

  # Compute the "-force_torque" part
  K_T_times_lambda = K_matrix_T_vector_prod(bodies, vector[0:Ncomp_blobs], Nblobs, K_bodies = K_bodies)
  # Add constraint forces if any
  if Nconstraints > 0:
    C_T_times_phi = C_matrix_T_vector_prod(bodies, constraints, vector[Ncomp_blobs + Ncomp_bodies:Ncomp_tot], Nconstraints, C_constraints = C_constraints)
    res[Ncomp_blobs : Ncomp_blobs+Ncomp_bodies] = np.reshape(-K_T_times_lambda + C_T_times_phi, (Ncomp_bodies))
  else:
    res[Ncomp_blobs : Ncomp_blobs+Ncomp_bodies] = np.reshape(-K_T_times_lambda, (Ncomp_bodies))

  # Modify to account for prescribed kinematics
  offset = 0
  for k, b in enumerate(bodies):
    if b.prescribed_kinematics is True:
      res[3*offset : 3*(offset+b.Nblobs)] += (K_times_U[offset : (offset+b.Nblobs)]).flatten()
      res[Ncomp_blobs + k*6: Ncomp_blobs + (k+1)*6] += vector[Ncomp_blobs + k*6: Ncomp_blobs + (k+1)*6]
    offset += b.Nblobs

  # Compute the "constraint velocity: B" part if any
  if Nconstraints > 0:
    C_times_U = C_matrix_vector_prod(bodies, constraints, v[Nblobs:Nblobs+2*Nbodies], Nconstraints, C_constraints = C_constraints)
    res[Ncomp_blobs+Ncomp_bodies:Ncomp_tot] = np.reshape(C_times_U , (Ncomp_phi))
    
  return res


@utils.static_var('initialized', [])
@utils.static_var('mobility_bodies', [])
@utils.static_var('K_bodies', [])
@utils.static_var('M_factorization_blobs', [])
@utils.static_var('M_factorization_blobs_inv', [])
@utils.static_var('mobility_inv_blobs', [])
def build_block_diagonal_preconditioners_det_stoch(bodies, r_vectors, Nblobs, eta, a, *args, **kwargs):
  '''
  Build the deterministic and stochastic block diagonal preconditioners for rigid bodies.
  It solves exactly the mobility problem for each body
  independently, i.e., no interation between bodies is taken
  into account.

  If the mobility of a body at the blob
  level is M=L^T * L with L the Cholesky factor  we form the stochastic preconditioners
  
  P = inv(L)
  P_inv = L

  and the deterministic preconditioner
  N = (K.T * M^{-1} * K)^{-1}
  
  and return the functions to compute matrix vector products
  y = (P.T * M * P) * x
  y = P_inv * x
  y = N*F - N*K.T*M^{-1}*slip
  '''
  initialized = build_block_diagonal_preconditioners_det_stoch.initialized
  mobility_bodies = []
  K_bodies = []
  M_factorization_blobs = []
  M_factorization_blobs_inv = []
  mobility_inv_blobs = []

  if(kwargs.get('step') % kwargs.get('update_PC') == 0) or len(build_block_diagonal_preconditioners_det_stoch.mobility_bodies) == 0:
    # Loop over bodies
    for k, b in enumerate(bodies):
      if (b.prescribed_kinematics or b.Nblobs == 1) and len(initialized) > 0:
        mobility_bodies.append(build_block_diagonal_preconditioners_det_stoch.mobility_bodies[k])
        K_bodies.append(build_block_diagonal_preconditioners_det_stoch.K_bodies[k])
        M_factorization_blobs.append(build_block_diagonal_preconditioners_det_stoch.M_factorization_blobs[k])
        M_factorization_blobs_inv.append(build_block_diagonal_preconditioners_det_stoch.M_factorization_blobs_inv[k])
        mobility_inv_blobs.append(build_block_diagonal_preconditioners_det_stoch.mobility_inv_blobs[k])
      else:
        # 1. Compute blobs mobility 
        M = b.calc_mobility_blobs(eta, a)
        # 2. Compute Cholesy factorization, M = L^T * L
        L, lower = scipy.linalg.cho_factor(M)
        L = np.triu(L)   
        M_factorization_blobs.append(L.T)
        # 3. Compute inverse of L
        M_factorization_blobs_inv.append(scipy.linalg.solve_triangular(L, np.eye(b.Nblobs * 3), check_finite=False))
        # 4. Compute inverse mobility blobs
        mobility_inv_blobs.append(scipy.linalg.solve_triangular(L, scipy.linalg.solve_triangular(L, np.eye(b.Nblobs * 3), trans='T', check_finite=False), check_finite=False))
        # 5. Compute geometric matrix K
        K = b.calc_K_matrix()
        K_bodies.append(K)
        # 6. Compute body mobility
        mobility_bodies.append(np.linalg.pinv(np.dot(K.T, scipy.linalg.cho_solve((L,lower), K, check_finite=False))))

    # Save variables to use in next steps if PC is not updated
    build_block_diagonal_preconditioners_det_stoch.mobility_bodies = mobility_bodies
    build_block_diagonal_preconditioners_det_stoch.K_bodies = K_bodies
    build_block_diagonal_preconditioners_det_stoch.M_factorization_blobs = M_factorization_blobs
    build_block_diagonal_preconditioners_det_stoch.M_factorization_blobs_inv = M_factorization_blobs_inv
    build_block_diagonal_preconditioners_det_stoch.mobility_inv_blobs = mobility_inv_blobs

    # The function is initialized
    build_block_diagonal_preconditioners_det_stoch.initialized.append(1)
  else:
    # Use old values
    mobility_bodies = build_block_diagonal_preconditioners_det_stoch.mobility_bodies 
    K_bodies = build_block_diagonal_preconditioners_det_stoch.K_bodies
    M_factorization_blobs = build_block_diagonal_preconditioners_det_stoch.M_factorization_blobs 
    M_factorization_blobs_inv = build_block_diagonal_preconditioners_det_stoch.M_factorization_blobs_inv 
    mobility_inv_blobs = build_block_diagonal_preconditioners_det_stoch.mobility_inv_blobs 
    

  def block_diagonal_preconditioner(vector, bodies = None, mobility_bodies = None, mobility_inv_blobs = None, K_bodies = None, Nblobs = None, *args, **kwargs):
    '''
    Apply the block diagonal preconditioner.
    '''
    result = np.empty(vector.shape)
    offset = 0
    for k, b in enumerate(bodies):
      if b.prescribed_kinematics is False:
        # 1. Solve M*Lambda_tilde = slip
        slip = vector[3*offset : 3*(offset + b.Nblobs)]
        Lambda_tilde = np.dot(mobility_inv_blobs[k], slip)
        # 2. Compute rigid body velocity
        F = vector[3*Nblobs + 6*k : 3*Nblobs + 6*(k+1)]
        Y = np.dot(mobility_bodies[k], -F - np.dot(K_bodies[k].T, Lambda_tilde))
        # 3. Solve M*Lambda = (slip + K*Y)
        result[3*offset : 3*(offset + b.Nblobs)] = np.dot(mobility_inv_blobs[k], slip + np.dot(K_bodies[k], Y))
        # 4. Set result
        result[3*Nblobs + 6*k : 3*Nblobs + 6*(k+1)] = Y
      if b.prescribed_kinematics is True:
        # 1. Solve M*Lambda = (slip + K*Y)
        slip_KU = vector[3*offset : 3*(offset + b.Nblobs)]
        Lambda = np.dot(mobility_inv_blobs[k], slip_KU)

        # 2. Set force
        F = np.dot(K_bodies[k].T, Lambda)

        # 3. Set result
        result[3*offset : 3*(offset + b.Nblobs)] = Lambda
        result[3*Nblobs + 6*k : 3*Nblobs + 6*(k+1)] = F
      offset += b.Nblobs
    return result
  block_diagonal_preconditioner_partial = partial(block_diagonal_preconditioner, 
                                                  bodies = bodies, 
                                                  mobility_bodies = mobility_bodies, 
                                                  mobility_inv_blobs = mobility_inv_blobs,
                                                  K_bodies = K_bodies,
                                                  Nblobs = Nblobs)

  # Define preconditioned mobility matrix product
  def mobility_pc(w, bodies = None, P = None, r_vectors = None, eta = None, a = None, *args, **kwargs):
    result = np.empty_like(w)
    # Apply P
    offset = 0
    for k, b in enumerate(bodies):
      result[3*offset : 3*(offset + b.Nblobs)] = np.dot(P[k], w[3*offset : 3*(offset + b.Nblobs)]) 
      offset += b.Nblobs
    # Multiply by M
    result_2 = mobility_vector_prod(r_vectors, result, eta, a, *args, **kwargs)
    # Apply P.T
    offset = 0
    for k, b in enumerate(bodies):
      result[3*offset : 3*(offset + b.Nblobs)] = np.dot(P[k].T, result_2[3*offset : 3*(offset + b.Nblobs)])
      offset += b.Nblobs
    return result
  mobility_pc_partial = partial(mobility_pc, bodies = bodies, P = M_factorization_blobs_inv, r_vectors = r_vectors, eta = eta, a = a, *args, **kwargs)
  
  # Define inverse preconditioner P_inv
  def P_inv_mult(w, bodies = None, P_inv = None):
    offset = 0
    for k, b in enumerate(bodies):
      w[3*offset : 3*(offset + b.Nblobs)] = np.dot(P_inv[k], w[3*offset : 3*(offset + b.Nblobs)])
      offset += b.Nblobs
    return w
  P_inv_mult_partial = partial(P_inv_mult, bodies = bodies, P_inv = M_factorization_blobs)

  # Return preconditioner functions
  return block_diagonal_preconditioner_partial, mobility_pc_partial, P_inv_mult_partial

@utils.static_var('mobility_bodies', [])
@utils.static_var('mobility_bodies_identity', [])
@utils.static_var('K_bodies', [])
@utils.static_var('M_factorization_blobs', [])
@utils.static_var('M_factorization_blobs_inv', [])
@utils.static_var('mobility_inv_blobs', [])
def build_block_diagonal_preconditioners_det_identity_stoch(bodies, r_vectors, Nblobs, eta, a, *args, **kwargs):
  '''
  Build the deterministic and stochastic block diagonal preconditioners for rigid bodies.
  It solves exactly the deterministic unconstrained mobility problem for each body
  independently with M=I, i.e., no interaction between bodies are taken
  into account.
  the deterministic preconditioner is
  N = (K.T * K)^{-1}

  For the blob Brownian velocities, we consider hydrodynamic interactions
  If the mobility of a body at the blob
  level is M=L^T * L with L the Cholesky factor  we form the stochastic preconditioners
  
  P = inv(L)
  P_inv = L
  
  It returns the functions to compute matrix vector products
  y = (P.T * M * P) * x
  y = P_inv * x
  y = N*F - N*K.T*slip
  '''
  mobility_bodies = []
  mobility_bodies_identity = []
  K_bodies = []
  M_factorization_blobs = []
  M_factorization_blobs_inv = []
  mobility_inv_blobs = []

  if(kwargs.get('step') % kwargs.get('update_PC') == 0) or len(build_block_diagonal_preconditioners_det_identity_stoch.mobility_bodies) == 0:
    # Loop over bodies
    for b in bodies:
      # 1. Compute blobs mobility 
      M = b.calc_mobility_blobs(eta, a)
      # 2. Compute Cholesy factorization, M = L^T * L
      L, lower = scipy.linalg.cho_factor(M)
      L = np.triu(L)  
      M_factorization_blobs.append(L.T)
      # 3. Compute inverse of L
      M_factorization_blobs_inv.append(scipy.linalg.solve_triangular(L, np.eye(b.Nblobs * 3), check_finite=False))
      # 4. Compute inverse mobility blobs
      mobility_inv_blobs.append(scipy.linalg.solve_triangular(L, scipy.linalg.solve_triangular(L, np.eye(b.Nblobs * 3), trans='T', check_finite=False), check_finite=False))
      # 5. Compute geometric matrix K
      K = b.calc_K_matrix()
      K_bodies.append(K)
      # 6. Compute body mobility
      mobility_bodies.append(np.linalg.pinv(np.dot(K.T, scipy.linalg.cho_solve((L,lower), K, check_finite=False))))
      # 7. Compute body mobility with M=I
      mobility_bodies_identity.append(np.linalg.pinv(np.dot(K.T, K)))

    # Save variables to use in next steps if PC is not updated
    build_block_diagonal_preconditioners_det_identity_stoch.mobility_bodies = mobility_bodies
    build_block_diagonal_preconditioners_det_identity_stoch.mobility_bodies_identity = mobility_bodies_identity
    build_block_diagonal_preconditioners_det_identity_stoch.K_bodies = K_bodies
    build_block_diagonal_preconditioners_det_identity_stoch.M_factorization_blobs = M_factorization_blobs
    build_block_diagonal_preconditioners_det_identity_stoch.M_factorization_blobs_inv = M_factorization_blobs_inv
    build_block_diagonal_preconditioners_det_identity_stoch.mobility_inv_blobs = mobility_inv_blobs
  else:
    # Use old values
    mobility_bodies = build_block_diagonal_preconditioners_det_identity_stoch.mobility_bodies 
    mobility_bodies_identity = build_block_diagonal_preconditioners_det_identity_stoch.mobility_bodies_identity 
    K_bodies = build_block_diagonal_preconditioners_det_identity_stoch.K_bodies
    M_factorization_blobs = build_block_diagonal_preconditioners_det_identity_stoch.M_factorization_blobs 
    M_factorization_blobs_inv = build_block_diagonal_preconditioners_det_identity_stoch.M_factorization_blobs_inv 
    mobility_inv_blobs = build_block_diagonal_preconditioners_det_identity_stoch.mobility_inv_blobs 
    

  def block_diagonal_preconditioner_identity(vector, bodies = None, mobility_bodies_identity = None,  K_bodies = None, Nblobs = None, *args, **kwargs):
    '''
    Solve the unconstrained mobility problem with M=I
    '''
    result = np.empty(vector.shape)
    offset = 0
    for k, b in enumerate(bodies):
      # 1. Solve I*Lambda_tilde = slip
      slip = vector[3*offset : 3*(offset + b.Nblobs)]
      Lambda_tilde = slip
      # 2. Compute rigid body velocity
      F = vector[3*Nblobs + 6*k : 3*Nblobs + 6*(k+1)]
      Y = np.dot(mobility_bodies_identity[k], -F - np.dot(K_bodies[k].T, Lambda_tilde))
      # 3. Set result (here we don care about lamda since we won't use it as a preconditioner) 
      result[3*Nblobs + 6*k : 3*Nblobs + 6*(k+1)] = Y
      offset += b.Nblobs
    return result
  block_diagonal_preconditioner_identity_partial = partial(block_diagonal_preconditioner_identity, 
                                                  bodies = bodies, 
                                                  mobility_bodies_identity = mobility_bodies_identity, 
                                                  K_bodies = K_bodies,
                                                  Nblobs = Nblobs)

  # Define preconditioned mobility matrix product
  def mobility_pc(w, bodies = None, P = None, r_vectors = None, eta = None, a = None, *args, **kwargs):
    result = np.empty_like(w)
    # Apply P
    offset = 0
    for k, b in enumerate(bodies):
      result[3*offset : 3*(offset + b.Nblobs)] = np.dot(P[k], w[3*offset : 3*(offset + b.Nblobs)]) 
      offset += b.Nblobs
    # Multiply by M
    result_2 = mobility_vector_prod(r_vectors, result, eta, a, *args, **kwargs)
    # Apply P.T
    offset = 0
    for k, b in enumerate(bodies):
      result[3*offset : 3*(offset + b.Nblobs)] = np.dot(P[k].T, result_2[3*offset : 3*(offset + b.Nblobs)])
      offset += b.Nblobs
    return result
  mobility_pc_partial = partial(mobility_pc, bodies = bodies, P = M_factorization_blobs_inv, r_vectors = r_vectors, eta = eta, a = a, *args, **kwargs)
  
  # Define inverse preconditioner P_inv
  def P_inv_mult(w, bodies = None, P_inv = None):
    offset = 0
    for k, b in enumerate(bodies):
      w[3*offset : 3*(offset + b.Nblobs)] = np.dot(P_inv[k], w[3*offset : 3*(offset + b.Nblobs)])
      offset += b.Nblobs
    return w
  P_inv_mult_partial = partial(P_inv_mult, bodies = bodies, P_inv = M_factorization_blobs)

  # Return preconditioner functions
  return block_diagonal_preconditioner_identity_partial, mobility_pc_partial, P_inv_mult_partial


@utils.static_var('initialized', [])
@utils.static_var('mobility_bodies', [])
@utils.static_var('K_bodies', [])
@utils.static_var('C_art_bodies', [])
@utils.static_var('res_art_bodies', [])
@utils.static_var('mobility_inv_blobs', [])
def build_block_diagonal_preconditioner(bodies, articulated, r_vectors, Nblobs, eta, a, *args, **kwargs):
  '''
  build the block diagonal preconditioner for articulated rigid bodies.
  it solves exactly the mobility problem for each body
  independently, i.e., no interation between bodies is taken
  into account.
  the first version only preconditions the constraint part with the identity matrix. 
  '''
  initialized = build_block_diagonal_preconditioner.initialized
  mobility_inv_blobs = []
  mobility_bodies = []
  K_bodies = []
  C_art_bodies = []
  res_art_bodies = []
  if(kwargs.get('step') % kwargs.get('update_PC') == 0) or len(build_block_diagonal_preconditioner.mobility_bodies) == 0:
    # loop over bodies
    for k, b in enumerate(bodies):
      if (b.prescribed_kinematics or b.Nblobs == 1) and len(initialized) > 0:
        mobility_inv_blobs.append(build_block_diagonal_preconditioner.mobility_inv_blobs[k])
        mobility_bodies.append(build_block_diagonal_preconditioner.mobility_bodies[k])
        K_bodies.append(build_block_diagonal_preconditioner.K_bodies[k])
      else:
        # 1. compute blobs mobility and invert it
        M = b.calc_mobility_blobs(eta, a)
        # 2. compute cholesy factorization, M = L^t * L
        L, lower = scipy.linalg.cho_factor(M)
        L = np.triu(L)   
        # 3. compute inverse mobility blobs
        mobility_inv_blobs.append(scipy.linalg.solve_triangular(L, scipy.linalg.solve_triangular(L, np.eye(b.Nblobs * 3), trans='T', check_finite=False), check_finite=False))
        # 4. compute geometric matrix K
        K = b.calc_K_matrix()
        K_bodies.append(K)
        # 5. compute body mobility
        mobility_bodies.append(np.linalg.pinv(np.dot(K.T, scipy.linalg.cho_solve((L,lower), K, check_finite=False))))
    
    # Loop over articulated bodies
    for ka, art in enumerate(articulated):     
      # Compute C matrix for each articulated body
      C_art_bodies.append(art.calc_C_matrix_articulated_body())
      # Compute the product C*N for each articulated body
      CN = np.zeros((3*art.num_constraints,6*art.num_bodies))
      for kc, const in enumerate(art.constraints):
        b1 = const.ind_bodies[0]
        b2 = const.ind_bodies[1]
        # Use local indices to the articulated body
        b1loc = art.return_body_local_index(b1)
        b2loc = art.return_body_local_index(b2)
        C1 = C_art_bodies[ka][3*kc:3*(kc+1), 6*b1loc:6*(b1loc+1)]
        C2 = C_art_bodies[ka][3*kc:3*(kc+1), 6*b2loc:6*(b2loc+1)]
        CN[3*kc:3*(kc+1),6*b1loc:6*(b1loc+1)] = np.dot(C1,mobility_bodies[b1])
        CN[3*kc:3*(kc+1),6*b2loc:6*(b2loc+1)] = np.dot(C2,mobility_bodies[b2])
      # Compute resistance matrix G = (C*N*C^T)^{-1} of each articulated body
      CT = C_art_bodies[ka].T
      CNCT = np.dot(CN,CT)
      res_art_bodies.append(np.linalg.pinv(CNCT))
     
    # save variables to use in next steps if pc is not updated
    build_block_diagonal_preconditioner.mobility_bodies = mobility_bodies
    build_block_diagonal_preconditioner.K_bodies = K_bodies
    build_block_diagonal_preconditioner.C_art_bodies = C_art_bodies
    build_block_diagonal_preconditioner.res_art_bodies = res_art_bodies
    build_block_diagonal_preconditioner.mobility_inv_blobs = mobility_inv_blobs

    # the function is initialized
    build_block_diagonal_preconditioner.initialized.append(1)
  else:
    # use old values
    mobility_bodies = build_block_diagonal_preconditioner.mobility_bodies 
    K_bodies = build_block_diagonal_preconditioner.K_bodies
    C_art_bodies = build_block_diagonal_preconditioner.C_art_bodies
    res_art_bodies = build_block_diagonal_preconditioner.res_art_bodies
    mobility_inv_blobs = build_block_diagonal_preconditioner.mobility_inv_blobs 

  def block_diagonal_preconditioner(vector, bodies = None, articulated = None, mobility_bodies = None, mobility_inv_blobs = None, K_bodies = None, Nblobs = None):
    '''
    Apply the block diagonal preconditioner.
    '''
    result = np.zeros(vector.shape)
    Ncomp = len(vector)
    offset = 0
    # First compute unconstrained body velocities
    for k, b in enumerate(bodies):
      if b.prescribed_kinematics is False:
        # 1. solve M*Lambda_tilde = slip
        slip = vector[3*offset : 3*(offset + b.Nblobs)]
        Lambda_tilde = np.dot(mobility_inv_blobs[k], slip)
        
        # 2. compute rigid body velocity
        F = vector[3*Nblobs + 6*k : 3*Nblobs + 6*(k+1)]
        U_unconst = np.dot(mobility_bodies[k], -F - np.dot(K_bodies[k].T, Lambda_tilde))
        
        # 3. solve M*Lambda = (slip + K*y)
        Lambda = np.dot(mobility_inv_blobs[k], slip + np.dot(K_bodies[k], U_unconst))
        
        # 4. set result
        result[3*offset : 3*(offset + b.Nblobs)] = Lambda
        result[3*Nblobs + 6*k : 3*Nblobs + 6*(k+1)] = U_unconst

      if b.prescribed_kinematics is True:
        # 1. solve M*Lambda = (slip + K*y)
        slip_KU = vector[3*offset : 3*(offset + b.Nblobs)]
        Lambda = np.dot(mobility_inv_blobs[k], slip_KU)

        # 2. set force
        F = np.dot(K_bodies[k].T, Lambda)

        # 3. set result
        result[3*offset : 3*(offset + b.Nblobs)] = Lambda
        result[3*Nblobs + 6*k : 3*Nblobs + 6*(k+1)] = F
      offset += b.Nblobs

    # Compute constraint forces and body velocities for each articulated body separately
    U_unconst = result[3*Nblobs : 3*Nblobs + 6*len(bodies)]
    for ka, art in enumerate(articulated): 
      # Get first and last indices of the bodies in art 
      indb_first = art.ind_bodies[0]
      indb_last = art.ind_bodies[-1]
      # C*U_unconst 
      C_times_U = np.dot(C_art_bodies[ka],U_unconst[6*indb_first:6*(indb_last+1)]) 
      # Get first and last indices of the constraints in art 
      indc_first = art.ind_constraints[0]
      indc_last = art.ind_constraints[-1]
      # RHS: prescribed link velocity B  
      B = vector[3*Nblobs + 6*len(bodies) + 3*indc_first : 3*Nblobs + 6*len(bodies) + 3*(indc_last+1)]
      # Lagrange multipliers: Phi = G*(B-C*U_unconst)
      Phi = np.dot(res_art_bodies[ka],B -  C_times_U)
      # Constraint forces: Fc = C^T*Phi
      Fc = np.dot(C_art_bodies[ka].T,Phi) 
      # N*Fc
      NFc = np.zeros(6*art.num_bodies)
      for kb, b in enumerate(art.bodies):
        indb = art.ind_bodies[kb]
        NFc[6*kb:6*(kb+1)] = np.dot(mobility_bodies[indb],Fc[6*kb:6*(kb+1)])
        # Computes the correction for lambda due to constraints
        Lambda_corr = np.dot(mobility_inv_blobs[indb], np.dot(K_bodies[indb], NFc[6*kb:6*(kb+1)]))
        # Navigate through bodies to assign the correction at the right location
        offset = b.blobs_offset
        result[3*offset : 3*(offset + b.Nblobs)] += Lambda_corr

      # Set result U = U_unconst + N*Fc
      result[3*Nblobs + 6*indb_first:3*Nblobs + 6*(indb_last+1)] += NFc
      # Set result Phi 
      result[3*Nblobs + 6*len(bodies) + 3*indc_first : 3*Nblobs + 6*len(bodies) + 3*(indc_last+1)] = Phi
    return result
  block_diagonal_preconditioner_partial = partial(block_diagonal_preconditioner, 
                                                  bodies = bodies, 
                                                  articulated = articulated, 
                                                  mobility_bodies = mobility_bodies, 
                                                  mobility_inv_blobs = mobility_inv_blobs, 
                                                  K_bodies = K_bodies,
                                                  Nblobs = Nblobs)
  return block_diagonal_preconditioner_partial


@utils.static_var('initialized', [])
@utils.static_var('mobility_bodies', [])
@utils.static_var('k_bodies', [])
@utils.static_var('mobility_inv_blobs', [])
def build_block_diagonal_preconditioner_articulated_identity(bodies, constraints, articulated, r_vectors, Nblobs, eta, a, *args, **kwargs):
  '''
  build the block diagonal preconditioner for articulated rigid bodies.
  it solves exactly the mobility problem for each body
  independently, i.e., no interation between bodies is taken
  into account.
  the first version only preconditions the constraint part with the identity matrix. 
  '''
  initialized = build_block_diagonal_preconditioner.initialized
  mobility_inv_blobs = []
  mobility_bodies = []
  K_bodies = []
  if(kwargs.get('step') % kwargs.get('update_PC') == 0) or len(build_block_diagonal_preconditioner.mobility_bodies) == 0:
    # loop over bodies
    for k, b in enumerate(bodies):
      if (b.prescribed_kinematics or b.Nblobs == 1) and len(initialized) > 0:
        mobility_inv_blobs.append(build_block_diagonal_preconditioner.mobility_inv_blobs[k])
        mobility_bodies.append(build_block_diagonal_preconditioner.mobility_bodies[k])
        K_bodies.append(build_block_diagonal_preconditioner.K_bodies[k])
      else:
        # 1. compute blobs mobility and invert it
        M = b.calc_mobility_blobs(eta, a)
        # 2. compute cholesy factorization, M = L^t * L
        L, lower = scipy.linalg.cho_factor(M)
        L = np.triu(L)   
        # 3. compute inverse mobility blobs
        mobility_inv_blobs.append(scipy.linalg.solve_triangular(L, scipy.linalg.solve_triangular(L, np.eye(b.Nblobs * 3), trans='T', check_finite=False), check_finite=False))
        # 4. compute geometric matrix K
        K = b.calc_K_matrix()
        K_bodies.append(K)
        # 5. compute body mobility
        mobility_bodies.append(np.linalg.pinv(np.dot(K.T, scipy.linalg.cho_solve((L,lower), K, check_finite=False))))
    
    # save variables to use in next steps if pc is not updated
    build_block_diagonal_preconditioner.mobility_bodies = mobility_bodies
    build_block_diagonal_preconditioner.K_bodies = K_bodies
    build_block_diagonal_preconditioner.mobility_inv_blobs = mobility_inv_blobs

    # the function is initialized
    build_block_diagonal_preconditioner.initialized.append(1)
  else:
    # use old values
    mobility_bodies = build_block_diagonal_preconditioner.mobility_bodies 
    K_bodies = build_block_diagonal_preconditioner.K_bodies
    mobility_inv_blobs = build_block_diagonal_preconditioner.mobility_inv_blobs 

  def block_diagonal_preconditioner(vector, bodies = None, constraints = None, mobility_bodies = None, mobility_inv_blobs = None, K_bodies = None, Nblobs = None):
    '''
    Apply the block diagonal preconditioner.
    '''
    result = np.zeros(vector.shape)
    Ncomp = len(vector)
    offset = 0
    for k, b in enumerate(bodies):
      if b.prescribed_kinematics is False:
        # 1. solve M*Lambda_tilde = slip
        slip = vector[3*offset : 3*(offset + b.Nblobs)]
        Lambda_tilde = np.dot(mobility_inv_blobs[k], slip)
        
        # 2. compute rigid body velocity
        F = vector[3*Nblobs + 6*k : 3*Nblobs + 6*(k+1)]
        y = np.dot(mobility_bodies[k], -F - np.dot(K_bodies[k].T, Lambda_tilde))
        
        # 3. solve M*Lambda = (slip + K*y)
        Lambda = np.dot(mobility_inv_blobs[k], slip + np.dot(K_bodies[k], y))
        
        # 4. set result
        result[3*offset : 3*(offset + b.Nblobs)] = Lambda
        result[3*Nblobs + 6*k : 3*Nblobs + 6*(k+1)] = y
      if b.prescribed_kinematics is True:
        # 1. solve M*Lambda = (slip + K*y)
        slip_KU = vector[3*offset : 3*(offset + b.Nblobs)]
        Lambda = np.dot(mobility_inv_blobs[k], slip_KU)

        # 2. set force
        F = np.dot(K_bodies[k].T, Lambda)

        # 3. set result
        result[3*offset : 3*(offset + b.Nblobs)] = Lambda
        result[3*Nblobs + 6*k : 3*Nblobs + 6*(k+1)] = F
      offset += b.Nblobs

    # set constraint force = rhs
    result[3*Nblobs + 6*len(bodies):Ncomp] = vector[3*Nblobs + 6*len(bodies):Ncomp]
    return result
  block_diagonal_preconditioner_partial = partial(block_diagonal_preconditioner, 
                                                  bodies = bodies, 
                                                  constraints = constraints, 
                                                  mobility_bodies = mobility_bodies, 
                                                  mobility_inv_blobs = mobility_inv_blobs, 
                                                  K_bodies = K_bodies,
                                                  Nblobs = Nblobs)
  return block_diagonal_preconditioner_partial

@utils.static_var('initialized', [])
@utils.static_var('mobility_bodies', [])
@utils.static_var('C_art_bodies', [])
@utils.static_var('res_art_bodies', [])
def build_block_diagonal_preconditioner_articulated_single_blobs(bodies, articulated, Nblobs, Nconstraints, eta, a, *args, **kwargs):
  '''
  build the block diagonal preconditioner for articulated rigid bodies.
  it solves exactly the mobility problem for each body
  independently, i.e., no interation between bodies is taken
  into account.
  the first version only preconditions the constraint part with the identity matrix. 
  '''
  initialized = build_block_diagonal_preconditioner.initialized
  mobility_bodies = []
  C_art_bodies = []
  res_art_bodies = []
  if(kwargs.get('step') % kwargs.get('update_PC') == 0) or len(build_block_diagonal_preconditioner.mobility_bodies) == 0:
    # 1. compute diag single blob mobility
    factor_Mtt = 1./(6*np.pi*eta*a)
    factor_Mrr = 1./(8*np.pi*eta*a**3)
    M = np.eye(6)
    np.fill_diagonal(M[0:3,0:3], factor_Mtt)
    np.fill_diagonal(M[3:6,3:6], factor_Mrr)
    # loop over bodies
    for k, b in enumerate(bodies):
      # 2. add it to the body(=blob) mobility
      mobility_bodies.append(M)
    
    # Loop over articulated bodies
    for ka, art in enumerate(articulated):     
      # Compute C matrix for each articulated body
      C_art_bodies.append(art.calc_C_matrix_articulated_body())
      # Compute the product C*N for each articulated body
      CN = np.zeros((3*art.num_constraints,6*art.num_bodies))
      for kc, const in enumerate(art.constraints):
        b1 = const.ind_bodies[0]
        b2 = const.ind_bodies[1]
        # Use local indices to the articulated body
        b1loc = art.return_body_local_index(b1)
        b2loc = art.return_body_local_index(b2)
        C1 = C_art_bodies[ka][3*kc:3*(kc+1), 6*b1loc:6*(b1loc+1)]
        C2 = C_art_bodies[ka][3*kc:3*(kc+1), 6*b2loc:6*(b2loc+1)]
        CN[3*kc:3*(kc+1),6*b1loc:6*(b1loc+1)] = np.dot(C1,mobility_bodies[b1])
        CN[3*kc:3*(kc+1),6*b2loc:6*(b2loc+1)] = np.dot(C2,mobility_bodies[b2])
      # Compute resistance matrix G = (C*N*C^T)^{-1} of each articulated body
      CT = C_art_bodies[ka].T
      CNCT = np.dot(CN,CT)
      res_art_bodies.append(np.linalg.pinv(CNCT))
     
    # save variables to use in next steps if pc is not updated
    build_block_diagonal_preconditioner.mobility_bodies = mobility_bodies
    build_block_diagonal_preconditioner.C_art_bodies = C_art_bodies
    build_block_diagonal_preconditioner.res_art_bodies = res_art_bodies

    # the function is initialized
    build_block_diagonal_preconditioner.initialized.append(1)
  else:
    # use old values
    mobility_bodies = build_block_diagonal_preconditioner.mobility_bodies 
    C_art_bodies = build_block_diagonal_preconditioner.C_art_bodies
    res_art_bodies = build_block_diagonal_preconditioner.res_art_bodies

  def block_diagonal_preconditioner(vector, bodies = None, articulated = None, mobility_bodies = None, Nblobs = None, Nconstraints = None):
    '''
    Apply the block diagonal preconditioner.
    '''
    result = np.zeros(vector.shape)
    Ncomp = len(vector)
    # First get unconstrained body velocities for RHS: M*F
    result[3*Nconstraints:3*Nconstraints+6*Nblobs] = vector[0:6*Nblobs] 

    # Compute constraint forces and body velocities for each articulated body separately
    U_unconst = result[3*Nconstraints:3*Nconstraints+6*Nblobs]
    for ka, art in enumerate(articulated): 
      # Get first and last indices of the bodies in art 
      indb_first = art.ind_bodies[0]
      indb_last = art.ind_bodies[-1]
      # C*U_unconst 
      C_times_U = np.dot(C_art_bodies[ka],U_unconst[6*indb_first:6*(indb_last+1)])
      # Get first and last indices of the constraints in art 
      indc_first = art.ind_constraints[0]
      indc_last = art.ind_constraints[-1]
      # RHS: prescribed link velocity B  
      B = vector[6*Nblobs + 3*indc_first : 6*Nblobs + 3*(indc_last+1)]
      # Lagrange multipliers: Phi = G*(C*U_unconst-B)
      Phi = np.dot(res_art_bodies[ka],C_times_U-B)
      # Constraint forces: Fc = C^T*Phi
      Fc = np.dot(C_art_bodies[ka].T,Phi)
      # N*Fc
      NFc = np.zeros(6*art.num_bodies)
      for kb, b in enumerate(art.bodies):
        indb = art.ind_bodies[kb]
        NFc[6*kb:6*(kb+1)] = np.dot(mobility_bodies[indb],Fc[6*kb:6*(kb+1)])
      # Set result U = U_unconst - N*Fc
      result[3*Nconstraints + 6*indb_first: 3*Nconstraints + 6*(indb_last+1)] -= NFc
      # Set result Phi 
      result[3*indc_first : 3*(indc_last+1)] = Phi
    return result
  block_diagonal_preconditioner_partial = partial(block_diagonal_preconditioner, 
                                                  bodies = bodies, 
                                                  articulated = articulated, 
                                                  mobility_bodies = mobility_bodies, 
                                                  Nblobs = Nblobs,
                                                  Nconstraints = Nconstraints)
  return block_diagonal_preconditioner_partial



if __name__ == '__main__':
  # Get command line arguments
  parser = argparse.ArgumentParser(description='Run a multi-body simulation and save trajectory.')
  parser.add_argument('--input-file', dest='input_file', type=str, default='data.main', help='name of the input file')
  parser.add_argument('--print-residual', action='store_true', help='print gmres and lanczos residuals')
  args=parser.parse_args()
  input_file = args.input_file

  # Read input file
  read = read_input.ReadInput(input_file)
   
  # Set some variables for the simulation
  n_steps = read.n_steps 
  n_save = read.n_save
  n_relaxation = read.n_relaxation
  dt = read.dt
  eta = read.eta 
  g = read.g 
  a = read.blob_radius
  scheme  = read.scheme 
  output_name = read.output_name 
  structures = read.structures
  structures_ID = read.structures_ID
  multi_bodies_functions.calc_body_body_forces_torques = multi_bodies_functions.set_body_body_forces_torques(read.body_body_force_torque_implementation)

  # Copy input file to output
  # subprocess.call(["cp", input_file, output_name + '.inputfile'])
  copyfile(input_file,output_name + '.inputfile')

  # Set random generator state
  if read.random_state is not None:
    with open(read.random_state, 'rb') as f:
      np.random.set_state(cpickle.load(f))
  elif read.seed is not None:
    np.random.seed(int(read.seed))
  
  # Save random generator state
  with open(output_name + '.random_state', 'wb') as f:
    cpickle.dump(np.random.get_state(), f)

  # Create rigid bodies
  bodies = []
  body_types = []
  body_names = []
  blobs_offset = 0
  for ID, structure in enumerate(structures):
    print('Creating structures = ', structure[1])
    # Read vertex and clones files
    struct_ref_config = read_vertex_file.read_vertex_file(structure[0])
    num_bodies_struct, struct_locations, struct_orientations = read_clones_file.read_clones_file(structure[1])
    # Read slip file if it exists
    slip = None
    if(len(structure) > 2):
      slip = read_slip_file.read_slip_file(structure[2])
    body_types.append(num_bodies_struct)
    body_names.append(structures_ID[ID])
    # Create each body of type structure
    for i in range(num_bodies_struct):
      b = body.Body(struct_locations[i], struct_orientations[i], struct_ref_config, a)
      b.mobility_blobs = set_mobility_blobs(read.mobility_blobs_implementation)
      b.ID = structures_ID[ID]
      # Calculate body length for the RFD
      if b.Nblobs > 2000:
        b.body_length = 10.0
      elif i == 0:
        b.calc_body_length()
      else:
        b.body_length = bodies[-1].body_length
      # Compute the blobs offset for lambda in the whole system array
      b.blobs_offset = blobs_offset
      blobs_offset += b.Nblobs
      multi_bodies_functions.set_slip_by_ID(b, slip)
      # If structure is an obstacle
      if ID >= read.num_free_bodies:
        b.prescribed_kinematics = True
        b.prescribed_velocity = np.zeros(6)
      # Append bodies to total bodies list
      bodies.append(b)

  # Set some variables
  num_bodies_rigid = len(bodies)
      
  # Create articulated bodies
  articulated = []
  constraints = []
  bodies_offset = num_bodies_rigid
  constraints_offset = 0
  for ID, structure in enumerate(read.articulated):
    print('Creating articulated = ', structure[1])
    # Read vertex, clones and constraint files
    struct_ref_config = read_vertex_file_list.read_vertex_file_list(structure[0], output_name)
    num_bodies_struct, struct_locations, struct_orientations = read_clones_file.read_clones_file(structure[1])    
    constraints_info = read_constraints_file.read_constraints_file(structure[2], output_name)
    num_bodies_in_articulated = constraints_info[0]
    num_constraints = constraints_info[1]
    constraints_bodies = constraints_info[2]
    constraints_links = constraints_info[3]
    constraints_extra = constraints_info[4]
    # Read slip file if it exists
    slip = None
    if(len(structure) > 3):
      slip = read_slip_file.read_slip_file(structure[3])
    body_types.append(num_bodies_struct)
    body_names.append(read.articulated_ID[ID])
    # Create each body of type structure
    for i in range(num_bodies_struct):
      subbody = i % num_bodies_in_articulated
      b = body.Body(struct_locations[i], struct_orientations[i], struct_ref_config[subbody], a)
      b.mobility_blobs = set_mobility_blobs(read.mobility_blobs_implementation)
      b.ID = read.articulated_ID[ID]
      # Calculate body length for the RFD
      if b.Nblobs > 2000:
        b.body_length = 10.0
      elif i == 0:
        b.calc_body_length()
      else:
        b.body_length = bodies[-1].body_length
      # Compute the blobs offset for lambda in the whole system array
      b.blobs_offset = blobs_offset
      blobs_offset += b.Nblobs
      multi_bodies_functions.set_slip_by_ID(b, slip)
      # Append bodies to total bodies list
      bodies.append(b)

    # Total number of constraints and articulated rigid bodies
    num_constraints_total = num_constraints * (num_bodies_struct // num_bodies_in_articulated)
   
    # Create list of constraints
    for i in range(num_constraints_total):
      # Prepare info for constraint
      subconstraint = i % num_constraints
      articulated_body = i // num_constraints
      bodies_indices = constraints_bodies[subconstraint] + num_bodies_in_articulated * articulated_body + bodies_offset
      bodies_in_link = [bodies[bodies_indices[0]], bodies[bodies_indices[1]]]
      parameters = constraints_links[subconstraint]

      # Create constraint
      c = Constraint(bodies_in_link, bodies_indices,  articulated_body, parameters, constraints_extra[subconstraint])
      constraints.append(c)

    # Create articulated rigid body
    for i in range(num_bodies_struct // num_bodies_in_articulated):
      bodies_indices = bodies_offset + i * num_bodies_in_articulated + np.arange(num_bodies_in_articulated, dtype=int)
      bodies_in_articulated = bodies[bodies_indices[0] : bodies_indices[-1] + 1]
      constraints_indices = constraints_offset + i * num_constraints + np.arange(num_constraints, dtype=int)
      constraints_in_articulated = constraints[constraints_indices[0] : constraints_indices[-1] + 1]
      art = Articulated(bodies_in_articulated,
                        bodies_indices,
                        constraints_in_articulated,
                        constraints_indices,
                        num_bodies_in_articulated,
                        num_constraints,
                        constraints_bodies,
                        constraints_links,
                        constraints_extra)
      articulated.append(art)

    # Update offsets
    bodies_offset += num_bodies_struct
    constraints_offset += num_constraints_total

  bodies = np.array(bodies)
  
  # Set some more variables
  num_of_body_types = len(body_types)
  num_bodies = bodies.size
  Nblobs = sum([x.Nblobs for x in bodies])

  # Save bodies information
  with open(output_name + '.bodies_info', 'w') as f:
    f.write('num_of_body_types  ' + str(num_of_body_types) + '\n')
    f.write('body_names         ' + str(body_names) + '\n')
    f.write('body_types         ' + str(body_types) + '\n')
    f.write('num_bodies         ' + str(num_bodies) + '\n')
    f.write('num_blobs          ' + str(Nblobs) + '\n')

  # Create integrator
  if scheme.find('rollers') == -1:
    integrator = QuaternionIntegrator(bodies, Nblobs, scheme, tolerance = read.solver_tolerance, domain = read.domain) 
    integrator.build_block_diagonal_preconditioner = build_block_diagonal_preconditioner
    integrator.first_guess = np.zeros(Nblobs*3 + num_bodies*6 + len(constraints)*3)
  else:
    integrator = QuaternionIntegratorRollers(bodies, Nblobs, scheme, tolerance = read.solver_tolerance, domain = read.domain, 
                                             mobility_vector_prod_implementation = read.mobility_vector_prod_implementation) 
    integrator.calc_one_blob_forces = partial(multi_bodies_functions.calc_one_blob_forces,
                                              g = g,
                                              repulsion_strength_wall = read.repulsion_strength_wall, 
                                              debye_length_wall = read.debye_length_wall)
    integrator.calc_blob_blob_forces = partial(multi_bodies_functions.calc_blob_blob_forces,
                                               g = g,
                                               repulsion_strength_wall = read.repulsion_strength_wall, 
                                               debye_length_wall = read.debye_length_wall,
                                               repulsion_strength = read.repulsion_strength,
                                               debye_length = read.debye_length, 
                                               periodic_length = read.periodic_length)
    integrator.omega_one_roller = read.omega_one_roller
    integrator.free_kinematics = read.free_kinematics
    integrator.hydro_interactions = read.hydro_interactions
    integrator.build_block_diagonal_preconditioner = build_block_diagonal_preconditioner_articulated_single_blobs
    integrator.C_matrix_T_vector_prod = C_matrix_T_vector_prod 
    integrator.C_matrix_vector_prod = C_matrix_vector_prod 
    integrator.first_guess = np.zeros(num_bodies*6 + len(constraints)*3)

    
  integrator.calc_slip = partial(calc_slip,
                                 implementation = read.mobility_vector_prod_implementation, 
                                 blob_radius = a, 
                                 eta = a, 
                                 g = g) 
  integrator.get_blobs_r_vectors = get_blobs_r_vectors 
  integrator.mobility_blobs = set_mobility_blobs(read.mobility_blobs_implementation)
  integrator.mobility_vector_prod = set_mobility_vector_prod(read.mobility_vector_prod_implementation, bodies=bodies)
  mobility_vector_prod = set_mobility_vector_prod(read.mobility_vector_prod_implementation, bodies=bodies)
  integrator.force_torque_calculator = partial(multi_bodies_functions.force_torque_calculator_sort_by_bodies, 
                                               g = g, 
                                               repulsion_strength_wall = read.repulsion_strength_wall, 
                                               debye_length_wall = read.debye_length_wall, 
                                               repulsion_strength = read.repulsion_strength, 
                                               debye_length = read.debye_length, 
                                               periodic_length = read.periodic_length,
                                               omega_one_roller = read.omega_one_roller) 
  integrator.calc_K_matrix_bodies = calc_K_matrix_bodies
  integrator.calc_K_matrix = calc_K_matrix

  integrator.linear_operator = linear_operator_rigid
  integrator.build_block_diagonal_preconditioners_det_stoch = build_block_diagonal_preconditioners_det_stoch
  integrator.build_block_diagonal_preconditioners_det_identity_stoch = build_block_diagonal_preconditioners_det_identity_stoch
  integrator.eta = eta
  integrator.a = a
  integrator.kT = read.kT
  integrator.K_matrix_T_vector_prod = K_matrix_T_vector_prod
  integrator.K_matrix_vector_prod = K_matrix_vector_prod
  integrator.preprocess = multi_bodies_functions.preprocess
  integrator.postprocess = multi_bodies_functions.postprocess
  integrator.periodic_length = read.periodic_length
  integrator.update_PC = read.update_PC
  integrator.print_residual = args.print_residual
  integrator.rf_delta = read.rf_delta
  integrator.num_bodies_rigid = num_bodies_rigid
  integrator.constraints = constraints
  integrator.calc_C_matrix_constraints = calc_C_matrix_constraints
  integrator.articulated = articulated
  integrator.nonlinear_solver_tolerance = read.nonlinear_solver_tolerance
  multi_bodies_functions.calc_blob_blob_forces = multi_bodies_functions.set_blob_blob_forces(read.blob_blob_force_implementation, bodies=bodies)  
  integrator.plot_velocity_field = read.plot_velocity_field
  integrator.output_name = read.output_name
  try:
    integrator.plot_velocity_field_shell = multi_bodies_functions.plot_velocity_field_shell
  except:
    pass

  # Initialize HydroGrid library:
  if found_HydroGrid and read.call_HydroGrid:
    cc.calculate_concentration(output_name, 
                               read.periodic_length[0], 
                               read.periodic_length[1], 
                               int(read.green_particles[0]), 
                               int(read.green_particles[1]), 
                               int(read.cells[0]), 
                               int(read.cells[1]), 
                               0, 
                               dt * read.sample_HydroGrid, 
                               Nblobs, 
                               0, 
                               get_blobs_r_vectors(bodies, Nblobs))


  # Loop over time steps
  start_time = time.time()
  if read.save_clones == 'one_file':
    output_files = []
    buffering = max(1, min(body_types) * n_steps // n_save // 200)
    ID_loop = read.structures_ID + read.articulated_ID
    for i, ID in enumerate(ID_loop):
      name = output_name + '.' + ID + '.config'
      output_files.append(open(name, 'w', buffering=buffering))

  for step in range(read.initial_step, n_steps):
    # Save data if...
    if (step % n_save) == 0 and step >= 0:
      elapsed_time = time.time() - start_time
      print('Integrator = ', scheme, ', step = ', step, ', invalid configurations', integrator.invalid_configuration_count, ', wallclock time = ', time.time() - start_time)
      # For each type of structure save locations and orientations to one file
      body_offset = 0
      if read.save_clones == 'one_file_per_step':
        ID_loop = read.structures_ID + read.articulated_ID
        for i, ID in enumerate(ID_loop):
          name = output_name + '.' + ID + '.' + str(step).zfill(8) + '.clones'
          with open(name, 'w') as f_ID:
            f_ID.write(str(body_types[i]) + '\n')
            for j in range(body_types[i]):
              orientation = bodies[body_offset + j].orientation.entries
              f_ID.write('%s %s %s %s %s %s %s\n' % (bodies[body_offset + j].location[0], 
                                                     bodies[body_offset + j].location[1], 
                                                     bodies[body_offset + j].location[2], 
                                                     orientation[0], 
                                                     orientation[1], 
                                                     orientation[2], 
                                                     orientation[3]))
            body_offset += body_types[i]
      elif read.save_clones == 'one_file':
        for i, f_ID in enumerate(output_files):
          f_ID.write(str(body_types[i]) + '\n')
          for j in range(body_types[i]):
            orientation = bodies[body_offset + j].orientation.entries
            f_ID.write('%s %s %s %s %s %s %s\n' % (bodies[body_offset + j].location[0],
                                                   bodies[body_offset + j].location[1],
                                                   bodies[body_offset + j].location[2],
                                                   orientation[0],
                                                   orientation[1],
                                                   orientation[2],
                                                   orientation[3]))
          body_offset += body_types[i]

      else:
        print('Error, save_clones =', read.save_clones, 'is not implemented.')
        print('Use \"one_file_per_step\" or \"one_file\". \n')
        break

      # Save mobilities
      if read.save_blobs_mobility == 'True' or read.save_body_mobility == 'True':
        r_vectors_blobs = integrator.get_blobs_r_vectors(bodies, Nblobs)
        mobility_blobs = integrator.mobility_blobs(r_vectors_blobs, read.eta, read.blob_radius)
        if read.save_blobs_mobility == 'True':
          name = output_name + '.blobs_mobility.' + str(step).zfill(8) + '.dat'
          np.savetxt(name, mobility_blobs, delimiter='  ')
        if read.save_body_mobility == 'True':
          resistance_blobs = np.linalg.inv(mobility_blobs)
          K = integrator.calc_K_matrix(bodies, Nblobs)
          resistance_bodies = np.dot(K.T, np.dot(resistance_blobs, K))
          mobility_bodies = np.linalg.pinv(np.dot(K.T, np.dot(resistance_blobs, K)))
          name = output_name + '.body_mobility.' + str(step).zfill(8) + '.dat'
          np.savetxt(name, mobility_bodies, delimiter='  ')
        
    # Update HydroGrid
    if (step % read.sample_HydroGrid) == 0 and found_HydroGrid and read.call_HydroGrid:
      cc.calculate_concentration(output_name, 
                                 read.periodic_length[0], 
                                 read.periodic_length[1], 
                                 int(read.green_particles[0]), 
                                 int(read.green_particles[1]),  
                                 int(read.cells[0]), 
                                 int(read.cells[1]), 
                                 step, 
                                 dt * read.sample_HydroGrid, 
                                 Nblobs, 
                                 1, 
                                 get_blobs_r_vectors(bodies, Nblobs))
    
    # Save HydroGrid data
    if read.save_HydroGrid > 0 and found_HydroGrid and read.call_HydroGrid:
      if (step % read.save_HydroGrid) == 0:
        cc.calculate_concentration(output_name, 
                                   read.periodic_length[0], 
                                   read.periodic_length[1], 
                                   int(read.green_particles[0]), 
                                   int(read.green_particles[1]),  
                                   int(read.cells[0]), 
                                   int(read.cells[1]), 
                                   step, 
                                   dt * read.sample_HydroGrid, 
                                   Nblobs, 
                                   2, 
                                   get_blobs_r_vectors(bodies, Nblobs))

    # Advance time step
    integrator.advance_time_step(dt, step = step)

  # Save final data if...
  if ((step+1) % n_save) == 0 and step >= 0:
    print('Integrator = ', scheme, ', step = ', step+1, ', invalid configurations', integrator.invalid_configuration_count, ', wallclock time = ', time.time() - start_time)
    # For each type of structure save locations and orientations to one file
    body_offset = 0
    if read.save_clones == 'one_file_per_step':
      ID_loop = read.structures_ID + read.articulated_ID
      for i, ID in enumerate(ID_loop):
        name = output_name + '.' + ID + '.' + str(step+1).zfill(8) + '.clones'
        with open(name, 'w') as f_ID:
          f_ID.write(str(body_types[i]) + '\n')
          for j in range(body_types[i]):
            orientation = bodies[body_offset + j].orientation.entries
            f_ID.write('%s %s %s %s %s %s %s\n' % (bodies[body_offset + j].location[0], 
                                                   bodies[body_offset + j].location[1], 
                                                   bodies[body_offset + j].location[2], 
                                                   orientation[0], 
                                                   orientation[1], 
                                                   orientation[2], 
                                                   orientation[3]))
          body_offset += body_types[i]
      
    elif read.save_clones == 'one_file':
      for i, f_ID in enumerate(output_files):
        f_ID.write(str(body_types[i]) + '\n')
        for j in range(body_types[i]):
          orientation = bodies[body_offset + j].orientation.entries
          f_ID.write('%s %s %s %s %s %s %s\n' % (bodies[body_offset + j].location[0],
                                                 bodies[body_offset + j].location[1],
                                                 bodies[body_offset + j].location[2],
                                                 orientation[0],
                                                 orientation[1],
                                                 orientation[2],
                                                 orientation[3]))
        body_offset += body_types[i]

    else:
      print('Error, save_clones =', read.save_clones, 'is not implemented.')
      print('Use \"one_file_per_step\" or \"one_file\". \n')

    # Save mobilities
    if read.save_blobs_mobility == 'True' or read.save_body_mobility == 'True':
      r_vectors_blobs = integrator.get_blobs_r_vectors(bodies, Nblobs)
      mobility_blobs = integrator.mobility_blobs(r_vectors_blobs, read.eta, read.blob_radius)
      if read.save_blobs_mobility == 'True':
        name = output_name + '.blobs_mobility.' + str(step+1).zfill(8) + '.dat'
        np.savetxt(name, mobility_blobs, delimiter='  ')
      if read.save_body_mobility == 'True':
        resistance_blobs = np.linalg.inv(mobility_blobs)
        K = integrator.calc_K_matrix(bodies, Nblobs)
        resistance_bodies = np.dot(K.T, np.dot(resistance_blobs, K))
        mobility_bodies = np.linalg.pinv(np.dot(K.T, np.dot(resistance_blobs, K)))
        name = output_name + '.body_mobility.' + str(step+1).zfill(8) + '.dat'
        np.savetxt(name, mobility_bodies, delimiter='  ')
        
  # Update HydroGrid data
  if ((step+1) % read.sample_HydroGrid) == 0 and found_HydroGrid and read.call_HydroGrid:
    cc.calculate_concentration(output_name, 
                               read.periodic_length[0], 
                               read.periodic_length[1], 
                               int(read.green_particles[0]), 
                               int(read.green_particles[1]),  
                               int(read.cells[0]), 
                               int(read.cells[1]), 
                               step+1, 
                               dt * read.sample_HydroGrid, 
                               Nblobs, 
                               1, 
                               get_blobs_r_vectors(bodies, Nblobs))

  # Save HydroGrid data
  if read.save_HydroGrid > 0 and found_HydroGrid and read.call_HydroGrid:
    if ((step+1) % read.save_HydroGrid) == 0:
      cc.calculate_concentration(output_name, 
                                 read.periodic_length[0], 
                                 read.periodic_length[1], 
                                 int(read.green_particles[0]),
                                 int(read.green_particles[1]), 
                                 int(read.cells[0]), 
                                 int(read.cells[1]), 
                                 step+1, 
                                 dt * read.sample_HydroGrid, 
                                 Nblobs, 
                                 2, 
                                 get_blobs_r_vectors(bodies, Nblobs))


  # Free HydroGrid
  if found_HydroGrid and read.call_HydroGrid:
    cc.calculate_concentration(output_name, 
                               read.periodic_length[0], 
                               read.periodic_length[1], 
                               int(read.green_particles[0]), 
                               int(read.green_particles[1]),  
                               int(read.cells[0]), 
                               int(read.cells[1]), 
                               step+1, 
                               dt * read.sample_HydroGrid, 
                               Nblobs, 
                               3, 
                               get_blobs_r_vectors(bodies, Nblobs))


  # Save wallclock time 
  with open(output_name + '.time', 'w') as f:
    f.write(str(time.time() - start_time) + '\n')
  # Save number of invalid configurations and number of iterations in the
  # deterministic solvers and the Lanczos algorithm  
  with open(output_name + '.info', 'w') as f:
    nonlinear_counter = 0
    for i in range(len(articulated)):
      nonlinear_counter += articulated[i].nonlinear_iteration_counter
      
    f.write('invalid_configuration_count    = ' + str(integrator.invalid_configuration_count) + '\n'
            + 'deterministic_iterations_count = ' + str(integrator.det_iterations_count) + '\n'
            + 'stochastic_iterations_count    = ' + str(integrator.stoch_iterations_count) + '\n'
            + 'nonlinear_iterations_count     = ' + str(nonlinear_counter) + '\n')
  print('\n\n\n# End')
