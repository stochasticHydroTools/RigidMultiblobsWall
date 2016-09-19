import argparse
import numpy as np
import scipy.linalg as sla
import subprocess
import cPickle
from functools import partial
import sys
import time
sys.path.append('../')

import multi_bodies_functions
from mobility import mobility as mb
from quaternion_integrator.quaternion import Quaternion
from quaternion_integrator.quaternion_integrator_multi_bodies import QuaternionIntegrator
from quaternion_integrator.quaternion_integrator_rollers import QuaternionIntegratorRollers
from body import body 
from read_input import read_input
from read_input import read_vertex_file
from read_input import read_clones_file



def calc_slip(bodies, Nblobs):
  '''
  Function to calculate the slip in all the blobs.
  '''
  slip = np.empty((Nblobs, 3))
  offset = 0
  for b in bodies:
    slip_b = b.calc_slip()
    slip[offset:offset+b.Nblobs] = slip_b
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
  The implementation in C++ is much faster than 
  the one python; to use it the user should compile 
  the file mobility/mobility_ext.cc.

  These functions return an array with shape 
  (3*Nblobs, 3*Nblobs).
  '''
  # Implementations without wall
  if implementation == 'python_no_wall':
    return mb.rotne_prager_tensor

  # Implementations with wall
  elif implementation == 'python':
    return mb.single_wall_fluid_mobility
  elif implementation == 'C++':
    return  mb.boosted_single_wall_fluid_mobility


def set_mobility_vector_prod(implementation):
  '''
  Set the function to compute the matrix-vector
  product (M*F) with the mobility defined at the blob 
  level to the right implementation.
  
  The implementation in pycuda is much faster than the
  one in C++, which is much faster than the one python; 
  To use the pycuda implementation is necessary to have 
  installed pycuda and a GPU with CUDA capabilities. To
  use the C++ implementation the user has to compile 
  the file mobility/mobility_ext.cc.  
  ''' 
  # Implementations with wall
  if implementation == 'python':
    return mb.single_wall_fluid_mobility_product
  elif implementation == 'C++':
    return mb.boosted_mobility_vector_product
  elif implementation == 'pycuda':
    return mb.single_wall_mobility_trans_times_force_pycuda


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


def K_matrix_vector_prod(bodies, vector, Nblobs):
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
    K = b.calc_K_matrix()
    result[offset : offset+b.Nblobs] = np.reshape(np.dot(K, v[6*k : 6*(k+1)]), (b.Nblobs, 3))
    offset += b.Nblobs    

  return result


def K_matrix_T_vector_prod(bodies, vector, Nblobs):
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
    K = b.calc_K_matrix()
    result[k : k+1] = np.dot(K.T, v[3*offset : 3*(offset+b.Nblobs)])
    offset += b.Nblobs    

  result = np.reshape(result, (2*len(bodies), 3))
  return result


def linear_operator_rigid(vector, bodies, r_vectors, eta, a, *args, **kwargs):
  '''
  Return the action of the linear operator of the rigid body on vector v.
  The linear operator is
  |  M   -K|
  | -K^T  0|
  ''' 
  # Reserve memory for the solution and create some variables
  L = kwargs.get('periodic_length')
  Ncomp_blobs = r_vectors.size
  Nblobs = r_vectors.size / 3
  Ncomp_bodies = 6 * len(bodies)
  res = np.empty((Ncomp_blobs + Ncomp_bodies))
  v = np.reshape(vector, (vector.size/3, 3))
  
  # Compute the "slip" part
  res[0:Ncomp_blobs] = mobility_vector_prod(r_vectors, vector[0:Ncomp_blobs], eta, a, *args, **kwargs) 
  K_times_U = K_matrix_vector_prod(bodies, v[Nblobs : Nblobs+2*len(bodies)], Nblobs) 
  res[0:Ncomp_blobs] -= np.reshape(K_times_U, (3*Nblobs))

  # Compute the "-force_torque" part
  K_T_times_lambda = K_matrix_T_vector_prod(bodies, vector[0:Ncomp_blobs], Nblobs)
  res[Ncomp_blobs : Ncomp_blobs+Ncomp_bodies] = -np.reshape(K_T_times_lambda, (Ncomp_bodies))
  return res


def block_diagonal_preconditioner(vector, bodies, mobility_bodies, mobility_inv_blobs, Nblobs):
  '''
  Block diagonal preconditioner for rigid bodies.
  It solves exactly the mobility problem for each body
  independently, i.e., no interation between bodies is taken
  into account.
  '''
  result = np.empty(vector.shape)
  offset = 0
  for k, b in enumerate(bodies):
    # 1. Solve M*Lambda_tilde = slip
    slip = vector[3*offset : 3*(offset + b.Nblobs)]
    Lambda_tilde = np.dot(mobility_inv_blobs[k], slip)

    # 2. Compute rigid body velocity
    F = vector[3*Nblobs + 6*k : 3*Nblobs + 6*(k+1)]
    Y = np.dot(mobility_bodies[k], -F - np.dot(b.calc_K_matrix().T, Lambda_tilde))

    # 3. Solve M*Lambda = (slip + K*Y)
    Lambda = np.dot(mobility_inv_blobs[k], slip + np.dot(b.calc_K_matrix(), Y))
    
    # 4. Set result
    result[3*offset : 3*(offset + b.Nblobs)] = Lambda
    result[3*Nblobs + 6*k : 3*Nblobs + 6*(k+1)] = Y
    offset += b.Nblobs
  return result

def build_stochastic_block_diagonal_preconditioner(bodies, r_vectors, eta, a, *args, **kwargs):
  '''
  Build block diagonal preconditioner to generate the noise
  for rigid bodies. If the mobility of a body at the blob
  level is M=V*S*V.T we form the preconditioners
  
  P = S^{-1/2} * V.T
  P_inv = V * S^{1/2}
  
  and return the functions to compute matrix vector product
  y = (P * M * P.T) * x
  y = P_inv * x
  '''
  P = []
  P_inv = []
  for b in bodies:
    # Compute blobs mobility for one body
    mobility = b.calc_mobility_blobs(eta, a)
    
    # Compute eigenvalues and eigenvectors 
    eig_values, eig_vectors = np.linalg.eigh(mobility)
    
    # Compute the inverse of the square root of positive eigenvalues and set to zero otherwise
    eig_values_inv_sqrt = np.array([1.0/np.sqrt(x) if x > 0 else 0 for x in eig_values])
    eig_values_sqrt = np.array([np.sqrt(x) if x > 0 else 0 for x in eig_values])
    
    # Form preconditioners version P = identity matrix (no preconditioner)
    # P.append(np.eye(3 * b.Nblobs))
    # P_inv.append(np.eye(3 * b.Nblobs))

    # Form preconditioners, version P = S^{-1/2} * V.T
    # P.append(np.dot((np.eye(3 * b.Nblobs) * eig_values_inv_sqrt), eig_vectors.T))
    # P_inv.append(np.dot(eig_vectors, (np.eye(3 * b.Nblobs) * eig_values_sqrt)))
    
    # Form preconditioners version P = V * S^{-1/2} * V.T
    P.append(np.dot(eig_vectors, np.dot((np.eye(3 * b.Nblobs) * eig_values_inv_sqrt), eig_vectors.T)))
    P_inv.append(np.dot(eig_vectors, np.dot((np.eye(3 * b.Nblobs) * eig_values_sqrt), eig_vectors.T)))   
    
  # Define preconditioned mobility matrix product
  def mobility_pc(w, bodies = None, P = None, r_vectors = None, eta = None, a = None):
    result = np.empty_like(w)
    # Multiply by P.T
    offset = 0
    for k, b in enumerate(bodies):
      result[3*offset : 3*(offset + b.Nblobs)] = np.dot((P[k]).T, w[3*offset : 3*(offset + b.Nblobs)])
      offset += b.Nblobs
    # Multiply by M
    result_2 = mobility_vector_prod(r_vectors, result, eta, a)
    # Multiply by P
    offset = 0
    for k, b in enumerate(bodies):
      result[3*offset : 3*(offset + b.Nblobs)] = np.dot(P[k], result_2[3*offset : 3*(offset + b.Nblobs)])
      offset += b.Nblobs
    return result
  mobility_pc_partial = partial(mobility_pc, bodies = bodies, P = P, r_vectors = r_vectors, eta = eta, a = a)
  
  # Define inverse preconditioner P_inv
  def P_inv_mult(w, bodies = None, P_inv = None):
    offset = 0
    for k, b in enumerate(bodies):
      w[3*offset : 3*(offset + b.Nblobs)] = np.dot(P_inv[k], w[3*offset : 3*(offset + b.Nblobs)])
      offset += b.Nblobs
    return w
  P_inv_mult_partial = partial(P_inv_mult, bodies = bodies, P_inv = P_inv)

  # Return preconditioner functions
  return mobility_pc_partial, P_inv_mult_partial


if __name__ == '__main__':
  # Get command line arguments
  parser = argparse.ArgumentParser(description='Run a multi-body simulation '
                                   'with a deterministic forward Euler '
                                   'scheme and save trajectory.')
  parser.add_argument('--input-file', dest='input_file', type=str, default='data.main', 
                      help='name of the input file')
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
  structure_names = read.structure_names
  structures = read.structures
  structures_ID = read.structures_ID
  mobility_vector_prod = set_mobility_vector_prod(read.mobility_vector_prod_implementation)
  multi_bodies_functions.calc_blob_blob_forces = multi_bodies_functions.set_blob_blob_forces(read.blob_blob_force_implementation)
  multi_bodies_functions.calc_body_body_forces_torques = multi_bodies_functions.set_body_body_forces_torques(read.body_body_force_torque_implementation)

  # Copy input file to output
  subprocess.call(["cp", input_file, output_name + '.inputfile'])

  # Set random generator state
  if read.random_state is not None:
    with open(read.random_state, 'rb') as f:
      np.random.set_state(cPickle.load(f))
  elif read.seed is not None:
    np.random.seed(int(read.seed))
  
  # Save random generator state
  with open(output_name + '.random_state', 'wb') as f:
    cPickle.dump(np.random.get_state(), f)

  # Create rigid bodies
  bodies = []
  body_types = []
  for ID, structure in enumerate(structures):
    print 'Creating structures = ', structure[1]
    struct_ref_config = read_vertex_file.read_vertex_file(structure[0])
    num_bodies_struct, struct_locations, struct_orientations = read_clones_file.read_clones_file(structure[1])
    body_types.append(num_bodies_struct)
    # Creat each body of tyoe structure
    for i in range(num_bodies_struct):
      b = body.Body(struct_locations[i], struct_orientations[i], struct_ref_config, a)
      b.mobility_blobs = set_mobility_blobs(read.mobility_blobs_implementation)
      b.ID = structures_ID[ID]
      multi_bodies_functions.set_slip_by_ID(b)
      # Append bodies to total bodies list
      bodies.append(b)
  bodies = np.array(bodies)

  # Set some more variables
  num_of_body_types = len(structure_names)
  num_bodies = bodies.size
  Nblobs = sum([x.Nblobs for x in bodies])

  # Write bodies information
  with open(output_name + '.bodies_info', 'w') as f:
    f.write('num_of_body_types  ' + str(num_of_body_types) + '\n')
    f.write('body_types         ' + str(body_types) + '\n')
    f.write('num_bodies         ' + str(num_bodies) + '\n')
    f.write('num_blobs          ' + str(Nblobs) + '\n')

  # Create integrator
  if scheme.find('rollers') == -1:
    integrator = QuaternionIntegrator(bodies, Nblobs, scheme, tolerance = read.solver_tolerance) 
  else:
    integrator = QuaternionIntegratorRollers(bodies, Nblobs, scheme, tolerance = read.solver_tolerance) 
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

  integrator.calc_slip = calc_slip 
  integrator.get_blobs_r_vectors = get_blobs_r_vectors 
  integrator.mobility_blobs = set_mobility_blobs(read.mobility_blobs_implementation)
  integrator.force_torque_calculator = partial(multi_bodies_functions.force_torque_calculator_sort_by_bodies, 
                                               g = g, 
                                               repulsion_strength_wall = read.repulsion_strength_wall, 
                                               debye_length_wall = read.debye_length_wall, 
                                               repulsion_strength = read.repulsion_strength, 
                                               debye_length = read.debye_length, 
                                               periodic_length = read.periodic_length) 
  integrator.calc_K_matrix = calc_K_matrix
  integrator.linear_operator = linear_operator_rigid
  integrator.preconditioner = block_diagonal_preconditioner
  integrator.eta = eta
  integrator.a = a
  integrator.first_guess = np.zeros(Nblobs*3 + num_bodies*6)
  integrator.kT = read.kT
  integrator.mobility_vector_prod = mobility_vector_prod
  integrator.build_stochastic_block_diagonal_preconditioner = build_stochastic_block_diagonal_preconditioner
  integrator.preprocess = multi_bodies_functions.preprocess
  integrator.postprocess = multi_bodies_functions.postprocess
  integrator.periodic_length = read.periodic_length

  # Loop over time steps
  start_time = time.time()  
  for step in range(read.initial_step, n_steps):
    # Save data if...
    if (step % n_save) == 0 and step >= 0:
      elapsed_time = time.time() - start_time
      print 'Integrator = ', scheme, ', step = ', step, ', invalid configurations', integrator.invalid_configuration_count, ', wallclock time = ', time.time() - sta      # For each type of structure save locations and orientations to one file
      body_offset = 0
      if read.save_clones == 'one_file_per_step':
        for i, ID in enumerate(structures_ID):
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
        for i, ID in enumerate(structures_ID):
          name = output_name + '.' + ID + '.config'
          if step == 0:
            status = 'w'
          else:
            status = 'a'
          with open(name, status) as f_ID:
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
        print 'Error, save_clones =', read.save_clones, 'is not implemented.'
        print 'Use \"one_file_per_step\" or \"one_file\". \n'
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
        
    # Advance time step
    integrator.advance_time_step(dt)

  # Save final data if...
  if ((step+1) % n_save) == 0 and step >= 0:
    print 'Integrator = ', scheme, ', step = ', step+1, ', invalid configurations', integrator.invalid_configuration_count, ', wallclock time = ', time.time() - sta    # For each type of structure save locations and orientations to one file
    body_offset = 0
    if read.save_clones == 'one_file_per_step':
      for i, ID in enumerate(structures_ID):
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
      for i, ID in enumerate(structures_ID):
        name = output_name + '.' + ID + '.config'
        if step+1 == 0:
          status = 'w'
        else:
          status = 'a'
        with open(name, status) as f_ID:
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
      print 'Error, save_clones =', read.save_clones, 'is not implemented.'
      print 'Use \"one_file_per_step\" or \"one_file\". \n'

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
        
  # Save wallclock time 
  with open(output_name + '.time', 'w') as f:
    f.write(str(time.time() - start_time) + '\n')
  # Save number of invalid configurations
  with open(output_name + '.number_invalid_configurations', 'w') as f:
    f.write(str(integrator.invalid_configuration_count) + '\n')

  print '\n\n\n# End'
