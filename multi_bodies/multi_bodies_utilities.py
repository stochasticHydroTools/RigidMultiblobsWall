'''
This modules solves the mobility or the resistance problem for one
configuration of a multibody supensions and it can save some data like
the velocities or forces on the bodies, the mobility of a body or
the mobility of the blobs.
'''

import argparse
import numpy as np
import scipy.linalg as scla
import scipy.sparse.linalg as spla
import subprocess
from functools import partial
import sys
import time

# Find project functions
found_functions = False
path_to_append = ''
while found_functions is False:
  try:
    import multi_bodies_functions
    import multi_bodies
    from mobility import mobility as mob
    from quaternion_integrator.quaternion import Quaternion
    from quaternion_integrator.quaternion_integrator_multi_bodies import QuaternionIntegrator
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
    found_functions = True
  except ImportError:
    path_to_append += '../'
    print('searching functions in path ', path_to_append)
    sys.path.append(path_to_append)
    if len(path_to_append) > 21:
      print('\nProjected functions not found. Edit path in multi_bodies_utilities.py')
      sys.exit()

# Try to import the visit_writer (boost implementation)
try:
  import visit.visit_writer as visit_writer
except ImportError:
  pass


class gmres_counter(object):
  '''
  Callback generator to count iterations. 
  '''
  def __init__(self, print_residual = False):
    self.print_residual = print_residual
    self.niter = 0
  def __call__(self, rk=None):
    self.niter += 1
    if self.print_residual is True:
      if self.niter == 1:
        print('gmres =  0 1')
      print('gmres = ', self.niter, rk)


def plot_velocity_field(grid, r_vectors_blobs, lambda_blobs, blob_radius, eta, output, tracer_radius, *args, **kwargs):
  '''
  This function plots the velocity field to a grid. 
  '''
  # Prepare grid values
  grid = np.reshape(grid, (3,3)).T
  grid_length = grid[1] - grid[0]
  grid_points = np.array(grid[2], dtype=np.int32)
  num_points = grid_points[0] * grid_points[1] * grid_points[2]

  # Set grid coordinates
  dx_grid = grid_length / grid_points
  grid_x = np.array([grid[0,0] + dx_grid[0] * (x+0.5) for x in range(grid_points[0])])
  grid_y = np.array([grid[0,1] + dx_grid[1] * (x+0.5) for x in range(grid_points[1])])
  grid_z = np.array([grid[0,2] + dx_grid[2] * (x+0.5) for x in range(grid_points[2])])
  # Be aware, x is the fast axis.
  zz, yy, xx = np.meshgrid(grid_z, grid_y, grid_x, indexing = 'ij')
  grid_coor = np.zeros((num_points, 3))
  grid_coor[:,0] = np.reshape(xx, xx.size)
  grid_coor[:,1] = np.reshape(yy, yy.size)
  grid_coor[:,2] = np.reshape(zz, zz.size)

  # Set radius of blobs (= a) and grid nodes (= 0)
  radius_blobs = kwargs.get('radius_blobs')
  if radius_blobs is not None:
    radius_source = radius_blobs
  else:
    radius_source = np.ones(r_vectors_blobs.size // 3) * blob_radius 
  radius_target = np.ones(grid_coor.size // 3) * tracer_radius

  # Compute velocity field 
  mobility_vector_prod_implementation = kwargs.get('mobility_vector_prod_implementation')
  if mobility_vector_prod_implementation == 'python':
    grid_velocity = mob.mobility_vector_product_source_target_one_wall(r_vectors_blobs, 
                                                                       grid_coor, 
                                                                       lambda_blobs, 
                                                                       radius_source, 
                                                                       radius_target, 
                                                                       eta, 
                                                                       *args, 
                                                                       **kwargs)
  elif mobility_vector_prod_implementation == 'python_no_wall':
    grid_velocity = mob.mobility_vector_product_source_target_unbounded(r_vectors_blobs, 
                                                                        grid_coor, 
                                                                        lambda_blobs, 
                                                                        radius_source, 
                                                                        radius_target, 
                                                                        eta, 
                                                                        *args, 
                                                                        **kwargs)
  elif mobility_vector_prod_implementation == 'numba':
    grid_velocity = mob.single_wall_mobility_trans_times_force_source_target_numba(r_vectors_blobs, 
                                                                                   grid_coor, 
                                                                                   lambda_blobs, 
                                                                                   radius_source, 
                                                                                   radius_target, 
                                                                                   eta, 
                                                                                   *args, 
                                                                                   **kwargs)
  elif mobility_vector_prod_implementation == 'numba_no_wall':
    grid_velocity = mob.no_wall_mobility_trans_times_force_source_target_numba(r_vectors_blobs, 
                                                                               grid_coor, 
                                                                               lambda_blobs, 
                                                                               radius_source, 
                                                                               radius_target, 
                                                                               eta, 
                                                                               *args, 
                                                                               **kwargs)    
  elif mobility_vector_prod_implementation == 'pycuda':
    grid_velocity = mob.single_wall_mobility_trans_times_force_source_target_pycuda(r_vectors_blobs, 
                                                                                    grid_coor, 
                                                                                    lambda_blobs, 
                                                                                    radius_source, 
                                                                                    radius_target, 
                                                                                    eta, 
                                                                                    *args, 
                                                                                    **kwargs)
  else:
    print('mobility_vector_prod_implementation = ', mobility_vector_prod_implementation)
    print('The selected mobility_vector_prod_implementation cannot compute the velocity field')
    return
  
  # Prepara data for VTK writer 
  variables = [np.reshape(grid_velocity, grid_velocity.size)] 
  dims = np.array([grid_points[0]+1, grid_points[1]+1, grid_points[2]+1], dtype=np.int32) 
  nvars = 1
  vardims = np.array([3])
  centering = np.array([0])
  varnames = ['velocity\0']
  name = output + '.velocity_field.vtk'
  grid_x = grid_x - dx_grid[0] * 0.5
  grid_y = grid_y - dx_grid[1] * 0.5
  grid_z = grid_z - dx_grid[2] * 0.5
  grid_x = np.concatenate([grid_x, [grid[1,0]]])
  grid_y = np.concatenate([grid_y, [grid[1,1]]])
  grid_z = np.concatenate([grid_z, [grid[1,2]]])

  

  # Write velocity field
  visit_writer.boost_write_rectilinear_mesh(name,      # File's name
                                            0,         # 0=ASCII,  1=Binary
                                            dims,      # {mx, my, mz}
                                            grid_x,     # xmesh
                                            grid_y,     # ymesh
                                            grid_z,     # zmesh
                                            nvars,     # Number of variables
                                            vardims,   # Size of each variable, 1=scalar, velocity=3*scalars
                                            centering, # Write to cell centers of corners
                                            varnames,  # Variables' names
                                            variables) # Variables
  return


if __name__ ==  '__main__':
  # Get command line arguments
  parser = argparse.ArgumentParser(description='Solve the mobility or resistance problem'
                                   'for a multi-body suspension and save some data.')
  parser.add_argument('--input-file', dest='input_file', type=str, default='data.main', 
                      help='name of the input file')
  parser.add_argument('--print-residual', action='store_true', help='print gmres and lanczos residuals')
  args=parser.parse_args()
  input_file = args.input_file

  # Read input file
  read = read_input.ReadInput(input_file)

  # Copy input file to output
  subprocess.call(["cp", input_file, read.output_name + '.inputfile'])

  # Create rigid bodies
  bodies = []
  body_types = []
  body_names = []
  blobs_offset = 0
  for ID, structure in enumerate(read.structures):
    print('Creating structures = ', structure[1])
    struct_ref_config = read_vertex_file.read_vertex_file(structure[0])
    num_bodies_struct, struct_locations, struct_orientations = read_clones_file.read_clones_file(structure[1])
    # Read slip file if it exists
    slip = None
    if(len(structure) > 2):
      slip = read_slip_file.read_slip_file(structure[2])
    body_types.append(num_bodies_struct)
    body_names.append(read.structures_ID[ID])
    # Creat each body of tyoe structure
    for i in range(num_bodies_struct):
      b = body.Body(struct_locations[i], struct_orientations[i], struct_ref_config, read.blob_radius)
      b.mobility_blobs = multi_bodies.set_mobility_blobs(read.mobility_blobs_implementation)
      b.ID = read.structures_ID[ID]
      # Compute the blobs offset for lambda in the whole system array
      b.blobs_offset = blobs_offset
      blobs_offset += b.Nblobs
      multi_bodies_functions.set_slip_by_ID(b, slip)
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
    struct_ref_config = read_vertex_file_list.read_vertex_file_list(structure[0], read.output_name)
    num_bodies_struct, struct_locations, struct_orientations = read_clones_file.read_clones_file(structure[1])    
    constraints_info = read_constraints_file.read_constraints_file(structure[2], read.output_name)
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
      b = body.Body(struct_locations[i], struct_orientations[i], struct_ref_config[subbody], read.blob_radius)
      b.mobility_blobs = multi_bodies.set_mobility_blobs(read.mobility_blobs_implementation)
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

  # Set blob_radius array
  radius_blobs = []
  for k, b in enumerate(bodies):
    radius_blobs.append(b.blobs_radius)
  radius_blobs = np.concatenate(radius_blobs, axis=0)    

  # Set some more variables
  num_of_body_types = len(body_types)
  num_bodies = bodies.size
  Nblobs = sum([x.Nblobs for x in bodies])
  multi_bodies.mobility_vector_prod = multi_bodies.set_mobility_vector_prod(read.mobility_vector_prod_implementation, bodies=bodies) 
  multi_bodies_functions.calc_blob_blob_forces = multi_bodies_functions.set_blob_blob_forces(read.blob_blob_force_implementation)
  multi_bodies_functions.calc_body_body_forces_torques = multi_bodies_functions.set_body_body_forces_torques(read.body_body_force_torque_implementation)
  multi_bodies.mobility_blobs = multi_bodies.set_mobility_blobs(read.mobility_blobs_implementation)

  # Write bodies information
  with open(read.output_name + '.bodies_info', 'w') as f:
    f.write('num_of_body_types  ' + str(num_of_body_types) + '\n')
    f.write('body_names         ' + str(body_names) + '\n')
    f.write('body_types         ' + str(body_types) + '\n')
    f.write('num_bodies         ' + str(num_bodies) + '\n')
    f.write('num_blobs          ' + str(Nblobs) + '\n')

  # Calculate slip on blobs
  if multi_bodies.calc_slip is not None:
    slip = multi_bodies.calc_slip(bodies, Nblobs)
  else:
    slip = np.zeros((Nblobs, 3))

  # Read forces file
  if read.force_file is not None:
    force_torque = np.loadtxt(read.force_file)
    force_torque = np.reshape(force_torque, (2*num_bodies, 3))
  else:
    force_torque = np.zeros((2*num_bodies, 3))
    
  # Read velocity file
  if read.velocity_file is not None:
    velocity = np.loadtxt(read.velocity_file)
    velocity = np.reshape(velocity, (2*num_bodies, 3))
  else:
    velocity = np.zeros((2*num_bodies, 3))
    

  # If scheme == mobility solve mobility problem
  if read.scheme == 'mobility':
    start_time = time.time()  
    # Get blobs coordinates
    r_vectors_blobs = multi_bodies.get_blobs_r_vectors(bodies, Nblobs)

    # Use the code to compute force-torques on bodies if a file was not given
    if read.force_file is None:
      force_torque = multi_bodies_functions.force_torque_calculator_sort_by_bodies(bodies,
                                                                                   r_vectors_blobs,
                                                                                   g = read.g, 
                                                                                   repulsion_strength_wall = read.repulsion_strength_wall, 
                                                                                   debye_length_wall = read.debye_length_wall, 
                                                                                   repulsion_strength = read.repulsion_strength, 
                                                                                   debye_length = read.debye_length, 
                                                                                   periodic_length = read.periodic_length,
                                                                                   omega_one_roller = read.omega_one_roller)
      
    # Add the prescribed constraint velocity 
    Nconstraints = len(constraints)
    B = np.zeros((Nconstraints,3))
    if Nconstraints>0:
      for k, c in enumerate(constraints):
        B[k] = - (c.links_deriv_updated[0:3] - c.links_deriv_updated[3:6])

    # Set right hand side
    # System_size = self.Nblobs * 3 + len(self.bodies) * 6 + Nconstraints * 3 xxx
    System_size = Nblobs * 3 + num_bodies * 6 + Nconstraints * 3
    RHS = np.reshape(np.concatenate([slip.flatten(), -force_torque.flatten(), B.flatten()]), (System_size))
    
    # If prescribed velocity modify RHS
    offset = 0
    for k, b in enumerate(bodies):
      if b.prescribed_kinematics is True:
        # Add K*U to Right Hand side 
        KU = np.dot(b.calc_K_matrix(), b.calc_prescribed_velocity())
        RHS[3*offset : 3*(offset+b.Nblobs)] += KU.flatten()
        # Set F to zero
        RHS[3*Nblobs+k*6 : 3*Nblobs+(k+1)*6] = 0.0
      offset += b.Nblobs

    # Calculate K matrix
    K = multi_bodies.calc_K_matrix_bodies(bodies, Nblobs)

    # Calculate C matrix if any constraint
    if Nconstraints > 0:
      C = multi_bodies.calc_C_matrix_constraints(constraints)
    else:
      C = None
    
    # Set linear operators 
    linear_operator_partial = partial(multi_bodies.linear_operator_rigid, bodies=bodies, constraints=constraints, r_vectors=r_vectors_blobs,
                                      eta=read.eta, a=read.blob_radius, K_bodies=K, C=C, periodic_length=read.periodic_length)
    A = spla.LinearOperator((System_size, System_size), matvec = linear_operator_partial, dtype='float64')

    # 4. Pack preconditioner 
    r_vectors_blobs = multi_bodies.get_blobs_r_vectors(bodies, Nblobs)
    PC_partial = multi_bodies.build_block_diagonal_preconditioner(bodies, articulated, r_vectors_blobs, Nblobs, read.eta, read.blob_radius, step=0, update_PC=1)
    PC = spla.LinearOperator((System_size, System_size), matvec = PC_partial, dtype='float64')

    # Scale RHS to norm 1
    RHS_norm = np.linalg.norm(RHS)
    if RHS_norm > 0:
      RHS = RHS / RHS_norm
      
    # Solve preconditioned linear system # callback=make_callback()
    counter = gmres_counter(print_residual = args.print_residual)
    (sol_precond, info_precond) = utils.gmres(A, RHS, tol=read.solver_tolerance, M=PC, maxiter=1000, restart=60, callback=counter)

    # Scale RHS to norm 1
    if RHS_norm > 0:
      sol_precond = sol_precond * RHS_norm
    
    # If prescribed velocity we know the velocity
    for k, b in enumerate(bodies):
      if b.prescribed_kinematics is True:
        sol_precond[3*Nblobs + 6*k : 3*Nblobs + 6*(k+1)] = b.calc_prescribed_velocity()
    
    # Extract velocities and constraint forces on blobs
    velocity = np.reshape(sol_precond[3*Nblobs: 3*Nblobs + 6*num_bodies], (num_bodies, 6))
    lambda_blobs = np.reshape(sol_precond[0: 3*Nblobs], (Nblobs, 3))

    # Save velocity
    name = read.output_name + '.velocity.dat'
    np.savetxt(name, velocity, delimiter='  ')

    # Compute force-torques on bodies
    force = np.reshape(multi_bodies.K_matrix_T_vector_prod(bodies, lambda_blobs, Nblobs), (num_bodies, 6))
    
    # Save force
    name = read.output_name + '.force.dat'
    np.savetxt(name, force, delimiter='  ')
    print('Time to solve mobility problem =', time.time() - start_time )

    # Plot velocity field
    if read.plot_velocity_field.size > 1: 
      print('plot_velocity_field')
      plot_velocity_field(read.plot_velocity_field, r_vectors_blobs, lambda_blobs, read.blob_radius, read.eta, read.output_name, read.tracer_radius,
                          mobility_vector_prod_implementation = read.mobility_vector_prod_implementation, periodic_length=read.periodic_length, radius_blobs=radius_blobs)

  # If scheme == resistance solve resistance problem 
  elif read.scheme == 'resistance': 
    start_time = time.time() 
    # Get blobs coordinates 
    r_vectors_blobs = multi_bodies.get_blobs_r_vectors(bodies, Nblobs) 
    
    # Calculate block-diagonal matrix K
    K = multi_bodies.calc_K_matrix(bodies, Nblobs)

    # Set right hand side
    slip += multi_bodies.K_matrix_vector_prod(bodies, velocity, Nblobs) 
    RHS = np.reshape(slip, slip.size)
    
    # Calculate mobility (M) at the blob level
    mobility_blobs = multi_bodies.mobility_blobs(r_vectors_blobs, read.eta, read.blob_radius)

    # Compute constraint forces 
    force_blobs = np.linalg.solve(mobility_blobs, RHS)

    # Compute force-torques on bodies
    force = np.reshape(multi_bodies.K_matrix_T_vector_prod(bodies, force_blobs, Nblobs), (num_bodies, 6))
    
    # Save force
    name = read.output_name + '.force.dat'
    np.savetxt(name, force, delimiter='  ')
    print('Time to solve resistance problem =', time.time() - start_time  )

    # Plot velocity field
    if read.plot_velocity_field.size > 1: 
      print('plot_velocity_field')
      lambda_blobs = np.reshape(force_blobs, (Nblobs, 3))
      plot_velocity_field(read.plot_velocity_field, r_vectors_blobs, lambda_blobs, read.blob_radius, read.eta, read.output_name, read.tracer_radius,
                          mobility_vector_prod_implementation = read.mobility_vector_prod_implementation, radius_blobs=radius_blobs)
  
  elif read.scheme == 'body_mobility': 
    start_time = time.time()
    r_vectors_blobs = multi_bodies.get_blobs_r_vectors(bodies, Nblobs)
    mobility_blobs = multi_bodies.mobility_blobs(r_vectors_blobs, read.eta, read.blob_radius)
    resistance_blobs = np.linalg.inv(mobility_blobs)
    K = multi_bodies.calc_K_matrix(bodies, Nblobs)
    resistance_bodies = np.dot(K.T, np.dot(resistance_blobs, K))
    mobility_bodies = np.linalg.pinv(np.dot(K.T, np.dot(resistance_blobs, K)))
    name = read.output_name + '.body_mobility.dat'
    np.savetxt(name, mobility_bodies, delimiter='  ')
    print('Time to compute body mobility =', time.time() - start_time)
    
  elif (read.scheme == 'plot_velocity_field' and False):
    print('plot_velocity_field')
    # Compute slip 

    # Compute forces

    # Solve mobility problem

    # Compute velocity field

  print('\n\n\n# End')



      
