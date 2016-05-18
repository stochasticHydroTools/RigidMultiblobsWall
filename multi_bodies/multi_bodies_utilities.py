'''
This modules solves the mobility or the resistance problem for one
configuration of a multibody supensions and it can save some data like
the velocities or forces on the bodies, the mobility of the bodies or
the mobility of the blobs.
'''
import argparse
import numpy as np
import scipy.linalg as sla
import scipy.sparse.linalg as spla
import subprocess
import cPickle
from functools import partial
import sys
import time
sys.path.append('../')

import multi_bodies_functions
import multi_bodies
from mobility import mobility as mb
from quaternion_integrator.quaternion import Quaternion
from quaternion_integrator.quaternion_integrator_multi_bodies import QuaternionIntegrator
from body import body 
from read_input import read_input
from read_input import read_vertex_file
from read_input import read_clones_file








if __name__ ==  '__main__':
  # Get command line arguments
  parser = argparse.ArgumentParser(description='Solve the mobility or resistance problem'
                                   'for a multi-body suspension and save some data.')
  parser.add_argument('--input-file', dest='input_file', type=str, default='data.main', 
                      help='name of the input file')
  args=parser.parse_args()
  input_file = args.input_file

  # Read input file
  read = read_input.ReadInput(input_file)

  # Copy input file to output
  subprocess.call(["cp", input_file, read.output_name + '.inputfile'])

  # Create rigid bodies
  bodies = []
  body_types = []
  for ID, structure in enumerate(read.structures):
    print 'Creating structures = ', structure[1]
    struct_ref_config = read_vertex_file.read_vertex_file(structure[0])
    num_bodies_struct, struct_locations, struct_orientations = read_clones_file.read_clones_file(structure[1])
    body_types.append(num_bodies_struct)
    # Creat each body of tyoe structure
    for i in range(num_bodies_struct):
      b = body.Body(struct_locations[i], struct_orientations[i], struct_ref_config, read.blob_radius)
      b.mobility_blobs = multi_bodies.set_mobility_blobs(read.mobility_blobs_implementation)
      b.ID = read.structures_ID[ID]
      multi_bodies_functions.set_slip_by_ID(b)
      # Append bodies to total bodies list
      bodies.append(b)
  bodies = np.array(bodies)

  # Set some more variables
  num_of_body_types = len(read.structure_names)
  num_bodies = bodies.size
  Nblobs = sum([x.Nblobs for x in bodies])
  multi_bodies.mobility_vector_prod = multi_bodies.set_mobility_vector_prod(read.mobility_vector_prod_implementation)
  multi_bodies_functions.calc_blob_blob_forces = multi_bodies_functions.set_blob_blob_forces(read.blob_blob_force_implementation)

  # Write bodies information
  with open(read.output_name + '.bodies_info', 'w') as f:
    f.write('num_of_body_types  ' + str(num_of_body_types) + '\n')
    f.write('body_types         ' + str(body_types) + '\n')
    f.write('num_bodies         ' + str(num_bodies) + '\n')
    f.write('num_blobs          ' + str(Nblobs) + '\n')

  # Read slip file
  slip = np.zeros((Nblobs, 3))
  if read.slip_file is not None:
    with open(read.slip_file, 'r') as f:
      for k, line in enumerate(f):
        slip[k] = np.array(map(float, line.split()))
  
  # Read forces file
  force_torque = np.zeros((num_bodies, 6))
  if read.force_file is not None:
    with open(read.force_file, 'r') as f:
      for k, line in enumerate(f):
        force_torque[k] = np.array(map(float, line.split()))
    force_torque = np.reshape(force_torque, (2*num_bodies, 3))

  # Read velocity file
  velocity = np.zeros((num_bodies, 6))
  if read.velocity_file is not None:
    with open(read.velocity_file, 'r') as f:
      for k, line in enumerate(f):
        velocity[k] = np.array(map(float, line.split()))
    velocity = np.reshape(velocity, (2*num_bodies, 3))


  # If scheme == mobility solve mobility problem
  if read.scheme == 'mobility':
    # Get blobs coordinates
    r_vectors_blobs = multi_bodies.get_blobs_r_vectors(bodies, Nblobs)

    # Set right hand side
    System_size = Nblobs * 3 + num_bodies * 6
    RHS = np.reshape(np.concatenate([slip, -force_torque]), (System_size))
    
    # Set linear operators 
    linear_operator_partial = partial(multi_bodies.linear_operator_rigid, bodies=bodies, r_vectors=r_vectors_blobs, eta=read.eta, a=read.blob_radius)
    A = spla.LinearOperator((System_size, System_size), matvec = linear_operator_partial, dtype='float64')

    # Set preconditioner
    mobility_inv_blobs = []
    mobility_bodies = np.empty((len(bodies), 6, 6))
    # Loop over bodies
    for k, b in enumerate(bodies):
      # 1. Compute blobs mobility and invert it
      M = b.calc_mobility_blobs(read.eta, read.blob_radius)
      M_inv = np.linalg.inv(M)
      mobility_inv_blobs.append(M_inv)
      # 2. Compute body mobility
      N = b.calc_mobility_body(read.eta, read.blob_radius, M_inv = M_inv)
      mobility_bodies[k] = N

    # 4. Pack preconditioner
    PC_partial = partial(multi_bodies.block_diagonal_preconditioner, bodies=bodies, mobility_bodies=mobility_bodies, \
                           mobility_inv_blobs=mobility_inv_blobs, Nblobs=Nblobs)
    PC = spla.LinearOperator((System_size, System_size), matvec = PC_partial, dtype='float64')

    # Solve preconditioned linear system # callback=make_callback()
    (sol_precond, info_precond) = spla.gmres(A, RHS, tol=read.solver_tolerance, M=PC, maxiter=1000, restart=60) 
    
    # Extract velocities
    velocity = np.reshape(sol_precond[3*Nblobs: 3*Nblobs + 6*num_bodies], (num_bodies, 6))

    # Save velocity
    name = read.output_name + '.velocity.dat'
    np.savetxt(name, velocity, delimiter='  ')

  # If scheme == resistance solve resistance problem
  elif read.scheme == 'resistance':
    pass

  print '\n\n\n# End'
