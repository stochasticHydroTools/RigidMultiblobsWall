import argparse
import numpy as np
import sys
sys.path.append('../')
import subprocess
import cPickle
from functools import partial

from mobility import mobility as mb
from quaternion_integrator.quaternion import Quaternion
from quaternion_integrator.quaternion_integrator_gmres import QuaternionIntegratorGMRES
from quaternion_integrator.quaternion_integrator_multi_bodies import QuaternionIntegrator
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


def mobility_blobs(r_vectors, eta, a):
  '''
  Compute dense mobility at the blob level.
  Shape (3*Nblobs, 3*Nblobs).
  '''
  mobility =  mb.boosted_single_wall_fluid_mobility(r_vectors, eta, a)
  return mobility


def force_torque_calculator(bodies, r_vectors, g=1.0, repulsion_strength_wall=0.0, debey_length_wall=1.0):
  '''
  Return the forces and torque in each body with
  format (forces, torques) and shape (2*Nbodies, 3).
  '''
  force_torque_bodies = np.zeros((2*len(bodies), 3))
  offset = 0
  for k, b in enumerate(bodies):
    R = b.calc_rot_matrix()
    force_blobs = np.zeros((b.Nblobs, 3))
    # Compute forces on each blob
    for blob in range(b.Nblobs):
      h = r_vectors[offset+blob, 2]
      # Force on blob (wall repulsion + gravity)
      force_blobs[blob:(blob+1)] = np.array([0., 0., (repulsion_strength_wall * ((h - b.blob_radius)/debey_length_wall + 1.0) * \
                                                        np.exp(-1.0*(h - b.blob_radius)/debey_length_wall) / ((h - b.blob_radius)**2))])
      force_blobs[blob:(blob+1)] += - g * np.array([0.0, 0.0, b.blob_masses[blob]])

    # Add force to the body
    force_torque_bodies[k:(k+1)] += sum(force_blobs)
    # Add torque to the body
    force_torque_bodies[len(bodies)+k:len(bodies)+(k+1)] += np.dot(R.T, np.reshape(force_blobs, 3*b.Nblobs))
    offset += b.Nblobs
  return force_torque_bodies


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
  kT = read.kT
  scheme  = read.scheme 
  output_name = read.output_name 
  structure_names = read.structure_names
  seed = read.seed
  
  # Copy input file to output
  subprocess.call(["cp", input_file, output_name + '.inputfile'])

  # Set random generator state
  if seed is not None:
    with open(seed, 'rb') as f:
      np.random.set_state(cPickle.load(f))
  
  # Save random generator state
  with open(output_name + '.seed', 'wb') as f:
    cPickle.dump(np.random.get_state(), f)

  # Create rigid bodies
  bodies = []
  body_types = []
  for structure in structure_names:
    print 'Creating structures = ', structure
    struct_ref_config = read_vertex_file.read_vertex_file(structure + '.vertex')
    struct_locations, struct_orientations = read_clones_file.read_clones_file(structure + '.clones')
    body_types.append(len(struct_orientations))
    # Creat each body of tyoe structure
    for i in range(len(struct_orientations)):
      b = body.Body(struct_locations[i], struct_orientations[i], struct_ref_config, a)
      # Append bodies to total bodies list
      bodies.append(b)
  bodies = np.array(bodies)

  # Set some more variables
  num_of_body_types = len(structure_names)
  num_bodies = bodies.size
  num_blobs = sum([x.Nblobs for x in bodies])

  # Write bodies information
  with open(output_name + '.bodies_info', 'w') as f:
    f.write('num_of_body_types  ' + str(num_of_body_types) + '\n')
    f.write('body_types         ' + str(body_types) + '\n')
    f.write('num_bodies         ' + str(num_bodies) + '\n')
    f.write('num_blobs          ' + str(num_blobs) + '\n')

  # Create integrator
  integrator = QuaternionIntegrator(bodies, num_blobs, scheme)
  integrator.calc_slip = calc_slip
  integrator.get_blobs_r_vectors = get_blobs_r_vectors
  integrator.mobility_blobs = mobility_blobs
  integrator.force_torque_calculator = partial(force_torque_calculator, g=g)
  integrator.calc_K_matrix = calc_K_matrix
  integrator.eta = eta
  integrator.a = a

  # Open file to save configuration
  with open(output_name + '.bodies', 'w') as f:
    # Loop over time steps
    for step in range(-n_relaxation, n_steps):
      # Save data if...
      if (step % n_save) == 0 and step >= 0:
        print 'step = ', step  
        f.write(str(step * dt) + '\n') # The time
        for b in bodies:
          orientation = b.orientation.entries
          f.write('%s %s %s %s %s %s %s\n' % (b.location[0], b.location[1], b.location[2], orientation[0], orientation[1], orientation[2], orientation[3]))
      
      # integrator.deterministic_forward_euler_time_step(dt)
      integrator.advance_time_step(dt)
  
    # Save final data if...
    if ((step+1) % n_save) == 0 and step >= 0:
      print 'step = ', step+1
      f.write(str((step+1) * dt) + '\n') # The time
      for b in bodies:
        orientation = b.orientation.entries
        f.write('%s %s %s %s %s %s %s\n' % (b.location[0], b.location[1], b.location[2], orientation[0], orientation[1], orientation[2], orientation[3]))


    


  print '# End'
