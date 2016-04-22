import argparse
import numpy as np
import sys
sys.path.append('../')
import subprocess
import cPickle

from mobility import mobility as mb
from quaternion_integrator.quaternion import Quaternion
from quaternion_integrator.quaternion_integrator_gmres import QuaternionIntegratorGMRES
from quaternion_integrator.quaternion_integrator_multi_bodies import QuaternionIntegrator
from body import body 
from read_input import read_input
from read_input import read_vertex_file
from read_input import read_clones_file




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
  relaxation_steps = read.relaxation_steps
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
  for structure in structure_names:
    print 'Creating structure', structure
    struct_ref_config = read_vertex_file.read_vertex_file(structure + '.vertex')
    struct_locations, struct_orientations = read_clones_file.read_clones_file(structure + '.clones')
    # Creat each body of tyoe structure
    for i in range(len(struct_orientations)):
      b = body.Body(struct_locations[i], struct_orientations[i], struct_ref_config, a)
      # Append bodies to total bodies list
      bodies.append(b)
  bodies = np.array(bodies)

  # Set some more variables
  num_of_body_type = len(structure_names)
  num_bodies = bodies.size
  num_blobs = sum([x.Nblobs for x in bodies])


  # Write initial configuration
  # Write data to file, parameters first then trajectory.
  # with open(data_file, 'w', 1) as f:
    

  # Create integrator
  integrator = QuaternionIntegrator(bodies, scheme)

  # Loop over time steps
  for step in range(n_steps):
    print 'step = ', step
    # integrator.deterministic_forward_euler_time_step(dt)
    integrator.advance_time_step(dt)
  


    


  print '# End'
