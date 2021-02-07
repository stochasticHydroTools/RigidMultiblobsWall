import argparse
import numpy as np
import scipy.linalg
import scipy.spatial as spatial
import scipy.sparse.linalg as spla
import subprocess
try:
  import pickle as cpickle
except:
  try:
    import cpickle
  except:
    import _pickle as cpickle
from functools import partial
import sys
import time
import copy
import scipy.sparse as sp
from sksparse.cholmod import cholesky
#import matplotlib.pyplot as plt

# Find project functions
found_functions = False
path_to_append = ''
sys.path.append('../../')
while found_functions is False:
    try:
        from Lub_Solver import Lub_Solver as LS
        from stochastic_forcing import stochastic_forcing as stochastic
        from quaternion_integrator.quaternion import Quaternion
        from mobility import mobility as mb
        from body import body
        from read_input import read_input
        from read_input import read_vertex_file
        from read_input import read_clones_file
        from read_input import read_slip_file
        import general_application_utils
        import multi_bodies_functions

        found_functions = True
    except ImportError:
        path_to_append += '../'
        print('searching functions in path ', path_to_append)
        sys.path.append(path_to_append)
        if len(path_to_append) > 21:
            print('\nProjected functions not found. Edit path in multi_bodies.py')
            sys.exit()

if __name__ == '__main__':
    # Get command line arguments
    parser = argparse.ArgumentParser(description='Run a multi-body simulation and save trajectory.')
    parser.add_argument('--input-file', dest='input_file', type=str, default='data.main', help='name of the input file')
    parser.add_argument('--print-residual', action='store_true', help='print gmres and lanczos residuals')
    args = parser.parse_args()
    input_file = args.input_file

    # Read input file
    read = read_input.ReadInput(input_file)

    # Set some variables for the simulation
    eta = read.eta
    a = read.blob_radius
    output_name = read.output_name
    structures = read.structures
    print(structures)
    structures_ID = read.structures_ID
    
    # Copy input file to output
    subprocess.call(["cp", input_file, output_name + '.inputfile'])

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
	b.ID = structures_ID[ID]
	# Calculate body length for the RFD
	if i == 0:
	  b.calc_body_length()
	else:
	  b.body_length = bodies[-1].body_length
	# Append bodies to total bodies list
	bodies.append(b)
    bodies = np.array(bodies)

    # Set some more variables
    num_of_body_types = len(body_types)
    num_bodies = bodies.size
    num_particles = len(bodies)
    Nblobs = sum([x.Nblobs for x in bodies])
    cutoff = read.Lub_Cut
    L = read.periodic_length
    n_steps = read.n_steps 
    n_save = read.n_save
    dt = read.dt 
    
    print(L)
    
    for b in bodies:
      for i in range(3):
	if L[i] > 0:
	  while b.location[i] < 0:
	    b.location[i] += L[i]
	  while b.location[i] > L[i]:
	    b.location[i] -= L[i]
	  
    
    
    firm_delta = read.firm_delta
    debye_length_delta = 2.0*a*firm_delta/np.log(1.0e1) 
    repulsion_strength_delta = read.repulsion_strength_firm
    
    LSolv = LS(bodies,a,eta,cutoff,L,debye_length=firm_delta)
    
    kT = read.kT
    LSolv.dt = dt
    LSolv.kT = kT
    LSolv.tolerance = read.solver_tolerance
    
    multi_bodies_functions.calc_blob_blob_forces = multi_bodies_functions.set_blob_blob_forces(read.blob_blob_force_implementation)
    multi_bodies_functions.calc_body_body_forces_torques = multi_bodies_functions.set_body_body_forces_torques(read.body_body_force_torque_implementation)
    
    
    import time
    t0 = time.time()
    LSolv.Set_R_Mats()
    dt1 = time.time() - t0
    print(("Make R mats time : %s" %dt1))
    
    
    total_rej = 0
    
    #amount of time that B-field is off and chain can equilibriate
    equilib_time = 0.2
    
    for n in range(n_steps):
      print(n)
      t0 = time.time()
      time_s = n*dt
      
      mag_sign = 0.0
      if time_s > equilib_time: 
        mag_sign = -2.6 #-1.0 # -2.6
      
      FT_calc = partial(multi_bodies_functions.force_torque_calculator_sort_by_bodies, 
                                               g = read.g, 
                                               repulsion_strength_firm = repulsion_strength_delta,
                                               debye_length_firm = debye_length_delta, 
                                               firm_delta = firm_delta,
                                               repulsion_strength_wall = read.repulsion_strength_wall,
                                               debye_length_wall = read.debye_length_wall,
                                               repulsion_strength = read.repulsion_strength, 
                                               debye_length = read.debye_length, 
                                               periodic_length = read.periodic_length,
                                               spring_k = 10.0, #3.5,
                                               spring_l = 2.1*a,
                                               k_bend = 10.0*(2.1*a)*num_particles*kT,  #2.0*(2.1*a)*num_particles*kT
                                               k_twist = 0.1*(2.1*a)*num_particles*kT, #1.0*(2.1*a)*num_particles*kT
                                               k_twist_bend = 0.0*(2.1*a)*num_particles*kT,
                                               mag_force = mag_sign*1.0,
                                               beta_ang = 62.0*np.pi/180.0, #62.0*np.pi/180.0 worked and gave a semi stable helix #45.0*np.pi/180.0 worked for braid
                                               B_field_freq = 50, 
                                               time_s = time_s,
                                               a = a)
      
      dt1 = time.time() - t0
      r_vecs = np.array([b.location for b in bodies])

      
      t0 = time.time()
      reject_wall, reject_jump = LSolv.Update_Bodies_Trap(FT_calc)
      dt1 = time.time() - t0
      print(("walltime for time step : %s" %dt1))
      print(("Number of rejected timesteps wall: %s" %LSolv.num_rejections_wall))
      print(("Number of rejected timesteps jump: %s" %LSolv.num_rejections_jump))
      
      if ((reject_wall+reject_jump) == 0) and (n % n_save == 0):
	body_offset = 0
	for i, ID in enumerate(structures_ID):
	  name = output_name + '.' + ID + '.config'
	  if n == 0:
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
	total_rej += 1
	body_offset = 0
	for i, ID in enumerate(structures_ID):
	  name = output_name + '.' + ID + '.rejected_config'
	  if total_rej == 1:
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
