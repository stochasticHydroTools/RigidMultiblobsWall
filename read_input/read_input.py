'''
Simple class to read the input files to run a simulation.
'''

import numpy as np
import ntpath
import sys

class ReadInput(object):
  '''
  Simple class to read the input files to run a simulation.
  '''

  def __init__(self, entries):
    ''' Construnctor takes the name of the input file '''
    self.entries = entries
    self.input_file = entries
    self.options = {}
    number_of_structures = 0 
    number_of_obstacles = 0 
    number_of_articulated = 0 

    # Read input file
    comment_symbols = ['#']   
    with open(self.input_file, 'r') as f:
      # Loop over lines
      for line in f:
        # Strip comments
        if comment_symbols[0] in line:
          line, comment = line.split(comment_symbols[0], 1)

        # Save options to dictionary, Value may be more than one word
        line = line.strip()
        if line != '':
          option, value = line.split(None, 1)
          if option == 'structure':
            option += str(number_of_structures)
            number_of_structures += 1
          if option == 'obstacle':
            option += str(number_of_obstacles)
            number_of_obstacles += 1
          if option == 'articulated':
            option += str(number_of_articulated)
            number_of_articulated += 1
          self.options[option] = value

    # Set option to file or default values
    self.n_steps = int(self.options.get('n_steps') or 0)
    self.initial_step = int(self.options.get('initial_step') or 0)
    self.n_save = int(self.options.get('n_save') or 1)
    self.n_relaxation = int(self.options.get('n_relaxation') or 0)
    self.dt = float(self.options.get('dt') or 0.0)
    self.eta = float(self.options.get('eta') or 1.0)
    self.g = float(self.options.get('g') or 1.0)
    self.theta = float(self.options.get('tilt_angle') or 0.0)
    self.blob_radius = float(self.options.get('blob_radius') or 1.0)
    self.tracer_radius = float(self.options.get('tracer_radius') or 0.0)
    self.kT = float(self.options.get('kT') or 1.0)
    self.scheme = str(self.options.get('scheme') or 'deterministic_forward_euler')
    self.output_name = str(self.options.get('output_name') or 'run')
    self.random_state = self.options.get('random_state')
    self.seed = self.options.get('seed')
    self.repulsion_strength_wall = float(self.options.get('repulsion_strength_wall') or 1.0)
    self.debye_length_wall = float(self.options.get('debye_length_wall') or 1.0)
    self.mobility_blobs_implementation = str(self.options.get('mobility_blobs_implementation') or 'python')
    self.mobility_vector_prod_implementation = str(self.options.get('mobility_vector_prod_implementation') or 'python')
    self.repulsion_strength = float(self.options.get('repulsion_strength') or 1.0)
    self.debye_length = float(self.options.get('debye_length') or 1.0)
    self.blob_blob_force_implementation = str(self.options.get('blob_blob_force_implementation') or 'None')
    self.body_body_force_torque_implementation = str(self.options.get('body_body_force_torque_implementation') or 'None')
    self.save_body_mobility = str(self.options.get('save_body_mobility') or 'False')
    self.save_blobs_mobility = str(self.options.get('save_blobs_mobility') or 'False')
    self.save_velocities = str(self.options.get('save_velocities') or 'False')
    self.slip_file = self.options.get('slip_file')
    self.force_file = self.options.get('force_file')
    self.velocity_file = self.options.get('velocity_file')
    self.solver_tolerance = float(self.options.get('solver_tolerance') or 1e-08)
    self.nonlinear_solver_tolerance = float(self.options.get('nonlinear_solver_tolerance') or 1e-08)
    self.rf_delta = float(self.options.get('rf_delta') or 1e-03)
    self.save_clones = str(self.options.get('save_clones') or 'one_file_per_step')
    self.periodic_length = np.fromstring(self.options.get('periodic_length') or '0 0 0', sep=' ')
    self.omega_one_roller = np.fromstring(self.options.get('omega_one_roller') or '0 0 0', sep=' ')
    self.free_kinematics = str(self.options.get('free_kinematics') or 'True')
    self.plot_velocity_field = np.fromstring(self.options.get('plot_velocity_field') or 'None', sep=' ')
    self.green_particles = np.fromstring(self.options.get('green_particles') or '0 0', sep=' ', dtype=int)          
    self.cells = np.fromstring(self.options.get('cells') or '1 1', sep=' ', dtype=int)
    self.sample_HydroGrid = int(self.options.get('sample_HydroGrid') or 1)
    self.save_HydroGrid = int(self.options.get('save_HydroGrid') or 0)
    self.hydro_interactions = int(self.options.get('hydro_interactions') or 1)    
    self.update_PC = int(self.options.get('update_PC') or 1)
    self.domain = str(self.options.get('domain') or 'single_wall')
    self.call_HydroGrid = str(self.options.get('call_HydroGrid') or 'False') == 'True'
    self.repulsion_strength_firm = float(self.options.get('repulsion_strength_firm') or 0.0)
    self.firm_delta = float(self.options.get('firm_delta') or 1e-02)
    self.Lub_Cut = float(self.options.get('Lub_Cut') or 4.5)
    self.zmin = float(self.options.get('zmin') or 0)
    self.zmax = float(self.options.get('zmax') or 1e7)
    self.domType = str(self.options.get('domType') or 'RPB')

    # Create list with [vertex_file, clones_file] for each structure
    self.num_free_bodies = number_of_structures
    self.structures = []
    self.structures_ID = []
    self.articulated = []
    self.articulated_ID = []
    for i in range(number_of_structures):
      option = 'structure' + str(i)
      structure_files = str.split(str(self.options.get(option)))
      self.structures.append(structure_files)

    # Create list with [vertex_file, clones_file] for each obstacle
    for i in range(number_of_obstacles):
      option = 'obstacle' + str(i)
      structure_files = str.split(str(self.options.get(option)))
      self.structures.append(structure_files)

    # Create list with [vertex_file, clones_file, contraints_file] for each articulated
    for i in range(number_of_articulated):
      option = 'articulated' + str(i)
      structure_files = str.split(str(self.options.get(option)))
      head, tail = ntpath.split(structure_files[1])
      # then, remove end (.clones)
      tail = tail[:-7]
      self.articulated_ID.append(tail)
      self.articulated.append(structure_files)
      
    # Create structures ID for each kind 
    for struct in self.structures:
      # First, remove directory from structure name
      head, tail = ntpath.split(struct[1])
      # then, remove end (.clones)
      tail = tail[:-7]
      self.structures_ID.append(tail)

    # If we are restarting a simulation (initial_step > 0)
    # look for the .clones file in the output directory
    if self.initial_step > 0:
      for k, struct in enumerate(self.structures):
        recovery_file = self.output_name + '.'  + self.structures_ID[k] + '.' + str(self.initial_step).zfill(8) + '.clones'
        struct[1] = recovery_file

    # Obstacles are not implemented in some schemes
    if number_of_obstacles > 0:
      if (self.scheme == 'deterministic_forward_euler_dense_algebra') or \
         (self.scheme == 'stochastic_first_order_RFD') or \
         (self.scheme == 'stochastic_adams_bashforth') or \
         (self.scheme == 'stochastic_first_order_RFD_dense_algebra') or \
         (self.scheme == 'stochastic_traction_EM') or \
         (self.scheme == 'Fixman') or \
         (self.scheme == 'stochastic_traction_AB') or \
         (self.scheme == 'stochastic_Slip_Mid_DLA'):
        print('Obstacles are not implemented for scheme: ', self.scheme)
        sys.exit()

    return
