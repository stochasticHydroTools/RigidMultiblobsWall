'''
Write blobs configuration of an articulated body to xyz format.
'''
import numpy as np
import argparse
import sys

sys.path.append('../')
sys.path.append('./')

import multi_bodies 
from quaternion_integrator.quaternion import Quaternion
from body import body
from articulated.articulated import Articulated 
from constraint.constraint import Constraint
from read_input import read_input
from read_input import read_vertex_file
from read_input import read_clones_file
from read_input import read_constraints_file
from read_input import read_vertex_file_list


if __name__ == '__main__':

  # Inputs
  input_file = sys.argv[1]
  name_ID = sys.argv[2]
  config_file = sys.argv[3]
  print_joints = False
  print_tracker_points = False

  # Read input file
  read = read_input.ReadInput(input_file)
  n_steps = read.n_steps
  n_save = read.n_save

  # Create articulated bodies
  bodies = []
  articulated = []
  constraints = []
  body_types = []
  body_names = []
  bodies_offset = 0
  constraints_offset = 0
  num_blobs_total = 0
  for ID, structure in enumerate(read.articulated):
    if read.articulated_ID[ID] != name_ID:
      continue
    # Read vertex, clones and constraint files
    struct_ref_config = read_vertex_file_list.read_vertex_file_list(structure[0], None)
    num_bodies_struct, struct_locations, struct_orientations = read_clones_file.read_clones_file(structure[1])
    constraints_info = read_constraints_file.read_constraints_file(structure[2], None)
    num_bodies_in_articulated = constraints_info[0]
    num_constraints = constraints_info[1]
    constraints_bodies = constraints_info[2]
    constraints_links = constraints_info[3]
    constraints_extra = constraints_info[4]
    body_types.append(num_bodies_struct)
    body_names.append(read.articulated_ID[ID])
    # Create each body of type structure
    for i in range(num_bodies_struct):
      subbody = i % num_bodies_in_articulated
      b = body.Body(struct_locations[i], struct_orientations[i], struct_ref_config[subbody], read.blob_radius)
      b.ID = read.articulated_ID[ID]
      # Append bodies to total bodies list
      num_blobs_total += b.Nblobs
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


  # Read configuration
  with open(config_file, 'r') as f:
    for step in range(n_steps // n_save + 1): 
      # Read bodies
      data = f.readline()
      if data == '' or data.isspace():
        break

      num_total = num_blobs_total + (len(constraints) if print_joints else 0) + (len(bodies) if print_tracker_points else 0)
      print(num_total)
      print('#')
      # Get bodies config
      for k, b in enumerate(bodies):
        data = f.readline().split()
        b.location = [float(data[0]), float(data[1]), float(data[2])]
        b.orientation = Quaternion([float(data[3]), float(data[4]), float(data[5]), float(data[6])])
      
      if False:  
        # Get center of mass and substract it
        r_cm = np.zeros(3)
        for k, b in enumerate(bodies):
          r_cm += b.location
        r_cm = r_cm / len(bodies)
        for k, b in enumerate(bodies):
          b.location -= r_cm              
      # Print blobs      
      for k, b in enumerate(bodies):
        for ri in b.get_r_vectors():
          print('0 ', ri[0] , ri[1] , ri[2], 0, read.blob_radius)

      # Print joints
      if print_joints:
        for k, c in enumerate(constraints):
          c.update_links()
          l = bodies[c.ind_bodies[0]].location + c.links_updated[0:3]
          print('1 ', l[0] , l[1] , l[2], 1, read.blob_radius / 2)

      # Print bodies' tracker point
      if print_tracker_points:
        for b in bodies:
          print('2 ', b.location[0], b.location[1], b.location[2], 0.5, read.blob_radius * 1.25)
        
