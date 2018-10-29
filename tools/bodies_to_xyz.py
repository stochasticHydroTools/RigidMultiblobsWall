'''
Write blobs configuration from location and orientation to visualize with Visit.

How to use:
python bodies_to_xyz.py input_file name_body_ID output_name_body.config > output_name_body.xyz

with
input_file: the input file used to run the simulation.
name_body_ID: name of the rigid body to plot (i.e. active_dimer or passive_dimer).
output_name_body.config: the file generated by the main code with the configuration
                         of the rigid bodies with ID name_body_ID.
'''
from __future__ import division, print_function
import numpy as np
import argparse
import sys

sys.path.append('../')

import multi_bodies 
from quaternion_integrator.quaternion import Quaternion
from body import body 
from read_input import read_input
from read_input import read_vertex_file
from read_input import read_clones_file


if __name__ == '__main__':

  # 
  input_file = sys.argv[1]
  name_ID = sys.argv[2]
  config_file = sys.argv[3]
  
  # Read input file
  read = read_input.ReadInput(input_file)
  a = read.blob_radius
  structure_names = read.structure_names
  n_steps = read.n_steps
  n_save = read.n_save

  # Create rigid bodies
  bodies = []
  body_types = []
  num_blobs_ID = 0
  for ID, structure in enumerate(read.structures):
    # print 'Creating structures = ', structure[1] 
    struct_ref_config = read_vertex_file.read_vertex_file(structure[0]) 
    num_bodies_struct, struct_locations, struct_orientations = read_clones_file.read_clones_file(structure[1]) 
    body_types.append(num_bodies_struct) 
    # Creat each body of tyoe structure 
    for i in range(len(struct_orientations)): 
      b = body.Body(struct_locations[i], struct_orientations[i], struct_ref_config, a) 
      b.ID = read.structures_ID[ID] 
      # Append bodies to total bodies list 
      bodies.append(b) 
      if b.ID == name_ID:
        num_blobs_ID += b.Nblobs
  bodies = np.array(bodies) 
  num_bodies = bodies.size 
  num_blobs = sum([x.Nblobs for x in bodies]) 

  # Read configuration 
  with open(config_file, 'r') as f: 
    for step in range(n_steps // n_save + 1): 
      # Read bodies
      data = f.readline()
      if data == '' or data.isspace():
        break
      
      print(num_blobs_ID)
      print('#')
      r_vectors = []
      for k, b in enumerate(bodies):
        if b.ID == name_ID:
          data = f.readline().split()
          b.location = [float(data[0]), float(data[1]), float(data[2])]
          b.orientation = Quaternion([float(data[3]), float(data[4]), float(data[5]), float(data[6])])
          r_vectors.append(b.get_r_vectors())
      r_vectors = np.array(r_vectors)
      r_vectors = np.reshape(r_vectors, (r_vectors.size // 3, 3))
      for i in range(len(r_vectors)):
        print(name_ID[0].upper() + ' ' + str(r_vectors[i,0]) + ' ' + str(r_vectors[i,1]) + ' ' + str(r_vectors[i,2]))


