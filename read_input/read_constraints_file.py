'''
Small module to read the constraints of articulated rigid bodies.
The format of the constraint file is:

number_of_rigid_bodies
number_of_blobs_0 number_of_blobs_1 ... number_of_blobs_N
number_of_constraints
constraint_type_0 body_i body_j number_of_parameters parameters 
.
.
.

'''
import numpy as np
from shutil import copyfile
import ntpath


def read_constraints_file(name_file, output_name):
  comment_symbols = ['#']   
  with open(name_file, 'r') as f:
    counter = 0
    constraints_info = []
    constraints_indices = []
    constraints_links = []
    constraints_extra = []
    
    for line in f:
      # Strip comments
      if comment_symbols[0] in line:
        line, comment = line.split(comment_symbols[0], 1)

      # Ignore blank lines
      line = line.strip()
      if line == '':
        continue

      if counter == 0:
        num_rigid_bodies = int(line.split()[0])
      elif counter == 1:
        num_constraints = int(line.split()[0])
      else:
        constraints_info = line.split()
        constraints_indices.append(constraints_info[0:2])
        constraints_links.append(constraints_info[2:8])
        constraints_extra.append(constraints_info[8:])
        
      # Advance counter
      counter += 1

  constraints_indices = np.array(constraints_indices, dtype=int)
  constraints_links = np.array(constraints_links, dtype=float)
    
  # Copy file to output
  if output_name is not None:
    head, tail = ntpath.split(name_file)
    copyfile(name_file, output_name + '.' + tail)

  return num_rigid_bodies, num_constraints, constraints_indices, constraints_links, constraints_extra
