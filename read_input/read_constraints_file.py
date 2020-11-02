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
import numexpr as ne
from shutil import copyfile
import ntpath


def read_constraints_file(name_file, output_name):
  comment_symbols = ['#']   
  with open(name_file, 'r') as f:
    counter = 0
    constraints_info = []
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
        num_blobs = np.fromstring(line, sep=' ', dtype=int)
      elif counter == 2:
        num_constraints = int(line.split()[0])
      else:
        constraints_info.append(line.split())
        
      # Advance counter
      counter += 1

  constraints_info = np.array(constraints_info)
  constraints_type = constraints_info[:,0].astype(int)
  constraints_indices = constraints_info[:,1:3].astype(int)
  constraints_links = constraints_info[:, 4:10].astype(float)
  constraints_extra = constraints_info[:, 10:]
    

  # Copy file to output
  if output_name is not None:
    head, tail = ntpath.split(name_file)
    copyfile(name_file, output_name + '.' + tail)

  return num_rigid_bodies, num_blobs, num_constraints, constraints_type, constraints_indices, constraints_links, constraints_extra
