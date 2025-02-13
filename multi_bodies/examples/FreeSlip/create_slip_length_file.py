'''
This code write the '.slip_length' file for a spherical colloid.
The file contains the normals of the colloid surface, the slip length and the weights.
'''
from __future__ import division, print_function
import os
import numpy as np
import math as m
import time
import sys

# Find project functions
found_functions = False
path_to_append = '' 
while found_functions is False:
  try: 
    from read_input import read_vertex_file
    found_functions = True 
  except ImportError as exc:
    sys.stderr.write('Error: failed to import settings module ({})\n'.format(exc))
    path_to_append += '../'
    print('searching functions in path ', path_to_append)
    sys.path.append(path_to_append)
    if len(path_to_append) > 21:
      print('\nProjected functions not found. Edit path in create_laplace_file.py')
      sys.exit()

filename = '../../Structures/shell_N_42_Rg_1_Rh_1_1220.vertex'
output_name = '../../Structures/shell_N_42_Rg_1_slip_length_1e-09.slip_length'
slip_length = 1e-09
Rweight = 1.0

# Read file
struct_ref_config = read_vertex_file.read_vertex_file(filename)
Nb = struct_ref_config.shape[0]
Rg = np.linalg.norm(struct_ref_config[0,:])

# Extract coordinates
x = struct_ref_config[:,0]
y = struct_ref_config[:,1]
z = struct_ref_config[:,2]

# Compute normals
normals = struct_ref_config / Rg

# Slip parameter
slip_length_vec = np.ones((Nb, 1)) * slip_length

# Weights of each DOF based on a radius Rweight
weights = 4.0 * np.pi * Rweight**2 / Nb * np.ones((Nb, 1))

# Save the corresponding '.slip_length' file
to_save = np.concatenate((normals, slip_length_vec, weights), axis=1)
with open(output_name, 'w') as f_handle:
  f_handle.write('# Columns: normals, slip length, weights \n')
  np.savetxt(f_handle, to_save)
 



