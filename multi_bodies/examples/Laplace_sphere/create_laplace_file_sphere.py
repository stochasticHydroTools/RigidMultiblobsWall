'''
Write Laplace file. 
We use a first order Legendre polynomial for the reaction rate, emitting rate and surface mobility.
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

# Read vertex file to compute the normals and parameters
filename  = "../../Structures/shell_N_12_Rg_0_7921_Rh_1"
#filename  = "../../Structures/shell_N_42_Rg_0_8913_Rh_1"
#filename = "../../Structures/shell_N_162_Rg_0_9497_Rh_1"
#filename  = "../../Structures/shell_N_642_Rg_0_9767_Rh_1"
alpha_0 = 0
alpha_1 = 0
k_0 = 0
k_1 = 0
surface_mobility_0 = 1
surface_mobility_1 = 1
Rweight = 1

# Read file
struct_ref_config = read_vertex_file.read_vertex_file(filename + '.vertex')
Nb = struct_ref_config.shape[0]
Rg = np.linalg.norm(struct_ref_config[0,:])

# Extract coordinates
x = struct_ref_config[:,0]
y = struct_ref_config[:,1]
z = struct_ref_config[:,2]

# Get blobs polar angles
theta = np.arctan2(np.sqrt(x**2 + y**2), z)

# Compute normals
normals = struct_ref_config / Rg

# Reaction rates
k_vec = np.ones((Nb,1)) * k_0 + np.cos(theta) * k_1

# Emission rates
alpha_vec = np.ones((Nb,1)) * alpha_0 + np.cos(theta) * alpha_1

# Surface mobility
surface_mobility_vec = np.ones((Nb,1)) * surface_mobility_0 + np.cos(theta) * surface_mobility_1 

# Weights of each DOF based on a radius Rweight
weights = 4.0 * np.pi * Rweight**2 / Nb * np.ones((Nb,1))

# Save the corresponding '.laplace' file
to_save = np.concatenate((normals, k_vec, alpha_vec, surface_mobility_vec, weights), axis=1)
np.savetxt(filename + '.Laplace', to_save, header='Columns: normals, reaction rate, emitting rate, surface mobility, weights')
 
