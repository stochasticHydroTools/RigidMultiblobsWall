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
#filename  = "../../Structures/shell_N_12_Rg_0_7921_Rh_1"
#filename  = "../../Structures/shell_N_42_Rg_0_8913_Rh_1"
#filename = "../../Structures/shell_N_162_Rg_0_9497_Rh_1"
filename  = "../../Structures/shell_N_642_Rg_0_9767_Rh_1"
alpha = 0
k = 0
surface_mobility = 1
Rweight = 9.766578767440088e-01

# Read file
struct_ref_config = read_vertex_file.read_vertex_file(filename + '.vertex')
Nb = struct_ref_config.shape[0]
Rg = np.linalg.norm(struct_ref_config[0,:])

# Compute normals
normals = struct_ref_config / Rg
# Reaction rates
k_vec = np.zeros((Nb,1)) * k
# Emission rates
alpha_vec = np.ones((Nb,1)) * alpha
# Surface mobility
surface_mobility_vec = np.ones((Nb,1)) * surface_mobility
# Weights of each DOF based on a radius Rweight
weights = 4.0 * np.pi * Rweight**2 / Nb * np.ones((Nb,1))

# Save the corresponding '.laplace' file
to_save = np.concatenate((normals, k_vec, alpha_vec, surface_mobility_vec, weights), axis=1)
np.savetxt(filename + '_weigths_Rg.Laplace', to_save, header='Columns: normals, reaction rate, emitting rate, surface mobility, weights')
 
