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
#filename  = "../../Structures/shell_N_162_Rg_0_9497_Rh_1"
filename  = "../../Structures/shell_N_642_Rg_0_9767_Rh_1"
alpha_F = 0
alpha_B = 0
k_F = 5
k_B = 5
surface_mobility_F = 5
surface_mobility_B = 5

# Read file
struct_ref_config = read_vertex_file.read_vertex_file(filename + '.vertex')
Nb = struct_ref_config.shape[0]
Rg = np.linalg.norm(struct_ref_config[0,:])

# Compute normals
normals = struct_ref_config / Rg
# Reaction rates
k_vec = np.zeros((Nb,1)) 
# Emission rates
alpha_vec = np.zeros((Nb,1)) 
# Surface mobility
surface_mobility_vec = np.zeros((Nb,1)) 

for k in range(Nb):
  if struct_ref_config[k,2]>0:
    k_vec[k] = k_F
    alpha_vec[k] = alpha_F
    surface_mobility_vec[k] = surface_mobility_F
  elif struct_ref_config[k,2]==0:
    k_vec[k] = (k_B + k_F)/2
    alpha_vec[k] = (alpha_B + alpha_F)/2
    surface_mobility_vec[k] = (surface_mobility_B + surface_mobility_F)/2
  else:
    k_vec[k] = k_B
    alpha_vec[k] = alpha_B
    surface_mobility_vec[k] = surface_mobility_B
print('mean(alpha_vec) = ', np.mean(alpha_vec))
print('mean(k_vec) = ', np.mean(k_vec))
print('mean(surface_mobility_vec) = ', np.mean(surface_mobility_vec))

# Weights of each DOF based on a radius Rweight
Rweight = 1.0
weights = 4.0 * np.pi * Rweight**2 / Nb * np.ones((Nb,1))

# Save the corresponding '.laplace' file
to_save = np.concatenate((normals, k_vec, alpha_vec, surface_mobility_vec, weights), axis=1)
if alpha_F != alpha_B:
  filename += '_janus_alpha'
if k_F != k_B:
  filename += '_janus_k'
if surface_mobility_F != surface_mobility_B:
  filename += '_janus_M'
np.savetxt(filename + '.Laplace', to_save, header='Columns: normals, reaction rate, emitting rate, surface mobility, weights')
 
