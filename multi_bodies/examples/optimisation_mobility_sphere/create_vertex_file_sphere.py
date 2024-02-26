import argparse
import numpy as np
import scipy.linalg as scla
import scipy.sparse.linalg as spla
import subprocess
from functools import partial
import sys 
import time

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

def create_vertex_file_sphere(N,Rg,a):

  if N==12:
    str_Rg_orig = '0_7921'
  elif N==42:
    str_Rg_orig = '0_8913'
  elif N==162:
    str_Rg_orig = '0_9497'
  elif N==642:
    str_Rg_orig = '0_9767'
  elif N==2562:
    str_Rg_orig = '0_9888'

  orig_filename = '../../Structures/shell_N_' + str(N) + '_Rg_' + str_Rg_orig + '_Rh_1.vertex'
  struct_ref_config = read_vertex_file.read_vertex_file(orig_filename)
  Rg_orig = np.linalg.norm(struct_ref_config[0,:])
  
  struct_ref_config_new = struct_ref_config * Rg/Rg_orig
  to_save = struct_ref_config * Rg/Rg_orig
  str_Rg = str(format(Rg,'.4f')).replace('.','_')
  new_filename = '../../Structures/shell_N_' + str(N) +  '_Rg_' + str_Rg + '.vertex'
  with open(new_filename,'w') as f:
    f.write(str(int(N)) + ' ' +  str(a) + ' 0 ' + '\n')
    for sublist in to_save:
      f.write(' '.join([str(item) for item in sublist]) + '\n') 
   
  return new_filename

