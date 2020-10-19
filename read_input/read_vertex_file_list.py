'''
Small module to read a vertex files for a rigid articulated body.
'''
import numpy as np
from . import read_vertex_file

def read_vertex_file_list(name_files):
  comment_symbols = ['#']   
  struct_ref_config = []
  with open(name_files) as f:
    for line in f:
      # Strip comments
      if comment_symbols[0] in line:
        line, comment = line.split(comment_symbols[0], 1)

      # Ignore blank lines
      line = line.strip()
      if line != '':
        struct = read_vertex_file.read_vertex_file(line.split()[0])
        struct_ref_config.append(struct)      
    
  return struct_ref_config

