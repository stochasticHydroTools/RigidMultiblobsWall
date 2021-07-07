'''
Small module to read a vertex files for a rigid articulated body.
'''
import numpy as np
from shutil import copyfile
import ntpath
from . import read_vertex_file

def read_vertex_file_list(name_files, output_name):
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

  # Copy file to output
  if output_name is not None:
    head, tail = ntpath.split(name_files)
    copyfile(name_files, output_name + '.' + tail)
  return struct_ref_config

