'''
This code works as an interface to call the C library
visit_writer to write files in VTK format.
'''

import numpy as np
try:
  import visit_writer_interface
except ImportError:
  from visit import visit_writer_interface
    

def boost_write_rectilinear_mesh(name,       # File's name
                                 format_file,# 0=ASCII,  1=Binary
                                 dims,       # {mx, my, mz}
                                 xmesh,
                                 ymesh,
                                 zmesh,
                                 nvars,      # Number of variables
                                 vardims,    # Size of each variable, 1=scalar, velocity=3*scalars
                                 centering,  # Write to cell centers of corners
                                 varnames,   # Variables' names
                                 variables): # Variables

  visit_writer_interface.visit_writer_interface(name,
                                                np.array([format_file]),
                                                dims,
                                                xmesh,                                               
                                                ymesh,
                                                zmesh,
                                                np.array([nvars]),
                                                vardims,
                                                centering,
                                                varnames,
                                                variables)
  return
