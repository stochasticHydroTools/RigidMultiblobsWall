'''
This code works as an interface to call the C library
visit_writer to write files in VTK format.
'''

import numpy as np
import visit_writer_interface

#def boost_write_regular_mesh(name,       # File's name
                             #format,     # 0=ASCII,  1=Binary
                             #dims,       # {mx, my, mz}
                             #nvars,      # Number of variables
                             #vardims,    # Size of each variable, 1=scalar, velocity=3*scalars
                             #centering,  # Write to cell centers of corners
                             #varnames,   # Variables' names
                             #variables): # Variables

    #print ' boost_write_regular_mesh ---- START'
    #visit_writer_interface.visit_writer_interface(name,
                                                  #format,
                                                  #dims,
                                                  #nvars,
                                                  #vardims,
                                                  #centering,
                                                  #varnames,
                                                  #variables)
    #print ' boost_write_regular_mesh ---- DONE'
    

def boost_write_rectilinear_mesh(name,       # File's name
                                 format,     # 0=ASCII,  1=Binary
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
                                                format,
                                                dims,
                                                xmesh,                                               
                                                ymesh,
                                                zmesh,
                                                nvars,
                                                vardims,
                                                centering,
                                                varnames,
                                                variables)
  
  return
