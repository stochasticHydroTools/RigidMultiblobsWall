'''
Simple example of a flagellated bacteria. 
'''
from __future__ import division, print_function
import numexpr as ne
import multi_bodies_functions
from multi_bodies_functions import *


def set_slip_by_ID_new(body, slip, *args, **kwargs):
  '''
  This functions assing a slip function to each
  body depending on his ID. The ID of a structure
  is the name of the clones file (without .clones)
  given in the input file.
  As an example we give a default function which sets the
  slip to zero and a function for active rods with an
  slip along its axis. The user can create new functions
  for other kind of active bodies.
  '''
  body.function_slip = partial(flow_resolved, *args, **kwargs)
  return
multi_bodies_functions.set_slip_by_ID = set_slip_by_ID_new


def flow_resolved(body, *args, **kwargs):
  '''
  Adds the background flow.
  '''
  # Get blobs vectors 
  r_configuration = body.get_r_vectors()
  
  # Flow along x, gradiend along z
  return  flow_resolved_coord(r_configuration, *args, **kwargs)


def flow_resolved_coord(r, *args, **kwargs):
  '''
  Use Poisseuille flow.

  IMPORTANT: edit the variables flow_magnitude and radius_effect to the desired values.
  '''
  # Set slip options
  flow_magnitude = 0
  
  # Flow along x, gradiend along z
  N = r.size // 3  
  background_flow = np.zeros((N, 3))

  # Set background flow along z-axis
  background_flow[:,0] = flow_magnitude * np.cos(2 * np.pi * r[:,2] / 100.0)
    
  return background_flow

