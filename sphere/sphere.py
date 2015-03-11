''' Mobilities and other useful functions for sphere. '''
import numpy as np
import os
import sys
sys.path.append('..')
from fluids import mobility as mb

# Make sure figures folder exists
if not os.path.isdir(os.path.join(os.getcwd(), 'figures')):
  os.mkdir(os.path.join(os.getcwd(), 'figures'))
# Make sure data folder exists
if not os.path.isdir(os.path.join(os.getcwd(), 'data')):
  os.mkdir(os.path.join(os.getcwd(), 'data'))
# Make sure logs folder exists
if not os.path.isdir(os.path.join(os.getcwd(), 'logs')):
  os.mkdir(os.path.join(os.getcwd(), 'logs'))

#Parameters
ETA = 1.0  # Viscosity.
A = 0.5    # Radius of sphere.
M  = 0.5   # Mass*g of sphere.
H = 3.5    # Initial Distance from Wall.
KT = 0.2   # Temperature.
# Parameters for Yukawa potential
REPULSION_STRENGTH = 2.0
DEBYE_LENGTH = 0.25  

def sphere_check_function(location, orientation):
  ''' Check that sphere is not overlapping the wall. '''
  if location[0][2] < A:
    return False
  else:
    return True
  

def null_torque_calculator(location, orientation):
  return [0., 0., 0.]


def sphere_force_calculator(location, orientation):
  gravity = -1*M
  h = location[0][2]
  repulsion = (REPULSION_STRENGTH*((h - A)/DEBYE_LENGTH + 1)*
               np.exp(-1.*(h - A)/DEBYE_LENGTH)/((h - A)**2))
  return [0., 0., gravity + repulsion]


def sphere_mobility(location, orientation):
  location = location[0]
  fluid_mobility = mb.single_wall_self_mobility_with_rotation(location, ETA, A)
  return fluid_mobility

