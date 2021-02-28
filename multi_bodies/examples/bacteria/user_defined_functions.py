'''
Simple example of a flagellated bacteria. 
'''
from __future__ import division, print_function
import numexpr as ne
import multi_bodies_functions
from multi_bodies_functions import *


def calc_body_body_forces_torques_python_new(bodies, r_vectors, *args, **kwargs):
  '''
  Apply constant torque in the body frame of reference to a bacteria body and
  flagellum. The total torque is zero.

  This torque only applies to bodies with body ID "bacteria_constant_torque".
  '''
  Nbodies = len(bodies)
  force_torque_bodies = np.zeros((2*len(bodies), 3))

  # Get constant torque in the body frame of reference
  torque = kwargs.get('omega_one_roller')
  
  # Loop over bodies and apply torque in the laboratory frame of reference
  constant_torque_counter = 0
  for i in range(Nbodies):
    if bodies[i].ID == 'bacteria_constant_torque':
      rotation_matrix = bodies[i].orientation.rotation_matrix()
      if constant_torque_counter == 0:
        force_torque_bodies[2*i + 1] = np.dot(rotation_matrix, torque)
        constant_torque_counter = 1
      else:
        force_torque_bodies[2*i + 1] = -np.dot(rotation_matrix, torque)
        constant_torque_counter = 0
  return force_torque_bodies
multi_bodies_functions.calc_body_body_forces_torques_python = calc_body_body_forces_torques_python_new
