'''
Simple example of a flagellated bacteria. 
'''
from __future__ import division, print_function
import numexpr as ne
import multi_bodies_functions
from multi_bodies_functions import *
from articulated.articulated import *
from constraint.constraint import *


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
    rotation_matrix = bodies[i].orientation.rotation_matrix()
    if constant_torque_counter == 0:
      force_torque_bodies[2*i + 1] = np.dot(rotation_matrix, torque)
      constant_torque_counter = 1
    else:
      force_torque_bodies[2*i + 1] = -np.dot(rotation_matrix, torque)
      constant_torque_counter = 0
  return force_torque_bodies
multi_bodies_functions.calc_body_body_forces_torques_python = calc_body_body_forces_torques_python_new


def constraint_init_new(self, bodies, ind_bodies,  articulated_body, links, constraint_extra=None):
  '''
  Constructor. Take arguments like ...
  '''
  # List of the two bodies objects involved in the constraint
  self.bodies = bodies
  # Indices of the two bodies involved
  self.ind_bodies = ind_bodies
  # Index of articulated body to which the constraint belongs to
  self.articulated_body = articulated_body
  # 6 by 1 array that gives the (time-dependent) prescribed positions of the two links in the reference frame of the first body
  self.links = links
  self.links_updated = np.copy(links)
  # 3 by 1 array that gives the (time-dependent) prescribed velocity of the joint in the reference frame of the first body (RHS of the linear constraint problem)
  self.presc_vel = np.zeros(3)
  # Jacobian of the time-derivative of the constraint (3 by 12 matrix)
  self.C  = None
  # Info for time dependent constraints and time derivative of the links
  self.constraint_extra = constraint_extra
  self.links_deriv = np.zeros(6)
  self.links_deriv_updated = np.zeros(6)

  # tumbling times for all constraints
  self.tumbling_time = -1000
Constraint.__init__ = constraint_init_new


def tumbling_event(self, t):
  if len(self.constraint_extra) > 0:
    dt              = float(self.constraint_extra[0])
    tau             = float(self.constraint_extra[1])
    tumbling_period = float(self.constraint_extra[2])

    # If there is no tumbling event active do:
    if t - self.tumbling_time > tumbling_period:
      
      # Randomly start tumbling event
      if np.random.rand(1) < 1 - np.exp(-dt / tau):
        self.tumbling_time = t
  return
Constraint.tumbling_event = tumbling_event
    

def update_links_new(self, time=0):
  '''
  Rotate links to current orientation.
  '''
    
  for i in range(self.num_constraints):   
    if len(self.constraints_extra[i]) > 0:
      # Get parameters
      tumbling_period = float(self.constraints[i].constraint_extra[2])
      max_angle       = float(self.constraints[i].constraint_extra[3])
      l00             = float(self.constraints[i].constraint_extra[4])
      l01             = float(self.constraints[i].constraint_extra[5])
      l10             = float(self.constraints[i].constraint_extra[6])
     
      # If there is a tumbling event active do:
      if time - self.constraints[i].tumbling_time <= tumbling_period:

        # Angle of the flagella respect its equilibrium direction
        alpha = max_angle * np.pi * (np.sin(np.pi * (time - self.constraints[i].tumbling_time) / tumbling_period))**2
      
        # Set tilted flagellum
        self.constraints_links[i,0] =       l01 * (np.sin(alpha))
        self.constraints_links[i,2] = l00 + l01 * (np.cos(alpha))
        self.constraints_links[i,3] = 0 
        self.constraints_links[i,5] = l10      

      else:
        self.constraints_links[i,0] = 0
        self.constraints_links[i,2] = l00 + l01 
        self.constraints_links[i,3] = 0
        self.constraints_links[i,5] = l10 

    # Rotate links derivative to the laboratory frame of reference
    self.constraints_links_updated[i,0:3] = np.dot(self.bodies[self.constraints_bodies_indices[i,0]].orientation.rotation_matrix(), self.constraints_links[i,0:3])
    self.constraints_links_updated[i,3:6] = np.dot(self.bodies[self.constraints_bodies_indices[i,1]].orientation.rotation_matrix(), self.constraints_links[i,3:6])
        
  return
Articulated.update_links = update_links_new


def update_links_new(self, time=0):
  '''
  Rotate links to current orientation.
  ''' 
  if len(self.constraint_extra) > 0:
    # Get parameters
    tumbling_period = float(self.constraint_extra[2])
    max_angle       = float(self.constraint_extra[3])
    l00             = float(self.constraint_extra[4])
    l01             = float(self.constraint_extra[5])
    l10             = float(self.constraint_extra[6])
    
    # If there is a tumbling event active do:
    if time - self.tumbling_time <= tumbling_period:

      # Angle of the flagella respect its equilibrium direction
      alpha = max_angle * np.pi * (np.sin(np.pi * (time - self.tumbling_time) / tumbling_period))**2     

      # Set tilted flagellum
      self.links[0] =       l01 * (np.sin(alpha))**2
      self.links[2] = l00 + l01 * (np.cos(alpha))
      self.links[3] = 0 
      self.links[5] = l10 

      # Set link derivative
      self.links_deriv[0] =  (1 * np.pi * l01 / tumbling_period) * np.cos(alpha)
      self.links_deriv[2] = -(1 * np.pi * l01 / tumbling_period) * np.sin(alpha) 
      self.links_deriv[3] = 0
      self.links_deriv[5] = 0
     
      # Rotate links derivative to the laboratory frame of reference
      self.links_deriv_updated[0:3] = np.dot(self.bodies[0].orientation.rotation_matrix(), self.links_deriv[0:3])
      self.links_deriv_updated[3:6] = np.dot(self.bodies[1].orientation.rotation_matrix(), self.links_deriv[3:6])

    else:
      self.links[0] = 0
      self.links[2] = l00 + l01 
      self.links[3] = 0
      self.links[5] = l10 
      self.links_deriv_updated[:] = 0

  # Rotate links and its derivative to the laboratory frame of reference
  self.links_updated[0:3] = np.dot(self.bodies[0].orientation.rotation_matrix(), self.links[0:3])
  self.links_updated[3:6] = np.dot(self.bodies[1].orientation.rotation_matrix(), self.links[3:6])
        
  return
Constraint.update_links = update_links_new


