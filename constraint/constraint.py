'''
Small class to handle a single constraint. 
'''
from __future__ import division, print_function
from quaternion_integrator.quaternion import Quaternion
from body import body
import numpy as np
import copy
import sys
try:
  import numexpr as ne
except ImportError:
  pass

class Constraint(object):
  '''
  Small class to handle a single constraint.
  '''  
  def __init__(self, bodies, ind_bodies,  articulated_body, links, constraint_extra=None):
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


  def calc_rot_link_matrix(self):
    ''' 
    Calculate the (3 x 6) matrix [R_p x Delta l_pq]^x for the two bodies of the constraints where R_p is the rotation
    matrix of body p
    '''
    rot_link = np.zeros((3,6))

    # Compute product [R_p x Delta l_pq]
    # vec = np.dot(self.bodies[0].orientation.rotation_matrix(),self.links[0:3])
    vec = self.links_updated[0:3]
    # Compute product [R_p x Delta l_pq]^x
    rot_link[0,1] = -vec[2]
    rot_link[0,2] =  vec[1]
    rot_link[1,0] =  vec[2]
    rot_link[1,2] = -vec[0]
    rot_link[2,0] = -vec[1]
    rot_link[2,1] =  vec[0]

    # Compute product [R_q x Delta l_qp]
    # vec = np.dot(self.bodies[1].orientation.rotation_matrix(),self.links[3:6])
    vec = self.links_updated[3:6]
    # Compute product [R_q x Delta l_qp]^x
    rot_link[0,4] = -vec[2]
    rot_link[0,5] =  vec[1]
    rot_link[1,3] =  vec[2]
    rot_link[1,5] = -vec[0]
    rot_link[2,3] = -vec[1]
    rot_link[2,4] =  vec[0]

    return rot_link


  def calc_C_matrix(self):
    '''
    Return geometric matrix C = [I -rot_link_pq -I rot_link_qp] with shape (3, 12)
    '''
    rot_link = self.calc_rot_link_matrix()
    if self.ind_bodies[0] != self.ind_bodies[1]:
      return np.concatenate( ( np.eye(3), -rot_link[:,0:3], -np.eye(3), rot_link[:,3:6] ), axis=1)
    else:
      return np.concatenate( ( np.eye(3), -rot_link[:,0:3], np.zeros((3,3)), np.zeros((3,3)) ), axis=1)


  def calc_constraint_violation(self, time_point='current'):
    '''
    Compute the constraint violation g = q_p + R_p * l_qp - q_q - R_q * l_pq.
    '''
    if time_point == 'current':
      g = self.bodies[0].location - self.bodies[1].location
      g += np.dot(self.bodies[0].orientation.rotation_matrix(), self.links[0:3])
      g -= np.dot(self.bodies[1].orientation.rotation_matrix(), self.links[3:6])
    elif time_point == 'new':
      g = self.bodies[0].location_new - self.bodies[1].location_new
      g += np.dot(self.bodies[0].orientation_new.rotation_matrix(), self.links[0:3])
      g -= np.dot(self.bodies[1].orientation_new.rotation_matrix(), self.links[3:6])      
    return g

  
  def update_links(self, time=0):
    '''
    Rotate links to current orientation.
    '''
    if len(self.constraint_extra) == 0:
      self.links_updated[0:3] = np.dot(self.bodies[0].orientation.rotation_matrix(), self.links[0:3])
      self.links_updated[3:6] = np.dot(self.bodies[1].orientation.rotation_matrix(), self.links[3:6])
    else:
      t = time

      # Evaluate link and its time derivative in the body frame of reference
      self.links[0] = ne.evaluate(self.constraint_extra[0])
      self.links[1] = ne.evaluate(self.constraint_extra[1])
      self.links[2] = ne.evaluate(self.constraint_extra[2])
      self.links[3] = ne.evaluate(self.constraint_extra[3])
      self.links[4] = ne.evaluate(self.constraint_extra[4])
      self.links[5] = ne.evaluate(self.constraint_extra[5])
      self.links_deriv[0] = ne.evaluate(self.constraint_extra[6])
      self.links_deriv[1] = ne.evaluate(self.constraint_extra[7])
      self.links_deriv[2] = ne.evaluate(self.constraint_extra[8])
      self.links_deriv[3] = ne.evaluate(self.constraint_extra[9])
      self.links_deriv[4] = ne.evaluate(self.constraint_extra[10])
      self.links_deriv[5] = ne.evaluate(self.constraint_extra[11])

      # Rotate links and its derivative to the laboratory frame of reference
      self.links_updated[0:3] = np.dot(self.bodies[0].orientation.rotation_matrix(), self.links[0:3])
      self.links_updated[3:6] = np.dot(self.bodies[1].orientation.rotation_matrix(), self.links[3:6])
      self.links_deriv_updated[0:3] = np.dot(self.bodies[0].orientation.rotation_matrix(), self.links_deriv[0:3])
      if self.ind_bodies[0] != self.ind_bodies[1]:
        self.links_deriv_updated[3:6] = np.dot(self.bodies[1].orientation.rotation_matrix(), self.links_deriv[3:6])
      else:
        self.links_deriv_updated[3:6] = 0
    return
    
