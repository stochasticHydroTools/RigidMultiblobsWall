'''
Small class to handle a single constraint. 
'''
from __future__ import division, print_function
from quaternion_integrator.quaternion import Quaternion
from body import body
import numpy as np
import copy
import sys

class Constraint(object):
  '''
  Small class to handle a single constraint.
  '''  
  def __init__(self, bodies, ind_bodies,  articulated_body, links ):
    '''
    Constructor. Take arguments like ...
    '''
    # List of the two body objects involved in the constraint
    self.bodies = bodies
    # Indices of the two bodies involved
    self.ind_bodies = ind_bodies
    # Index of articulated body to which the constraint belongs to
    self.articulated_body = articulated_body
    # 2 by 3 array that gives the (time-dependent) prescribed positions of the two links in the reference frame of the first body
    self.links = links
    # 3 by 1 array that gives the (time-dependent) prescribed velocity of the joint in the reference frame of the first body (RHS of the linear constraint problem)
    self.presc_vel = np.zeros(3)
    # Jacobian of the time-derivative of the constraint (3 by 12 matrix)
    self.C  = None


  def calc_rot_link_matrix(self):
    ''' 
    Calculate the (3 x 6) matrix [R_p x Delta l_pq]^x for the two bodies of the constraints where R_p is the rotation
    matrix of body p
    '''
    rot_link = np.zeros((3,6))

    # Compute product [R_p x Delta l_pq]
    vec = np.dot(self.bodies[0].orientation.rotation_matrix(),self.links[0])
    # Compute product [R_p x Delta l_pq]^x
    rot_link[0,1] = -vec[2]
    rot_link[0,2] =  vec[1]
    rot_link[1,0] =  vec[2]
    rot_link[1,2] = -vec[0]
    rot_link[2,0] = -vec[1]
    rot_link[2,1] =  vec[0]
    # Compute product [R_q x Delta l_qp]
    vec = np.dot(self.bodies[1].orientation.rotation_matrix(),self.links[1])
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
    return np.concatenate( ( np.eye(3), -rot_link[:,0:3], -np.eye(3), rot_link[:,3:6] ), axis=1)

