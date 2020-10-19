'''
Small class to handle an articulated rigid body.
'''
from quaternion_integrator.quaternion import Quaternion
from body import body
import numpy as np
import copy
import sys

class Articulated(object):
  '''
  Small class to handle an articulated rigid body.
  '''  
  def __init__(self, bodies, ind_bodies, constraints, ind_constraints, num_bodies, num_blobs, num_constraints, constraints_info):
    '''
    Constructor. Take arguments like ...
    '''
    # List of the bodies in articulated rigid body and indices
    self.bodies = bodies
    self.ind_bodies = ind_bodies

    # List of the constraints in articulated rigid body and indices
    self.constraints = constraints
    self.ind_constraints = constraints

    # Number of rigid bodies and blobs
    self.num_bodies = num_bodies
    self.num_blobs = num_blobs

    # Constraints info
    self.num_constraints = num_constraints
    self.constraints_info = constraints_info

    # Center of mass position and velocity
    self.q_cm = np.zeros(3)
    self.u_cm = np.zeros(3)

    # Relative position of rigid bodies
    self.q_relative = np.zeros((self.num_bodies, 3))

    # Build connectivity matrix and pseudo inverse
    self.A = np.zeros((3 * self.num_constraints, 3 * self.num_bodies))
    for i in range(self.num_constraints):
      bodies_indices = self.constraints_info[i, 1:3].astype(int)
      self.A[3 * i : 3 * (i+1), 3 * bodies_indices[0] : 3 * (bodies_indices[0]+1)] = np.eye(3)
      self.A[3 * i : 3 * (i+1), 3 * bodies_indices[1] : 3 * (bodies_indices[1]+1)] = -np.eye(3)
    self.Ainv = np.linalg.pinv(self.A)


  def compute_cm(self):
    '''
    Compute center of mass.
    '''
    self.q_cm = np.zeros(3)
    for b in self.bodies:
      self.q_cm += b.location
    self.q_cm /= self.num_bodies
    return self.q_cm
  

  def compute_velocity_cm(self, velocities):
    '''
    Compute center of mass velocity.
    Here velocities are all the linear and angular velocities in the system, not only in the articulated body.
    '''
    vel_art = velocities[6 * self.ind_bodies[0] : 6 * (self.ind_bodies[-1] + 1)].reshape((self.num_bodies, 6))[:,0:3]
    self.u_cm = np.sum(vel_art, axis=0) / self.num_bodies
    return self.u_cm


  def update_cm(self, dt):
    '''
    Update center of mass using Forward Euler.
    '''
    self.q_cm += dt * self.u_cm
    return self.q_cm

  
  def correct_respect_cm(self, dt):
    '''
    Correct bodies position respect the cm.
    '''
    # Compute center of mass using relative positions
    q_cm = np.sum(self.q_relative, axis=0)      
    q_cm /= self.num_bodies

    # Correct respect cm
    for k, b in enumerate(self.bodies):
      b.location_new = self.q_relative[k] + self.q_cm - q_cm
    return
  
    
  def solve_relative_position(self):
    '''
    Solve the relative position given the orientation.
    '''
    # Build RHS
    b = np.zeros((self.num_constraints, 3))
    for i in range(self.num_constraints):
      bodies_indices = self.constraints_info[i, 1:3].astype(int)
      b[i] = -np.dot(self.bodies[bodies_indices[0]].orientation_new.rotation_matrix(), self.constraints_info[i, 4:7])
      b[i] += np.dot(self.bodies[bodies_indices[1]].orientation_new.rotation_matrix(), self.constraints_info[i, 7:10])
    
    # Solve linear system
    self.q_relative = np.dot(self.Ainv, b.flatten()).reshape((self.num_bodies, 3))
    
    return self.q_relative
    

  def non_linear_solver(self):
    '''
    Use nonlinear solver to enforce constraints.
    '''
    g = np.zeros((self.num_constraints, 3))
    for k, c in enumerate(self.constraints):
      g[k] = c.calc_constraint_violation(time_point=1)

    print('g = \n', np.linalg.norm(g))
    return
