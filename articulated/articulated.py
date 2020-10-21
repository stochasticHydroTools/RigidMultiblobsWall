'''
Small class to handle an articulated rigid body.
'''
import numpy as np
import scipy.optimize as scop
import copy
import sys
from functools import partial
from quaternion_integrator.quaternion import Quaternion
from body import body
import general_application_utils as utils


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
    self.ind_constraints = ind_constraints

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
    

  def calc_C_matrix_articulated_body(self):
    '''  
    Calculate the constraint block-diagonal matrix C of an articulated body
    Shape (3*num_constraints, 6*num_bodies).
    '''
    C = np.zeros((3*self.num_constraints, 6*self.num_bodies))
    for k, c in enumerate(self.constraints):
      C_constraint = c.calc_C_matrix()
      b1loc = self.return_body_local_index(c.ind_bodies[0])
      b2loc = self.return_body_local_index(c.ind_bodies[1])
      C1 = C_constraint[:,0:6]
      C2 = C_constraint[:,6:12]
      C[3*k:3*(k+1), 6*b1loc:6*(b1loc+1)] = C1 
      C[3*k:3*(k+1), 6*b2loc:6*(b2loc+1)] = C2
    return C


  def return_body_local_index(self,ind):
    return np.where(self.ind_bodies == ind)[0][0]

 
  def non_linear_solver(self, tol=1e-08, verbose=False):
    '''
    Use nonlinear solver to enforce constraints.
    '''
    # Compute constraints violation
    g = np.zeros((self.num_constraints, 3))
    for k, c in enumerate(self.constraints):
      g[k] = c.calc_constraint_violation(time_point=1)
    g_total_old = np.linalg.norm(g)

    # If error is small return
    if g_total_old < tol:
      return

    # Get bodies coordinates
    q = np.zeros((self.num_bodies, 3))
    for k, b in enumerate(self.bodies):
      q[k] = b.location_new

    # Pre-rotate links
    links = np.zeros((self.num_constraints, 6))
    for k, c in enumerate(self.constraints):
      bodies_indices = self.constraints_info[k, 1:3].astype(int)
      links[k, 0:3] = np.dot(self.bodies[bodies_indices[0]].orientation_new.rotation_matrix(), self.constraints_info[k, 4:7])
      links[k, 3:6] = np.dot(self.bodies[bodies_indices[1]].orientation_new.rotation_matrix(), self.constraints_info[k, 7:10])

    # Define residual function
    def residual(x, q, A, links, constraints_info):
      # Extract new displacements and rotations
      bodies_indices = constraints_info[:, 1:3].astype(int)
      num_bodies = x.size // 7
      dq = x[0 : 3 * num_bodies]
      theta = x[3 * num_bodies : ].reshape((num_bodies, 4))
      theta_norm = np.linalg.norm(theta, axis=1)
      theta = theta / theta_norm[:, None]

      # Compute rotation matrices
      R = np.zeros((num_bodies, 3, 3))
      diag = theta[:,0]**2 - 0.5
      R[:,0,0] = theta[:,1]**2 + diag
      R[:,0,1] = theta[:,1] * theta[:,2] - theta[:,0] * theta[:,3]
      R[:,0,2] = theta[:,1] * theta[:,3] + theta[:,0] * theta[:,2]
      R[:,1,0] = theta[:,2] * theta[:,1] + theta[:,0] * theta[:,3]
      R[:,1,1] = theta[:,2]**2 + diag
      R[:,1,2] = theta[:,2] * theta[:,3] - theta[:,0] * theta[:,1]
      R[:,2,0] = theta[:,3] * theta[:,1] - theta[:,0] * theta[:,2]
      R[:,2,1] = theta[:,3] * theta[:,2] + theta[:,0] * theta[:,1]
      R[:,2,2] = theta[:,3]**2 + diag
      R = 2 * R
        
      # Compute new residual
      g_new = np.dot(A, dq).reshape((A.shape[0] // 3, 3)) + \
        np.einsum('kij,kj->ki', R[bodies_indices[:,0]], links[:, 0:3]) - \
        np.einsum('kij,kj->ki', R[bodies_indices[:,1]], links[:, 3:6]) + \
        (q[bodies_indices[:,0]] - q[bodies_indices[:,1]])
      return g_new.flatten()
    residual_partial = partial(residual, q=q, A=self.A, links=links, constraints_info=self.constraints_info)

    # Call nonlinear solver
    xin = np.zeros(7 * len(self.bodies))
    xin[3 * len(self.bodies) :: 4] = 1.0
    result = scop.least_squares(residual_partial, xin, verbose=(2 if verbose else 0), ftol=tol, xtol=tol, gtol=None, method='dogbox')

    # Update solution
    x = result.x
    for k, b in enumerate(self.bodies):
      dq = x[3 * k : 3 * (k+1)]
      theta_k = x[3 * self.num_bodies + 4 * k : 3 * self.num_bodies + 4 * (k+1)]
      quaternion_correction = Quaternion(theta_k / np.linalg.norm(theta_k))
      b.location_new += dq
      b.orientation_new = quaternion_correction * b.orientation_new

    # Print constraints violations
    if verbose:
      print('nfev            = ', result.nfev)
      print('njev            = ', result.njev)        
      print('cost            = ', result.cost)
      print('norm(x-xin)     = ', np.linalg.norm(x - xin))
      print('g_old           = ', g_total_old)        
      print('g               = ', np.linalg.norm(result.fun), '\n')
    return
      
    
