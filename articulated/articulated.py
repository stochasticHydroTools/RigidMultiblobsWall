'''
Small class to handle an articulated rigid body.
'''
import numpy as np
import scipy.optimize as scop
import scipy.sparse as scsp
import copy
import sys
from functools import partial
from quaternion_integrator.quaternion import Quaternion
from body import body
import general_application_utils as utils
try:
  import numexpr as ne
except ImportError:
  pass


class Articulated(object):
  '''
  Small class to handle an articulated rigid body.
  '''  
  def __init__(self, bodies, ind_bodies, constraints, ind_constraints, num_bodies, num_constraints, constraints_bodies, constraints_links, constraints_extra):
    '''
    Constructor. Take arguments like ...
    '''
    # List of the bodies in articulated rigid body and indices
    self.bodies = bodies
    self.ind_bodies = ind_bodies

    # List of the constraints in articulated rigid body and indices
    self.constraints = constraints
    self.ind_constraints = ind_constraints

    # Number of rigid bodies 
    self.num_bodies = num_bodies

    # Constraints info
    self.num_constraints = num_constraints
    self.constraints_bodies_indices = constraints_bodies
    self.constraints_links = constraints_links
    self.constraints_links_updated = np.copy(constraints_links)
    self.constraints_extra = constraints_extra

    # Center of mass position and velocity
    self.q_cm = np.zeros(3)
    self.u_cm = np.zeros(3)

    # Relative position of rigid bodies
    self.q_relative = np.zeros((self.num_bodies, 3))

    # Build connectivity matrix and pseudo inverse
    self.A = np.zeros((3 * self.num_constraints, 3 * self.num_bodies))
    for i in range(self.num_constraints):
      self.A[3 * i : 3 * (i+1), 3 * self.constraints_bodies_indices[i,0] : 3 * (self.constraints_bodies_indices[i,0]+1)] = np.eye(3)
      self.A[3 * i : 3 * (i+1), 3 * self.constraints_bodies_indices[i,1] : 3 * (self.constraints_bodies_indices[i,1]+1)] = -np.eye(3)
    self.Ainv = np.linalg.pinv(self.A)

    # Iterations counter non-linear solver
    self.nonlinear_iteration_counter = 0


  def compute_cm(self, time_point='current'):
    '''
    Compute center of mass.
    '''
    self.q_cm = np.zeros(3)
    if time_point == 'old':
      for b in self.bodies:
        self.q_cm += b.location_old
    elif time_point == 'current':
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

  
  def correct_respect_cm(self):
    '''
    Correct bodies position respect the cm.
    '''
    # Compute center of mass using relative positions
    q_cm = np.sum(self.q_relative, axis=0)
    q_cm /= self.num_bodies

    # Correct respect cm
    for k, b in enumerate(self.bodies):
      b.location = self.q_relative[k] + self.q_cm - q_cm
    return
  
    
  def solve_relative_position(self):
    '''
    Solve the relative position given the orientation.

    b = R_p * l_qp - R_q * l_pq
    '''
    # Build RHS
    b = np.zeros((self.num_constraints, 3))
    for i in range(self.num_constraints):
      b[i] = -self.constraints_links_updated[i, 0:3]
      b[i] += self.constraints_links_updated[i, 3:6]
    
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
      C[3*k:3*(k+1), 6*b1loc:6*(b1loc+1)] += C1 
      C[3*k:3*(k+1), 6*b2loc:6*(b2loc+1)] += C2
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
      g[k] = c.calc_constraint_violation(time_point='current')
    g_total = np.linalg.norm(g)
    g_total_inf = np.linalg.norm(g, ord=np.inf)

    if verbose:
      print('g_total_inf = ', g_total_inf)

    # If error is small return
    if g_total_inf < tol:
      return

    # Get bodies coordinates
    q = np.zeros((self.num_bodies, 3))
    for k, b in enumerate(self.bodies):
      q[k] = b.location

    # Define residual function
    @utils.static_var('counter', 0)
    def residual(x, q, A, links, bodies_indices, *args, **kwargs):
      residual.counter += 1
      # Extract new displacements and rotations
      num_bodies = x.size // 7
      dq = x[0 : 3 * num_bodies]
      theta = x[3 * num_bodies : ].reshape((num_bodies, 4))
      theta_norm = np.linalg.norm(theta, axis=1)

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
      return np.concatenate([g_new.flatten(), theta_norm**2 - 1]).flatten()

    # Prepare inputs for nonlinear solver
    bodies_indices = self.constraints_bodies_indices 
    xin = np.zeros(7 * len(self.bodies))
    xin[3 * len(self.bodies) :: 4] = 1.0
    
    if self.num_constraints < 1:
      def jacobian(x, links, bodies_indices, num_constraints, *args, **kwargs):
        '''
        Jacobian for rotations, dense matrix.
        '''
        # Extract new displacements and rotations
        num_bodies = x.size // 7
        theta = x[3 * num_bodies : ].reshape((num_bodies, 4))
        theta_norm = np.linalg.norm(theta, axis=1)
        # theta = theta / theta_norm[:, None]
        offset = 3 * num_bodies

        # Fill Jacobian
        J = np.zeros((3 * num_constraints + num_bodies, 7 * num_bodies))
        indx = np.arange(num_constraints, dtype=int) * 3
        indx_theta = np.arange(self.num_bodies)
        bi = bodies_indices[:,0]
        bj = bodies_indices[:,1]
        J[indx,     3 * bi]     = 1 
        J[indx + 1, 3 * bi + 1] = 1 
        J[indx + 2, 3 * bi + 2] = 1 
        J[indx,     3 * bj]     = -1 
        J[indx + 1, 3 * bj + 1] = -1 
        J[indx + 2, 3 * bj + 2] = -1 

        # S
        J[indx + 0, offset + 4 * bi + 0] =  ( 2 * links[:,2] * theta[bi,2] - 2 * links[:,1] * theta[bi,3] + 4 * links[:,0] * theta[bi,0])
        J[indx + 1, offset + 4 * bi + 0] =  (-2 * links[:,2] * theta[bi,1] + 2 * links[:,0] * theta[bi,3] + 4 * links[:,1] * theta[bi,0])
        J[indx + 2, offset + 4 * bi + 0] =  ( 2 * links[:,1] * theta[bi,1] - 2 * links[:,0] * theta[bi,2] + 4 * links[:,2] * theta[bi,0])
        # PX
        J[indx + 0, offset + 4 * bi + 1] =  (4 * links[:,0] * theta[bi,1] + 2 * links[:,1] * theta[bi,2] + 2 * links[:,2] * theta[bi,3])
        J[indx + 1, offset + 4 * bi + 1] =  (                               2 * links[:,0] * theta[bi,2] - 2 * links[:,2] * theta[bi,0]) 
        J[indx + 2, offset + 4 * bi + 1] =  (                               2 * links[:,0] * theta[bi,3] + 2 * links[:,1] * theta[bi,0])
        # PY
        J[indx + 0, offset + 4 * bi + 2] =  (2 * links[:,1] * theta[bi,1]                                + 2 * links[:,2] * theta[bi,0]) 
        J[indx + 1, offset + 4 * bi + 2] =  (2 * links[:,0] * theta[bi,1] + 4 * links[:,1] * theta[bi,2] + 2 * links[:,2] * theta[bi,3]) 
        J[indx + 2, offset + 4 * bi + 2] =  (2 * links[:,1] * theta[bi,3]                                - 2 * links[:,0] * theta[bi,0]) 
        # PZ
        J[indx + 0, offset + 4 * bi + 3] =  (2 * links[:,2] * theta[bi,1] - 2 * links[:,1] * theta[bi,0])
        J[indx + 1, offset + 4 * bi + 3] =  (2 * links[:,2] * theta[bi,2] + 2 * links[:,0] * theta[bi,0])
        J[indx + 2, offset + 4 * bi + 3] =  (2 * links[:,0] * theta[bi,1] + 2 * links[:,1] * theta[bi,2] + 4 * links[:,2] * theta[bi,3])

        # S
        J[indx + 0, offset + 4 * bj + 0] = -( 2 * links[:,5] * theta[bj, 2] - 2 * links[:,4] * theta[bj, 3] + 4 * links[:,3] * theta[bj, 0]) 
        J[indx + 1, offset + 4 * bj + 0] = -(-2 * links[:,5] * theta[bj, 1] + 2 * links[:,3] * theta[bj, 3] + 4 * links[:,4] * theta[bj, 0]) 
        J[indx + 2, offset + 4 * bj + 0] = -( 2 * links[:,4] * theta[bj, 1] - 2 * links[:,3] * theta[bj, 2] + 4 * links[:,5] * theta[bj, 0]) 
        # PX
        J[indx + 0, offset + 4 * bj + 1] = -(4 * links[:,3] * theta[bj,1] + 2 * links[:,4] * theta[bj,2] + 2 * links[:,5] * theta[bj,3])
        J[indx + 1, offset + 4 * bj + 1] = -(                               2 * links[:,3] * theta[bj,2] - 2 * links[:,5] * theta[bj,0]) 
        J[indx + 2, offset + 4 * bj + 1] = -(                               2 * links[:,3] * theta[bj,3] + 2 * links[:,4] * theta[bj,0])
        # PY
        J[indx + 0, offset + 4 * bj + 2] = -(2 * links[:,4] * theta[bj,1]                                + 2 * links[:,5] * theta[bj,0]) 
        J[indx + 1, offset + 4 * bj + 2] = -(2 * links[:,3] * theta[bj,1] + 4 * links[:,4] * theta[bj,2] + 2 * links[:,5] * theta[bj,3]) 
        J[indx + 2, offset + 4 * bj + 2] = -(2 * links[:,4] * theta[bj,3]                                - 2 * links[:,3] * theta[bj,0])
        # Pz
        J[indx + 0, offset + 4 * bj + 3] = -(2 * links[:,5] * theta[bj,1] - 2 * links[:,4] * theta[bj,0])
        J[indx + 1, offset + 4 * bj + 3] = -(2 * links[:,5] * theta[bj,2] + 2 * links[:,3] * theta[bj,0])
        J[indx + 2, offset + 4 * bj + 3] = -(2 * links[:,3] * theta[bj,1] + 2 * links[:,4] * theta[bj,2] + 4 * links[:,5] * theta[bj,3])
        J[3 * self.num_constraints + indx_theta, 3 * self.num_bodies + 4 * indx_theta + 0] = 2 * theta[indx_theta, 0]
        J[3 * self.num_constraints + indx_theta, 3 * self.num_bodies + 4 * indx_theta + 1] = 2 * theta[indx_theta, 1]
        J[3 * self.num_constraints + indx_theta, 3 * self.num_bodies + 4 * indx_theta + 2] = 2 * theta[indx_theta, 2] 
        J[3 * self.num_constraints + indx_theta, 3 * self.num_bodies + 4 * indx_theta + 3] = 2 * theta[indx_theta, 3]
        return J

    else:
      def jacobian(x, links, bodies_indices, num_constraints, *args, **kwargs):
        '''
        Jacobian for rotations, sparse matrix.
        '''
        # Extract new displacements and rotations
        num_bodies = x.size // 7
        theta = x[3 * num_bodies : ].reshape((num_bodies, 4))
        theta_norm = np.linalg.norm(theta, axis=1)
        # theta = theta / theta_norm[:, None]
        offset = 3 * num_bodies
        rows = np.zeros(num_constraints * 3 * 10 + num_bodies * 4, dtype=int)
        columns = np.zeros(num_constraints * 3 * 10 + num_bodies * 4, dtype=int)
        data = np.zeros(num_constraints * 3 * 10 + num_bodies * 4)

        # Fill Jacobian
        filled = 0
        J = np.zeros((3 * num_constraints + num_bodies, 7 * num_bodies))
        indx = np.arange(num_constraints, dtype=int) * 3
        indx_theta = np.arange(self.num_bodies)
        bi = bodies_indices[:,0]
        bj = bodies_indices[:,1]
        # J[indx,     3 * bi]     = 1
        rows[filled : filled+indx.size] = indx
        columns[filled : filled+indx.size] = 3 * bi
        data[filled : filled+indx.size] = 1
        filled += indx.size
        # J[indx + 1, 3 * bi + 1] = 1
        rows[filled : filled+indx.size] = indx + 1
        columns[filled : filled+indx.size] = 3 * bi + 1
        data[filled : filled+indx.size] = 1
        filled += indx.size        
        # J[indx + 2, 3 * bi + 2] = 1
        rows[filled : filled+indx.size] = indx + 2
        columns[filled : filled+indx.size] = 3 * bi + 2
        data[filled : filled+indx.size] = 1
        filled += indx.size
        # J[indx,     3 * bj]     = -1
        rows[filled : filled+indx.size] = indx
        columns[filled : filled+indx.size] = 3 * bj
        data[filled : filled+indx.size] = -1
        filled += indx.size
        # J[indx + 1, 3 * bj + 1] = -1
        rows[filled : filled+indx.size] = indx + 1
        columns[filled : filled+indx.size] = 3 * bj + 1
        data[filled : filled+indx.size] = -1
        filled += indx.size
        # J[indx + 2, 3 * bj + 2] = -1
        rows[filled : filled+indx.size] = indx + 2
        columns[filled : filled+indx.size] = 3 * bj + 2
        data[filled : filled+indx.size] = -1
        filled += indx.size
               
        # S
        # J[indx + 0, offset + 4 * bi + 0] =  ( 2 * links[:,2] * theta[bi,2] - 2 * links[:,1] * theta[bi,3] + 4 * links[:,0] * theta[bi,0])
        rows[filled : filled+indx.size] = indx + 0
        columns[filled : filled+indx.size] = offset + 4 * bi + 0
        data[filled : filled+indx.size] = ( 2 * links[:,2] * theta[bi,2] - 2 * links[:,1] * theta[bi,3] + 4 * links[:,0] * theta[bi,0])
        filled += indx.size        
        # J[indx + 1, offset + 4 * bi + 0] =  (-2 * links[:,2] * theta[bi,1] + 2 * links[:,0] * theta[bi,3] + 4 * links[:,1] * theta[bi,0])
        rows[filled : filled+indx.size] = indx + 1
        columns[filled : filled+indx.size] = offset + 4 * bi + 0
        data[filled : filled+indx.size] = (-2 * links[:,2] * theta[bi,1] + 2 * links[:,0] * theta[bi,3] + 4 * links[:,1] * theta[bi,0])
        filled += indx.size        
        # J[indx + 2, offset + 4 * bi + 0] =  ( 2 * links[:,1] * theta[bi,1] - 2 * links[:,0] * theta[bi,2] + 4 * links[:,2] * theta[bi,0])
        rows[filled : filled+indx.size] = indx + 2
        columns[filled : filled+indx.size] = offset + 4 * bi + 0
        data[filled : filled+indx.size] = ( 2 * links[:,1] * theta[bi,1] - 2 * links[:,0] * theta[bi,2] + 4 * links[:,2] * theta[bi,0])
        filled += indx.size 
        # PX
        # J[indx + 0, offset + 4 * bi + 1] =  (4 * links[:,0] * theta[bi,1] + 2 * links[:,1] * theta[bi,2] + 2 * links[:,2] * theta[bi,3])
        rows[filled : filled+indx.size] = indx + 0
        columns[filled : filled+indx.size] = offset + 4 * bi + 1
        data[filled : filled+indx.size] = (4 * links[:,0] * theta[bi,1] + 2 * links[:,1] * theta[bi,2] + 2 * links[:,2] * theta[bi,3])
        filled += indx.size        
        # J[indx + 1, offset + 4 * bi + 1] =  (                               2 * links[:,0] * theta[bi,2] - 2 * links[:,2] * theta[bi,0])
        rows[filled : filled+indx.size] = indx + 1
        columns[filled : filled+indx.size] = offset + 4 * bi + 1
        data[filled : filled+indx.size] = (                               2 * links[:,0] * theta[bi,2] - 2 * links[:,2] * theta[bi,0])
        filled += indx.size        
        # J[indx + 2, offset + 4 * bi + 1] =  (                               2 * links[:,0] * theta[bi,3] + 2 * links[:,1] * theta[bi,0])
        rows[filled : filled+indx.size] = indx + 2
        columns[filled : filled+indx.size] = offset + 4 * bi + 1
        data[filled : filled+indx.size] = (                               2 * links[:,0] * theta[bi,3] + 2 * links[:,1] * theta[bi,0])
        filled += indx.size        
        # PY
        # J[indx + 0, offset + 4 * bi + 2] =  (2 * links[:,1] * theta[bi,1]                                + 2 * links[:,2] * theta[bi,0])
        rows[filled : filled+indx.size] = indx + 0
        columns[filled : filled+indx.size] = offset + 4 * bi + 2
        data[filled : filled+indx.size] = (2 * links[:,1] * theta[bi,1]                                + 2 * links[:,2] * theta[bi,0])
        filled += indx.size        
        # J[indx + 1, offset + 4 * bi + 2] =  (2 * links[:,0] * theta[bi,1] + 4 * links[:,1] * theta[bi,2] + 2 * links[:,2] * theta[bi,3])
        rows[filled : filled+indx.size] = indx + 1
        columns[filled : filled+indx.size] = offset + 4 * bi + 2
        data[filled : filled+indx.size] = (2 * links[:,0] * theta[bi,1] + 4 * links[:,1] * theta[bi,2] + 2 * links[:,2] * theta[bi,3])
        filled += indx.size 
        # J[indx + 2, offset + 4 * bi + 2] =  (2 * links[:,1] * theta[bi,3]                                - 2 * links[:,0] * theta[bi,0])
        rows[filled : filled+indx.size] = indx + 2
        columns[filled : filled+indx.size] = offset + 4 * bi + 2
        data[filled : filled+indx.size] = (2 * links[:,1] * theta[bi,3]                                - 2 * links[:,0] * theta[bi,0])
        filled += indx.size 
        # PZ
        # J[indx + 0, offset + 4 * bi + 3] =  (2 * links[:,2] * theta[bi,1] - 2 * links[:,1] * theta[bi,0])
        rows[filled : filled+indx.size] = indx + 0
        columns[filled : filled+indx.size] = offset + 4 * bi + 3
        data[filled : filled+indx.size] = (2 * links[:,2] * theta[bi,1] - 2 * links[:,1] * theta[bi,0])
        filled += indx.size 
        # J[indx + 1, offset + 4 * bi + 3] =  (2 * links[:,2] * theta[bi,2] + 2 * links[:,0] * theta[bi,0])
        rows[filled : filled+indx.size] = indx + 1
        columns[filled : filled+indx.size] = offset + 4 * bi + 3
        data[filled : filled+indx.size] = (2 * links[:,2] * theta[bi,2] + 2 * links[:,0] * theta[bi,0])
        filled += indx.size 
        # J[indx + 2, offset + 4 * bi + 3] =  (2 * links[:,0] * theta[bi,1] + 2 * links[:,1] * theta[bi,2] + 4 * links[:,2] * theta[bi,3])
        rows[filled : filled+indx.size] = indx + 2
        columns[filled : filled+indx.size] = offset + 4 * bi + 3
        data[filled : filled+indx.size] = (2 * links[:,0] * theta[bi,1] + 2 * links[:,1] * theta[bi,2] + 4 * links[:,2] * theta[bi,3])
        filled += indx.size 

        # S
        # J[indx + 0, offset + 4 * bj + 0] = -( 2 * links[:,5] * theta[bj, 2] - 2 * links[:,4] * theta[bj, 3] + 4 * links[:,3] * theta[bj, 0])
        rows[filled : filled+indx.size] = indx + 0
        columns[filled : filled+indx.size] = offset + 4 * bj + 0
        data[filled : filled+indx.size] = -( 2 * links[:,5] * theta[bj, 2] - 2 * links[:,4] * theta[bj, 3] + 4 * links[:,3] * theta[bj, 0])
        filled += indx.size 
        # J[indx + 1, offset + 4 * bj + 0] = -(-2 * links[:,5] * theta[bj, 1] + 2 * links[:,3] * theta[bj, 3] + 4 * links[:,4] * theta[bj, 0])
        rows[filled : filled+indx.size] = indx + 1
        columns[filled : filled+indx.size] = offset + 4 * bj + 0
        data[filled : filled+indx.size] = -(-2 * links[:,5] * theta[bj, 1] + 2 * links[:,3] * theta[bj, 3] + 4 * links[:,4] * theta[bj, 0])
        filled += indx.size 
        # J[indx + 2, offset + 4 * bj + 0] = -( 2 * links[:,4] * theta[bj, 1] - 2 * links[:,3] * theta[bj, 2] + 4 * links[:,5] * theta[bj, 0])
        rows[filled : filled+indx.size] = indx + 2
        columns[filled : filled+indx.size] = offset + 4 * bj + 0
        data[filled : filled+indx.size] = -( 2 * links[:,4] * theta[bj, 1] - 2 * links[:,3] * theta[bj, 2] + 4 * links[:,5] * theta[bj, 0])
        filled += indx.size         
        # PX
        # J[indx + 0, offset + 4 * bj + 1] = -(4 * links[:,3] * theta[bj,1] + 2 * links[:,4] * theta[bj,2] + 2 * links[:,5] * theta[bj,3])
        rows[filled : filled+indx.size] = indx + 0
        columns[filled : filled+indx.size] = offset + 4 * bj + 1
        data[filled : filled+indx.size] = -(4 * links[:,3] * theta[bj,1] + 2 * links[:,4] * theta[bj,2] + 2 * links[:,5] * theta[bj,3])
        filled += indx.size         
        # J[indx + 1, offset + 4 * bj + 1] = -(                               2 * links[:,3] * theta[bj,2] - 2 * links[:,5] * theta[bj,0])
        rows[filled : filled+indx.size] = indx + 1
        columns[filled : filled+indx.size] = offset + 4 * bj + 1
        data[filled : filled+indx.size] = -(                               2 * links[:,3] * theta[bj,2] - 2 * links[:,5] * theta[bj,0])
        filled += indx.size         
        # J[indx + 2, offset + 4 * bj + 1] = -(                               2 * links[:,3] * theta[bj,3] + 2 * links[:,4] * theta[bj,0])
        rows[filled : filled+indx.size] = indx + 2
        columns[filled : filled+indx.size] = offset + 4 * bj + 1
        data[filled : filled+indx.size] = -(                               2 * links[:,3] * theta[bj,3] + 2 * links[:,4] * theta[bj,0])
        filled += indx.size                 
        # PY
        # J[indx + 0, offset + 4 * bj + 2] = -(2 * links[:,4] * theta[bj,1]                                + 2 * links[:,5] * theta[bj,0])
        rows[filled : filled+indx.size] = indx + 0
        columns[filled : filled+indx.size] = offset + 4 * bj + 2
        data[filled : filled+indx.size] = -(2 * links[:,4] * theta[bj,1]                                + 2 * links[:,5] * theta[bj,0])
        filled += indx.size                 
        # J[indx + 1, offset + 4 * bj + 2] = -(2 * links[:,3] * theta[bj,1] + 4 * links[:,4] * theta[bj,2] + 2 * links[:,5] * theta[bj,3])
        rows[filled : filled+indx.size] = indx + 1
        columns[filled : filled+indx.size] = offset + 4 * bj + 2
        data[filled : filled+indx.size] = -(2 * links[:,3] * theta[bj,1] + 4 * links[:,4] * theta[bj,2] + 2 * links[:,5] * theta[bj,3])
        filled += indx.size                 
        # J[indx + 2, offset + 4 * bj + 2] = -(2 * links[:,4] * theta[bj,3]                                - 2 * links[:,3] * theta[bj,0])
        rows[filled : filled+indx.size] = indx + 2
        columns[filled : filled+indx.size] = offset + 4 * bj + 2
        data[filled : filled+indx.size] = -(2 * links[:,4] * theta[bj,3]                                - 2 * links[:,3] * theta[bj,0])
        filled += indx.size                 
        # Pz
        # J[indx + 0, offset + 4 * bj + 3] = -(2 * links[:,5] * theta[bj,1] - 2 * links[:,4] * theta[bj,0])
        rows[filled : filled+indx.size] = indx + 0
        columns[filled : filled+indx.size] = offset + 4 * bj + 3
        data[filled : filled+indx.size] = -(2 * links[:,5] * theta[bj,1] - 2 * links[:,4] * theta[bj,0])
        filled += indx.size                 
        # J[indx + 1, offset + 4 * bj + 3] = -(2 * links[:,5] * theta[bj,2] + 2 * links[:,3] * theta[bj,0])
        rows[filled : filled+indx.size] = indx + 1
        columns[filled : filled+indx.size] = offset + 4 * bj + 3
        data[filled : filled+indx.size] = -(2 * links[:,5] * theta[bj,2] + 2 * links[:,3] * theta[bj,0])
        filled += indx.size                 
        # J[indx + 2, offset + 4 * bj + 3] = -(2 * links[:,3] * theta[bj,1] + 2 * links[:,4] * theta[bj,2] + 4 * links[:,5] * theta[bj,3])
        rows[filled : filled+indx.size] = indx + 2
        columns[filled : filled+indx.size] = offset + 4 * bj + 3
        data[filled : filled+indx.size] = -(2 * links[:,3] * theta[bj,1] + 2 * links[:,4] * theta[bj,2] + 4 * links[:,5] * theta[bj,3])
        filled += indx.size                 

        # J[3 * self.num_constraints + indx_theta, 3 * self.num_bodies + 4 * indx_theta + 0] = 2 * theta[indx_theta, 0]
        rows[filled : filled+indx_theta.size] = 3 * self.num_constraints + indx_theta
        columns[filled : filled+indx_theta.size] = 3 * self.num_bodies + 4 * indx_theta 
        data[filled : filled+indx_theta.size] = 2 * theta[indx_theta, 0]
        filled += indx_theta.size        
        # J[3 * self.num_constraints + indx_theta, 3 * self.num_bodies + 4 * indx_theta + 1] = 2 * theta[indx_theta, 1]
        rows[filled : filled+indx_theta.size] = 3 * self.num_constraints + indx_theta
        columns[filled : filled+indx_theta.size] = 3 * self.num_bodies + 4 * indx_theta + 1
        data[filled : filled+indx_theta.size] = 2 * theta[indx_theta, 1]
        filled += indx_theta.size  
        # J[3 * self.num_constraints + indx_theta, 3 * self.num_bodies + 4 * indx_theta + 2] = 2 * theta[indx_theta, 2]
        rows[filled : filled+indx_theta.size] = 3 * self.num_constraints + indx_theta
        columns[filled : filled+indx_theta.size] = 3 * self.num_bodies + 4 * indx_theta + 2
        data[filled : filled+indx_theta.size] = 2 * theta[indx_theta, 2]
        filled += indx_theta.size  
        # J[3 * self.num_constraints + indx_theta, 3 * self.num_bodies + 4 * indx_theta + 3] = 2 * theta[indx_theta, 3]
        rows[filled : filled+indx_theta.size] = 3 * self.num_constraints + indx_theta
        columns[filled : filled+indx_theta.size] = 3 * self.num_bodies + 4 * indx_theta + 3
        data[filled : filled+indx_theta.size] = 2 * theta[indx_theta, 3]
        filled += indx_theta.size
        return scsp.csr_matrix((data, (rows, columns)))

    # Set bounds for nonlinear solver
    bound = np.sqrt(self.num_bodies) * g_total_inf
    bounds_min = -np.ones(xin.size) * bound
    bounds_min[3 * len(self.bodies)::4] = -1 - bound
    bounds_max = np.ones(xin.size) * bound
    bounds_max[3 * len(self.bodies)::4] = 1 + bound
      
    # Call nonlinear solver
    result = scop.least_squares(residual,
                                xin,
                                verbose=(2 if verbose else 0),
                                ftol=tol,
                                xtol=tol*1e-03,
                                gtol=None,
                                method='dogbox',
                                bounds=(bounds_min, bounds_max),
                                jac=jacobian,
                                x_scale='jac',
                                max_nfev = xin.size,
                                kwargs={'q':q, 'A':self.A, 'links':self.constraints_links_updated, 'bodies_indices':bodies_indices, 'num_constraints':self.num_constraints})
    
    # Update solution
    self.nonlinear_iteration_counter += residual.counter
    x = result.x
    dq_cm = np.sum(x[0 : 3 * self.num_bodies].reshape((self.num_bodies, 3)), axis=0) / self.num_bodies
    for k, b in enumerate(self.bodies):
      dq = x[3 * k : 3 * (k+1)] - dq_cm
      theta_k = x[3 * self.num_bodies + 4 * k : 3 * self.num_bodies + 4 * (k+1)]
      quaternion_correction = Quaternion(theta_k / np.linalg.norm(theta_k))
      b.location += dq
      b.orientation = quaternion_correction * b.orientation

    # Print constraints violations
    if verbose:
      print('residual.counter = ', residual.counter)
      print('nfev             = ', result.nfev)
      print('njev             = ', result.njev)        
      print('cost             = ', result.cost)
      print('norm(x-xin)_inf  = ', np.linalg.norm(x - xin, ord=np.inf))
      print('norm(x-xin)      = ', np.linalg.norm(x - xin))
      print('g_old_inf        = ', g_total_inf)  
      print('g_old            = ', g_total)  
      print('g_inf            = ', np.linalg.norm(result.fun, ord=np.inf))
      print('g                = ', np.linalg.norm(result.fun), '\n')
    return


  def update_links(self, time=0):
    '''
    Rotate links to current orientation.
    '''

    for i in range(self.num_constraints):
      if len(self.constraints_extra[i]) == 0:
        self.constraints_links_updated[i,0:3] = np.dot(self.bodies[self.constraints_bodies_indices[i,0]].orientation.rotation_matrix(), self.constraints_links[i,0:3])
        self.constraints_links_updated[i,3:6] = np.dot(self.bodies[self.constraints_bodies_indices[i,1]].orientation.rotation_matrix(), self.constraints_links[i,3:6])
      else:
        t = time
      
        # Evaluate link and its time derivative in the body frame of reference
        self.constraints_links[i,0] = ne.evaluate(self.constraints_extra[i][0])
        self.constraints_links[i,1] = ne.evaluate(self.constraints_extra[i][1])
        self.constraints_links[i,2] = ne.evaluate(self.constraints_extra[i][2])
        self.constraints_links[i,3] = ne.evaluate(self.constraints_extra[i][3])
        self.constraints_links[i,4] = ne.evaluate(self.constraints_extra[i][4])
        self.constraints_links[i,5] = ne.evaluate(self.constraints_extra[i][5])

        # Rotate links and its derivative to the laboratory frame of reference
        self.constraints_links_updated[i,0:3] = np.dot(self.bodies[self.constraints_bodies_indices[i,0]].orientation.rotation_matrix(), self.constraints_links[i,0:3])
        self.constraints_links_updated[i,3:6] = np.dot(self.bodies[self.constraints_bodies_indices[i,1]].orientation.rotation_matrix(), self.constraints_links[i,3:6])
    return
