'''
Integrator for several rigid bodies.
'''
import numpy as np
import math as m
import scipy.sparse.linalg as spla
from functools import partial

from quaternion import Quaternion
from stochastic_forcing import stochastic_forcing as stochastic

class QuaternionIntegrator(object):
  '''
  Integrator that timesteps using deterministic forwars Euler scheme.
  '''  
  def __init__(self, bodies, Nblobs, scheme): 
    ''' 
    Init object 
    '''
    self.bodies = bodies
    self.Nblobs = Nblobs
    self.scheme = scheme
    self.mobility_bodies = np.empty((len(bodies), 6, 6))

    # Other variables
    self.get_blobs_r_vectors = None
    self.mobility_blobs = None
    self.force_torque_calculator = None
    self.calc_K_matrix = None
    self.linear_operator = None
    self.eta = None
    self.a = None
    self.velocities = None
    self.velocities_previous_step = None
    self.first_step = True
    self.rf_delta = 1e-06
    self.kT = 0.0

    # Optional variables
    self.calc_slip = None
    self.calc_force_torque = None
    self.mobility_inv_blobs = None
    self.first_guess = None
    self.preconditioner = None
    
    return 

  def advance_time_step(self, dt):
    '''
    Advance time step with integrator self.scheme
    '''
    return getattr(self, self.scheme)(dt)
    

  def deterministic_forward_euler(self, dt): 
    ''' 
    Take a time step of length dt using the deterministic forward Euler scheme. 
    The function uses gmres to solve the rigid body equations.

    The linear and angular velocities are sorted lile
    velocities = (v_1, w_1, v_2, w_2, ...)
    where v_i and w_i are the linear and angular velocities of body i.
    ''' 
    while True: 
      # Solve mobility problem
      velocities = self.solve_mobility_problem()

      # Update location orientation 
      for k, b in enumerate(self.bodies):
        b.location_new = b.location + velocities[6*k:6*k+3] * dt
        quaternion_dt = Quaternion.from_rotation((velocities[6*k+3:6*k+6]) * dt)
        b.orientation_new = quaternion_dt * b.orientation
        
      # Check positions, if valid return 
      valid_configuration = True
      for b in self.bodies:
        valid_configuration = b.check_function(b.location_new, b.orientation_new)
        if valid_configuration is False:
          break
      if valid_configuration is True:
        for b in self.bodies:
          b.location = b.location_new
          b.orientation = b.orientation_new
        return

      print 'Invalid configuration'
      return
      

  def deterministic_forward_euler_dense_algebra(self, dt): 
    ''' 
    Take a time step of length dt using the deterministic forward Euler scheme. 
    The function uses dense algebra methods to solve the equations.
    
    The linear and angular velocities are sorted lile
    velocities = (v_1, w_1, v_2, w_2, ...)
    where v_i and w_i are the linear and angular velocities of body i.
    ''' 
    while True: 
      # Solve mobility problem
      velocities, mobility_bodies = self.solve_mobility_problem_dense_algebra()

      # Update location orientation 
      for k, b in enumerate(self.bodies):
        b.location_new = b.location + velocities[6*k:6*k+3] * dt
        quaternion_dt = Quaternion.from_rotation((velocities[6*k+3:6*k+6]) * dt)
        b.orientation_new = quaternion_dt * b.orientation
        
      # Check positions, if valid return 
      valid_configuration = True
      for b in self.bodies:
        valid_configuration = b.check_function(b.location_new, b.orientation_new)
        if valid_configuration is False:
          break
      if valid_configuration is True:
        for b in self.bodies:
          b.location = b.location_new
          b.orientation = b.orientation_new
        return

      print 'Invalid configuration'
      return
      
  
  def deterministic_adams_bashforth(self, dt): 
    ''' 
    Take a time step of length dt using the deterministic Adams-Bashforth of
    order two scheme. The function uses gmres to solve the rigid body equations.

    The linear and angular velocities are sorted lile
    velocities = (v_1, w_1, v_2, w_2, ...)
    where v_i and w_i are the linear and angular velocities of body i.
    ''' 
    while True: 
      # Solve mobility problem
      velocities = self.solve_mobility_problem()

      # Update location and orientation
      if self.first_step == False:
        # Use Adams-Bashforth
        for k, b in enumerate(self.bodies):
          b.location_new = b.location + (1.5 * velocities[6*k:6*k+3] - 0.5 * self.velocities_previous_step[6*k:6*k+3]) * dt
          quaternion_dt = Quaternion.from_rotation((1.5 * velocities[6*k+3:6*k+6] \
                                                      - 0.5 * self.velocities_previous_step[6*k+3:6*k+6]) * dt)
          b.orientation_new = quaternion_dt * b.orientation
      else:
        # Use forward Euler
        for k, b in enumerate(self.bodies):
          b.location_new = b.location + velocities[6*k:6*k+3] * dt
          quaternion_dt = Quaternion.from_rotation((velocities[6*k+3:6*k+6]) * dt)
          b.orientation_new = quaternion_dt * b.orientation              

      # Check positions, if valid return 
      valid_configuration = True
      for b in self.bodies:
        valid_configuration = b.check_function(b.location_new, b.orientation_new)
        if valid_configuration is False:
          break
      if valid_configuration is True:
        # Save velocities for next step
        self.first_step = False
        self.velocities_previous_step = velocities
        for b in self.bodies:
          b.location = b.location_new
          b.orientation = b.orientation_new          
        return
    
    print 'Invalid configuration'      
    return


  def stochastic_first_order_RFD_dense_algebra(self, dt): 
    ''' 
    Take a time step of length dt using a stochastic
    first order Randon Finite Difference (RFD) schame.
    The function uses dense algebra methods to solve the equations.

    The linear and angular velocities are sorted lile
    velocities = (v_1, w_1, v_2, w_2, ...)
    where v_i and w_i are the linear and angular velocities of body i.
    ''' 
    while True: 
      # Solve mobility problem
      velocities, mobility_bodies = self.solve_mobility_problem_dense_algebra()

      # Generate random vector
      rfd_noise = np.random.normal(0.0, 1.0, len(self.bodies) * 6)     

      # Add noise contribution sqrt(2kT/dt)*N^{1/2}*W
      # velocities += stochastic.stochastic_forcing_eig(mobility_bodies, factor = np.sqrt(2*self.kT / dt))
      velocities += stochastic.stochastic_forcing_cholesky(mobility_bodies, factor = np.sqrt(2*self.kT / dt))

      # Update configuration for rfd
      for k, b in enumerate(self.bodies):
        b.location_new = b.location + rfd_noise[k*6 : k*6+3] * self.rf_delta * self.blob_radius
        quaternion_dt = Quaternion.from_rotation(rfd_noise[(k*6+3):(k*6+6)] * self.rf_delta * self.blob_radius)
        b.orientation_new = quaternion_dt * b.orientation

      # Compute bodies' mobility at new configuration
      # Get blobs coordinates
      r_vectors_blobs = np.empty((self.Nblobs, 3))
      offset = 0
      for b in self.bodies:
        r_vectors_blobs[offset:(offset+b.Nblobs)] = b.get_r_vectors(b.location_new, b.orientation_new)
        offset += b.Nblobs

      # Calculate mobility (M) at the blob level
      mobility_blobs = self.mobility_blobs(r_vectors_blobs, self.eta, self.a)

      # Calculate resistance at the blob level (use np.linalg.inv or np.linalg.pinv)
      resistance_blobs = np.linalg.inv(mobility_blobs)

      # Calculate block-diagonal matrix K
      K = np.zeros((3*self.Nblobs, 6*len(self.bodies)))
      offset = 0
      for k, b in enumerate(self.bodies):
        K[3*offset:3*(offset+b.Nblobs), 6*k:6*k+6] = b.calc_K_matrix(location = b.location_new, orientation = b.orientation_new)
        offset += b.Nblobs
     
      # Calculate mobility (N) at the body level. Use np.linalg.inv or np.linalg.pinv
      mobility_bodies_new = np.linalg.pinv(np.dot(K.T, np.dot(resistance_blobs, K)))

      # Add thermal drift to velocity
      velocities += (self.kT / (self.rf_delta * self.blob_radius)) * np.dot(mobility_bodies_new - mobility_bodies, rfd_noise) 
      
      # Update location orientation 
      for k, b in enumerate(self.bodies):
        b.location_new = b.location + velocities[6*k:6*k+3] * dt
        quaternion_dt = Quaternion.from_rotation((velocities[6*k+3:6*k+6]) * dt)
        b.orientation_new = quaternion_dt * b.orientation
        
      # Check positions, if valid return 
      valid_configuration = True
      for b in self.bodies:
        valid_configuration = b.check_function(b.location_new, b.orientation_new)
        if valid_configuration is False:
          break
      if valid_configuration is True:
        for b in self.bodies:
          b.location = b.location_new
          b.orientation = b.orientation_new
        return

      print 'Invalid configuration'
      return

  def solve_mobility_problem(self): 
    ''' 
    Solve the mobility problem using preconditioned GMRES. Compute 
    velocities on the bodies subject to active slip and enternal 
    forces-torques.

    The linear and angular velocities are sorted lile
    velocities = (v_1, w_1, v_2, w_2, ...)
    where v_i and w_i are the linear and angular velocities of body i.
    ''' 
    while True: 
      # Calculate slip on blobs
      if self.calc_slip is not None:
        slip = self.calc_slip(self.bodies, self.Nblobs)
      else:
        slip = np.zeros((self.Nblobs, 3))

      # Get blobs coordinates
      r_vectors_blobs = self.get_blobs_r_vectors(self.bodies, self.Nblobs)

      # Calculate force-torque on bodies
      force_torque = self.force_torque_calculator(self.bodies, r_vectors_blobs)

      # Set right hand side
      System_size = self.Nblobs * 3 + len(self.bodies) * 6
      RHS = np.reshape(np.concatenate([slip, -force_torque]), (System_size))

      # Set linear operators 
      linear_operator_partial = partial(self.linear_operator, bodies=self.bodies, r_vectors=r_vectors_blobs, eta=self.eta, a=self.a)
      A = spla.LinearOperator((System_size, System_size), matvec = linear_operator_partial, dtype='float64')

      # Set preconditioner
      mobility_inv_blobs = []
      # Loop over bodies
      for k, b in enumerate(self.bodies):
        # 1. Compute blobs mobility and invert it
        M = b.calc_mobility_blobs(self.eta, self.a)
        M_inv = np.linalg.inv(M)
        mobility_inv_blobs.append(M_inv)
        # 2. Compute body mobility
        N = b.calc_mobility_body(self.eta, self.a, M_inv = M_inv)
        self.mobility_bodies[k] = N

      # 4. Pack preconditioner
      PC_partial = partial(self.preconditioner, bodies=self.bodies, mobility_bodies=self.mobility_bodies, \
                             mobility_inv_blobs=mobility_inv_blobs, Nblobs=self.Nblobs)
      PC = spla.LinearOperator((System_size, System_size), matvec = PC_partial, dtype='float64')

      # Scale RHS to norm 1
      RHS_norm = np.linalg.norm(RHS)
      if RHS_norm > 0:
        RHS = RHS / RHS_norm

      # Solve preconditioned linear system # callback=make_callback()
      (sol_precond, info_precond) = spla.gmres(A, RHS, x0=self.first_guess, tol=self.tolerance, M=PC, maxiter=1000, restart=60) 
      self.first_guess = sol_precond  

      # Scale solution with RHS norm
      if RHS_norm > 0:
        sol_precond = sol_precond * RHS_norm
      
      # Extract velocities
      return np.reshape(sol_precond[3*self.Nblobs: 3*self.Nblobs + 6*len(self.bodies)], (len(self.bodies) * 6))


  def solve_mobility_problem_dense_algebra(self): 
    ''' 
    Solve the mobility problem using dense algebra methods. Compute 
    velocities on the bodies subject to active slip and enternal 
    forces-torques.
    
    The linear and angular velocities are sorted lile
    velocities = (v_1, w_1, v_2, w_2, ...)
    where v_i and w_i are the linear and angular velocities of body i.
    ''' 
    while True: 
      # Calculate slip on blobs
      if self.calc_slip is not None:
        slip = self.calc_slip(self.bodies, self.Nblobs)
      else:
        slip = np.zeros((self.Nblobs, 3))

      # Get blobs coordinates
      r_vectors_blobs = self.get_blobs_r_vectors(self.bodies, self.Nblobs)

      # Calculate mobility (M) at the blob level
      mobility_blobs = self.mobility_blobs(r_vectors_blobs, self.eta, self.a)

      # Calculate resistance at the blob level (use np.linalg.inv or np.linalg.pinv)
      resistance_blobs = np.linalg.inv(mobility_blobs)

      # Calculate constraint force due to slip l = M^{-1}*slip
      force_slip = np.dot(resistance_blobs, np.reshape(slip, (3*self.Nblobs,1)))

      # Calculate force-torque on bodies
      force_torque = self.force_torque_calculator(self.bodies, r_vectors_blobs)

      # Calculate block-diagonal matrix K
      K = self.calc_K_matrix(self.bodies, self.Nblobs)
     
      # Calculate mobility (N) at the body level. Use np.linalg.inv or np.linalg.pinv
      mobility_bodies = np.linalg.pinv(np.dot(K.T, np.dot(resistance_blobs, K)))

      # Compute velocities
      return (np.dot(mobility_bodies, np.reshape(force_torque, 6*len(self.bodies))), mobility_bodies)


# Callback generator
def make_callback():
    closure_variables = dict(counter=0, residuals=[]) 
    def callback(residuals):
        closure_variables["counter"] += 1
        closure_variables["residuals"].append(residuals)
        print closure_variables["counter"], residuals
    return callback


