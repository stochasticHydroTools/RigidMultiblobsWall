'''
Integrator for several rigid bodies.
'''
import numpy as np
import math as m
import scipy.sparse.linalg as spla
from functools import partial

from quaternion import Quaternion
from stochastic_forcing import stochastic_forcing as stochastic
from mobility import mobility as mob

import scipy

class QuaternionIntegrator(object):
  '''
  Integrator that timesteps using deterministic forwars Euler scheme.
  '''  
  def __init__(self, bodies, Nblobs, scheme, tolerance = None): 
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
    self.kT = 0.0
    self.tolerance = 1e-08
    self.rf_delta = 1e-05

    # Optional variables
    self.calc_slip = None
    self.calc_force_torque = None
    self.mobility_inv_blobs = None
    self.first_guess = None
    self.preconditioner = None
    self.mobility_vector_prod = None    
    if tolerance is not None:
      self.tolerance = tolerance
      self.rf_delta = 0.1 * np.power(self.tolerance, 1.0/3.0)
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
      sol_precond = self.solve_mobility_problem()
      
      # Extract velocities
      velocities = np.reshape(sol_precond[3*self.Nblobs: 3*self.Nblobs + 6*len(self.bodies)], (len(self.bodies) * 6))

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
      sol_precond = self.solve_mobility_problem()

      # Extract velocities
      velocities = np.reshape(sol_precond[3*self.Nblobs: 3*self.Nblobs + 6*len(self.bodies)], (len(self.bodies) * 6))

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


  def stochastic_first_order_RFD(self, dt): 
    ''' 
    Take a time step of length dt using a stochastic
    first order Randon Finite Difference (RFD) schame.

    The linear and angular velocities are sorted lile
    velocities = (v_1, w_1, v_2, w_2, ...)
    where v_i and w_i are the linear and angular velocities of body i.
    ''' 
    while True: 
      print 'delta', self.rf_delta
      print 'tol', self.tolerance
      # Save initial configuration
      for k, b in enumerate(self.bodies):
        b.location_old = b.location
        b.orientation_old = b.orientation

      # Generate random vector
      rfd_noise = np.random.normal(0.0, 1.0, len(self.bodies) * 6)     

      # Set function M*f at the blob level
      r_vectors_blobs = self.get_blobs_r_vectors(self.bodies, self.Nblobs)
      def mult_mobility_blobs(force = None, r_vectors = None, eta = None, a = None):
        return self.mobility_vector_prod(r_vectors, force, eta, a)
      mobility_mult_partial = partial(mult_mobility_blobs, r_vectors = r_vectors_blobs, eta = self.eta, a = self.a) 

      # Add noise contribution sqrt(2kT/dt)*N^{1/2}*W
      velocities_noise, it_lanczos = stochastic.stochastic_forcing_lanczos(factor = np.sqrt(2*self.kT / dt),
                                                                           tolerance = self.tolerance, 
                                                                           dim = self.Nblobs * 3, 
                                                                           mobility_mult = mobility_mult_partial)

      # Solve mobility problem
      sol_precond = self.solve_mobility_problem(noise = velocities_noise, x0 = self.first_guess, save_first_guess = True)

      # Extract velocities
      velocities = np.reshape(sol_precond[3*self.Nblobs: 3*self.Nblobs + 6*len(self.bodies)], (len(self.bodies) * 6))

      # Update configuration for rfd 
      for k, b in enumerate(self.bodies):
        b.location = b.location_old - rfd_noise[k*6 : k*6+3] * self.rf_delta * 0.5
        quaternion_dt = Quaternion.from_rotation(rfd_noise[(k*6+3):(k*6+6)] * self.rf_delta * (-0.5))
        b.orientation = quaternion_dt * b.orientation_old

      # Add thermal drift contribution with N at x = x - random_displacement
      System_size = self.Nblobs * 3 + len(self.bodies) * 6
      sol_precond = self.solve_mobility_problem(RHS = np.reshape(np.concatenate([np.zeros(3*self.Nblobs), rfd_noise]), (System_size)))

      # Update configuration for rfd 
      for k, b in enumerate(self.bodies):
        b.location = b.location_old + rfd_noise[k*6 : k*6+3] * self.rf_delta * 0.5
        quaternion_dt = Quaternion.from_rotation(rfd_noise[(k*6+3):(k*6+6)] * self.rf_delta * 0.5)
        b.orientation = quaternion_dt * b.orientation_old

      # Modify RHS for drift solve
      # Set linear operators 
      r_vectors_blobs = self.get_blobs_r_vectors(self.bodies, self.Nblobs)
      linear_operator_partial = partial(self.linear_operator, bodies=self.bodies, r_vectors=r_vectors_blobs, eta=self.eta, a=self.a)
      A = spla.LinearOperator((System_size, System_size), matvec = linear_operator_partial, dtype='float64')
      RHS = np.reshape(np.concatenate([np.zeros(3*self.Nblobs), rfd_noise]), (System_size)) - A * sol_precond

      # Add thermal drift contribution with N at x = x + random_displacement
      sol_precond = self.solve_mobility_problem(RHS = RHS)

      # Extract velocities
      velocities_drift = np.reshape(sol_precond[3*self.Nblobs: 3*self.Nblobs + 6*len(self.bodies)], (len(self.bodies) * 6))

      # Add all velocity contributions
      velocities += (self.kT / self.rf_delta) * velocities_drift
      
      # Update location orientation 
      for k, b in enumerate(self.bodies):
        b.location_new = b.location_old + velocities[6*k:6*k+3] * dt
        quaternion_dt = Quaternion.from_rotation((velocities[6*k+3:6*k+6]) * dt)
        b.orientation_new = quaternion_dt * b.orientation_old
        
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
      else:
        for b in self.bodies:
          b.location = b.location_old
          b.orientation = b.orientation_old

      print 'Invalid configuration'
    return


  def stochastic_first_order_RFD_dense_algebra(self, dt): 
    ''' 
    Take a time step of length dt using a stochastic
    first order Randon Finite Difference (RFD) scheme.
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
      velocities += stochastic.stochastic_forcing_eig(mobility_bodies, factor = np.sqrt(2*self.kT / dt))
      # velocities += stochastic.stochastic_forcing_cholesky(mobility_bodies, factor = np.sqrt(2*self.kT / dt))

      # Update configuration for rfd
      for k, b in enumerate(self.bodies):
        b.location_new = b.location + rfd_noise[k*6 : k*6+3] * self.rf_delta
        quaternion_dt = Quaternion.from_rotation(rfd_noise[(k*6+3):(k*6+6)] * self.rf_delta)
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
      mobility_bodies_new = np.linalg.pinv(np.dot(K.T, np.dot(resistance_blobs, K)), rcond=1e-14)

      # Add thermal drift to velocity
      velocities += (self.kT / self.rf_delta) * np.dot(mobility_bodies_new - mobility_bodies, rfd_noise) 
      
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

  def solve_mobility_problem(self, RHS = None, noise = None, x0 = None, save_first_guess = False): 
    ''' 
    Solve the mobility problem using preconditioned GMRES. Compute 
    velocities on the bodies subject to active slip and enternal 
    forces-torques.

    The linear and angular velocities are sorted lile
    velocities = (v_1, w_1, v_2, w_2, ...)
    where v_i and w_i are the linear and angular velocities of body i.
    ''' 
    while True: 
      System_size = self.Nblobs * 3 + len(self.bodies) * 6

      # Get blobs coordinates
      r_vectors_blobs = self.get_blobs_r_vectors(self.bodies, self.Nblobs)

      # If RHS = None set RHS = [slip, -force_torque]
      if RHS is None:
        # Calculate slip on blobs
        if self.calc_slip is not None:
          slip = self.calc_slip(self.bodies, self.Nblobs)
        else:
          slip = np.zeros((self.Nblobs, 3))
        # Calculate force-torque on bodies
        force_torque = self.force_torque_calculator(self.bodies, r_vectors_blobs)
        # Set right hand side
        RHS = np.reshape(np.concatenate([slip, -force_torque]), (System_size))

      # Add noise to the slip
      if noise is not None:
        RHS[0:r_vectors_blobs.size] -= noise

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
      (sol_precond, info_precond) = spla.gmres(A, RHS, x0=x0, tol=self.tolerance, M=PC, maxiter=1000, restart=60) 

      # Scale solution with RHS norm
      if RHS_norm > 0:
        sol_precond = sol_precond * RHS_norm
      if save_first_guess:
        self.first_guess = sol_precond  
      
      # Return solution
      return sol_precond


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
      mobility_bodies = np.linalg.pinv(np.dot(K.T, np.dot(resistance_blobs, K)), rcond=1e-14)

      # Compute velocities, return velocities and bodies' mobility
      return (np.dot(mobility_bodies, np.reshape(force_torque, 6*len(self.bodies))), mobility_bodies)


# Callback generator
def make_callback():
    closure_variables = dict(counter=0, residuals=[]) 
    def callback(residuals):
        closure_variables["counter"] += 1
        closure_variables["residuals"].append(residuals)
        print closure_variables["counter"], residuals
    return callback


