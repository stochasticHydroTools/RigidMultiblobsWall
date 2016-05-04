'''
Integrator for several rigid bodies.
'''
import numpy as np
import math as m
import scipy.sparse.linalg as spla
from functools import partial

from quaternion import Quaternion

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

    # Optional variables
    self.calc_slip = None
    self.calc_force_torque = None
    self.mobility_blobs_cholesky = None
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
        b.location += velocities[6*k:6*k+3] * dt
        quaternion_dt = Quaternion.from_rotation((velocities[6*k+3:6*k+6]) * dt)
        b.orientation = quaternion_dt * b.orientation
        
      # Check positions, if valid return 
      valid_configuration = True
      for b in self.bodies:
        valid_configuration = b.check_function()
        if valid_configuration is False:
          break
      if valid_configuration is True:
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
      velocities = self.solve_mobility_problem_dense_algebra()

      # Update location orientation 
      for k, b in enumerate(self.bodies):
        b.location += velocities[6*k:6*k+3] * dt
        quaternion_dt = Quaternion.from_rotation((velocities[6*k+3:6*k+6]) * dt)
        b.orientation = quaternion_dt * b.orientation
        
      # Check positions, if valid return 
      valid_configuration = True
      for b in self.bodies:
        valid_configuration = b.check_function()
        if valid_configuration is False:
          break
      if valid_configuration is True:
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
          b.location += (1.5 * velocities[6*k:6*k+3] - 0.5 * self.velocities_previous_step[6*k:6*k+3]) * dt
          quaternion_dt = Quaternion.from_rotation((1.5 * velocities[6*k+3:6*k+6] \
                                                      - 0.5 * self.velocities_previous_step[6*k+3:6*k+6]) * dt)
          b.orientation = quaternion_dt * b.orientation
      else:
        # Use forward Euler
        for k, b in enumerate(self.bodies):
          b.location += velocities[6*k:6*k+3] * dt
          quaternion_dt = Quaternion.from_rotation((velocities[6*k+3:6*k+6]) * dt)
          b.orientation = quaternion_dt * b.orientation              

      # Check positions, if valid return 
      valid_configuration = True
      for b in self.bodies:
        valid_configuration = b.check_function()
        if valid_configuration is False:
          break
      if valid_configuration is True:
        # Save velocities for next step
        self.first_step = False
        self.velocities_previous_step = velocities
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
      self.mobility_blobs_cholesky = []
      # 1. Loop over bodies
      for k, b in enumerate(self.bodies):
        # 2. Compute body mobility
        N = b.calc_mobility_body(self.eta, self.a)
        self.mobility_bodies[k] = N
        # 3. Compute cholesky factorization
        L = np.empty((3*b.Nblobs, 3*b.Nblobs))
        L = b.calc_mobility_blobs_cholesky(self.eta, self.a)
        self.mobility_blobs_cholesky.append(L)
      # 4. Pack preconditioner
      PC_partial = partial(self.preconditioner, bodies=self.bodies, mobility_bodies=self.mobility_bodies, \
                             mobility_blobs_cholesky=self.mobility_blobs_cholesky, Nblobs=self.Nblobs)
      PC = spla.LinearOperator((System_size, System_size), matvec = PC_partial, dtype='float64')
      # PC = None

      # Solve preconditioned linear system # callback=make_callback()
      (sol_precond, info_precond) = spla.gmres(A, RHS, x0=self.first_guess, tol=1e-8, M=PC, maxiter=1000, restart=60) 
      self.first_guess = sol_precond  

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
      resistance_bodies = np.dot(K.T, np.dot(resistance_blobs, K))
      mobility_bodies = np.linalg.inv(np.dot(K.T, np.dot(resistance_blobs, K)))

      # Compute velocities
      return np.dot(mobility_bodies, np.reshape(force_torque, 6*len(self.bodies)))


# Callback generator
def make_callback():
    closure_variables = dict(counter=0, residuals=[]) 
    def callback(residuals):
        closure_variables["counter"] += 1
        closure_variables["residuals"].append(residuals)
        print closure_variables["counter"], residuals
    return callback
