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
    self.K_matrix_T_vector_prod = None
    self.linear_operator = None
    self.eta = None
    self.a = None
    self.velocities = None
    self.velocities_previous_step = None
    self.first_step = True
    self.tolerance = 1e-10
    self.rf_delta = 1e-06
    self.pc_delta = 0.25
    self.kT = 0.0
    self.rand_run = 'False'

    # Optional variables
    self.calc_slip = None
    self.calc_force_torque = 'True'
    self.mobility_inv_blobs = None
    self.mobility_vector_prod = None
    self.first_guess = None
    self.preconditioner = None
    
    # test run variables
    self.mobility_test = False
    self.vel = np.zeros((len(bodies)*6))
    #self.mean_disp = np.zeros((self.Nblobs, 3))
    
    # variable that will use dense LA to compute the Brownian increments in the itterative PC scheme (if true).
    self.rand_dense = 'True'
    
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

  def stochastic_PC(self, dt): 
    ''' 
    Take a time step of length dt using the deterministic forward Euler scheme. 
    The function uses dense algebra methods to solve the equations.
    
    The linear and angular velocities are sorted lile
    velocities = (v_1, w_1, v_2, w_2, ...)
    where v_i and w_i are the linear and angular velocities of body i.
    ''' 
    while True: 
      
      # Save initial configuration
      for k, b in enumerate(self.bodies):
        b.location_old = b.location
	b.orientation_old = b.orientation
	
      
      if (self.rand_run == 'True'):
        # generate all of the necisarry random increments for the predictor step
        W1 = np.random.normal(0.0, 1.0, self.Nblobs*3)
        W2 = np.random.normal(0.0, 1.0, self.Nblobs*3)
        Wcor = W1 + np.random.normal(0.0, 1.0, self.Nblobs*3)
      
        # Set function M*f at the blob level
        r_vectors_blobs = self.get_blobs_r_vectors(self.bodies, self.Nblobs)
        def mult_mobility_blobs(force = None, r_vectors = None, eta = None, a = None):
          return self.mobility_vector_prod(r_vectors, force, eta, a)
        mobility_mult_partial = partial(mult_mobility_blobs, r_vectors = r_vectors_blobs, eta = self.eta, a = self.a) 

        #compute M*W2 to be used by the corrector step
        MnxW2 = mobility_mult_partial(W2)

	if (self.rand_dense == 'True'):
          # Calculate mobility (M) at the blob level
          mobility_blobs = self.mobility_blobs(r_vectors_blobs, self.eta, self.a)
          velocities_noise_W1 = stochastic.stochastic_forcing_eig(mobility_blobs, factor = 1.0, z = W1)
          velocities_noise_Wcor = stochastic.stochastic_forcing_eig(mobility_blobs, factor = 1.0, z = Wcor)
        else:
	  #compute blob noise contribution sqrt(kT/dt)*M^{1/2}*W for both steps
	  velocities_noise_W1, it_lanczos_pred = stochastic.stochastic_forcing_lanczos(factor = 1.0,
									      tolerance = self.tolerance, 
									      dim = self.Nblobs * 3, 
									      mobility_mult = mobility_mult_partial,
									      z = W1)
	
	  velocities_noise_Wcor, it_lanczos_cor = stochastic.stochastic_forcing_lanczos(factor = 1.0,
									      tolerance = self.tolerance, 
									      dim = self.Nblobs * 3, 
									      mobility_mult = mobility_mult_partial,
									      z = Wcor)
      
      
      
        #compute random slip for pred. and cor. steps and random force for cor.
        rand_slip_pred = np.sqrt(4*self.kT / dt)*(velocities_noise_W1 + np.sqrt( self.pc_delta / (self.eta*self.a*6.0*np.pi) )*W2)
        rand_slip_cor = np.sqrt(self.kT / dt)*(velocities_noise_Wcor - np.sqrt(self.eta*self.a*6.0*np.pi/(self.pc_delta) )*MnxW2) 
        rand_force_cor = self.K_matrix_T_vector_prod(self.bodies,np.sqrt(self.kT*self.eta*self.a*6.0*np.pi/(dt*self.pc_delta))*W2 ,self.Nblobs)
      
	RHS_pred = np.concatenate([-1.0*rand_slip_pred, np.zeros(len(self.bodies) * 6)])
        RHS_cor = np.concatenate([-1.0*rand_slip_cor, -1.0*np.reshape(rand_force_cor,len(self.bodies) * 6)])
      
      # Solve mobility problem for pred. step
      if (self.rand_run == 'True'):
        velocities_mid = self.solve_mobility_problem(rand_RHS = RHS_pred)
      else:
        velocities_mid = self.solve_mobility_problem()

      # Update location orientation to mid point
      for k, b in enumerate(self.bodies):
        b.location = b.location + velocities_mid[6*k:6*k+3] * dt * 0.5
        quaternion_dt = Quaternion.from_rotation((velocities_mid[6*k+3:6*k+6]) * dt * 0.5)
        b.orientation = quaternion_dt * b.orientation
        
      if (self.rand_run == 'True'):
        velocities_new = self.solve_mobility_problem(rand_RHS = RHS_cor)
      else:
	velocities_new = self.solve_mobility_problem()
      
        
      # Update location orientation to end point
      for k, b in enumerate(self.bodies):
        b.location_new = b.location_old + velocities_new[6*k:6*k+3] * dt
        quaternion_dt = Quaternion.from_rotation((velocities_new[6*k+3:6*k+6]) * dt)
        b.orientation_new = quaternion_dt * b.orientation_old  
      
        
      # Check positions, if valid return 
      valid_configuration = True
      for b in self.bodies:
        valid_configuration = b.check_function(b.location_new, b.orientation_new)
        if valid_configuration is False:
	  for b1 in self.bodies:
	    b1.location = b1.location_old
	    b1.orientation = b1.orientation_old
          break
      if valid_configuration is True:
	if (self.mobility_test == 'True'):
          self.vel = velocities_new
          for b in self.bodies:
            b.location = b.location_old
            b.orientation = b.orientation_old
	  return
        else:
          for b in self.bodies:
            b.location = b.location_new
            b.orientation = b.orientation_new
          return

      print 'Invalid configuration'
    return

  def stochastic_PC_dense_algebra(self, dt): 
    ''' 
    Take a time step of length dt using the deterministic forward Euler scheme. 
    The function uses dense algebra methods to solve the equations.
    
    The linear and angular velocities are sorted lile
    velocities = (v_1, w_1, v_2, w_2, ...)
    where v_i and w_i are the linear and angular velocities of body i.
    ''' 
    while True: 
      
      # Save initial configuration
      for k, b in enumerate(self.bodies):
        b.location_old = b.location
	b.orientation_old = b.orientation
	
      
      if (self.rand_run == 'True'):
	# Solve mobility problem predictor step
        velocities_mid, mobility_bodies, mobility_blobs, resistance_blobs, K, r_vectors_blobs = self.solve_mobility_problem_DLA()
	
        # generate all of the necisarry random increments for the predictor step
        W1 = np.random.normal(0.0, 1.0, self.Nblobs*3)
        W2 = np.random.normal(0.0, 1.0, self.Nblobs*3)
        Wcor = W1 + np.random.normal(0.0, 1.0, self.Nblobs*3)
      
        # Set function M*f at the blob level
        def mult_mobility_blobs(force = None, r_vectors = None, eta = None, a = None):
          return self.mobility_vector_prod(r_vectors, force, eta, a)
        mobility_mult_partial = partial(mult_mobility_blobs, r_vectors = r_vectors_blobs, eta = self.eta, a = self.a) 

        # Compute M*W2 and K^T*W2 to be used by the corrector step
        MnxW2 = np.sqrt(self.eta*self.a*6.0*np.pi/(self.pc_delta) )*mobility_mult_partial(W2)
        KTW2 = np.sqrt(self.eta*self.a*6.0*np.pi/(self.pc_delta) )*np.dot(K.T,W2)
        
        # Calculate relevant bit of stochastic increments at time level n
        Mhalf_W1 = stochastic.stochastic_forcing_eig(mobility_blobs, factor = 1.0, z = W1)
        Mhalf_Wcor = stochastic.stochastic_forcing_eig(mobility_blobs, factor = 1.0, z = Wcor)
        
        # Compute c1*N*K^T*M^(-1)*(c1*W2 + M^(1/2)*W1) for pred. step
        c1 = np.sqrt( self.pc_delta / (self.eta*self.a*6.0*np.pi) )
        RHS_pred = np.sqrt(4*self.kT / dt)*np.dot(mobility_bodies,
						  np.dot(K.T,
						  np.dot(resistance_blobs,
						  (c1*W2 + Mhalf_W1))))
      
        # Compute pred. step velocities
        velocities_mid += RHS_pred

	for k, b in enumerate(self.bodies):
	  b.location = b.location + velocities_mid[6*k:6*k+3] * dt * 0.5
	  quaternion_dt = Quaternion.from_rotation((velocities_mid[6*k+3:6*k+6]) * dt * 0.5)
	  b.orientation = quaternion_dt * b.orientation
        
        # Solve mobility problem predictor step 
        velocities_new, mobility_bodies, mobility_blobs, resistance_blobs, K, r_vectors_blobs = self.solve_mobility_problem_DLA()
        
        # Compute RHS of cor step so that N*RHS_cor is the correct increment
        RHS_cor = np.sqrt(self.kT / dt)*(KTW2 + np.dot(K.T,np.dot(resistance_blobs,Mhalf_Wcor-MnxW2)))
        
        # Compute cor. step velocities
        velocities_new += np.dot(mobility_bodies,RHS_cor)
        
        # Update location orientation to end point
	for k, b in enumerate(self.bodies):
	  b.location_new = b.location_old + velocities_new[6*k:6*k+3] * dt
	  quaternion_dt = Quaternion.from_rotation((velocities_new[6*k+3:6*k+6]) * dt)
	  b.orientation_new = quaternion_dt * b.orientation_old
	  
      else:
	# Solve mobility problem predictor step
	velocities_mid = self.solve_mobility_problem_DLA()[1]
	
	# Update location orientation to mid point
	for k, b in enumerate(self.bodies):
	  b.location = b.location + velocities_mid[6*k:6*k+3] * dt * 0.5
	  quaternion_dt = Quaternion.from_rotation((velocities_mid[6*k+3:6*k+6]) * dt * 0.5)
	  b.orientation = quaternion_dt * b.orientation
	
	# Solve mobility problem predictor step
	velocities_new = self.solve_mobility_problem_DLA()[1]
	
	# Update location orientation to end point
	for k, b in enumerate(self.bodies):
	  b.location_new = b.location_old + velocities_new[6*k:6*k+3] * dt
	  quaternion_dt = Quaternion.from_rotation((velocities_new[6*k+3:6*k+6]) * dt)
	  b.orientation_new = quaternion_dt * b.orientation_old
	
        
      # Check positions, if valid return 
      valid_configuration = True
      for b in self.bodies:
        valid_configuration = b.check_function(b.location_new, b.orientation_new)
        if valid_configuration is False:
	  for b1 in self.bodies:
	    b1.location = b1.location_old
	    b1.orientation = b1.orientation_old
          break
      if valid_configuration is True:
	if (self.mobility_test == 'True'):
          self.vel = velocities_new
          for b in self.bodies:
            b.location = b.location_old
            b.orientation = b.orientation_old
	  return
        else:
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
      
      # Save initial configuration
      for k, b in enumerate(self.bodies):
        b.location_old = b.location
	b.orientation_old = b.orientation
      
      ## TODO: make option in input file???
      #r_vectors_blobs_old = np.empty((self.Nblobs, 3))
      #offset = 0
      #for b in self.bodies:
        #r_vectors_blobs_old[offset:(offset+b.Nblobs)] = b.get_r_vectors(b.location, b.orientation)
        #offset += b.Nblobs
      ## TODO
      
      
      
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
      
      
      ## TODO: make option in input file
      #r_vectors_blobs_new = np.empty((self.Nblobs, 3))
      #offset = 0
      #for b in self.bodies:
        #r_vectors_blobs_new[offset:(offset+b.Nblobs)] = b.get_r_vectors(b.location_new, b.orientation_new)
        #offset += b.Nblobs
      ## TODO
      
      # Check positions, if valid return 
      valid_configuration = True
      for b in self.bodies:
        valid_configuration = b.check_function(b.location_new, b.orientation_new)
        if valid_configuration is False:
          break
      if valid_configuration is True:
	if (self.mobility_test == 'True'):
          self.vel = velocities
          #self.mean_disp = self.mean_disp + (r_vectors_blobs_new - r_vectors_blobs_old)
          for b in self.bodies:
            b.location = b.location_old
            b.orientation = b.orientation_old
	  return
        else:
          for b in self.bodies:
            b.location = b.location_new
            b.orientation = b.orientation_new
          return

      print 'Invalid configuration'
    return
  
  def stochastic_second_order_RFD_dense_algebra(self, dt):
    '''
    Take a time step of length dt using a stochastic
    first order Randon Finite Difference (RFD) schame.
    The function uses dense algebra methods to solve the equations.

    The linear and angular velocities are sorted lile
    velocities = (v_1, w_1, v_2, w_2, ...)
    where v_i and w_i are the linear and angular velocities of body i.
    '''
    while True:
     
      # Save initial configuration
      for k, b in enumerate(self.bodies):
        b.location_old = b.location
	b.orientation_old = b.orientation
     
      # Solve mobility problem
      velocities, mobility_bodies = self.solve_mobility_problem_dense_algebra()

      # Generate random vector
      rfd_noise = np.random.normal(0.0, 1.0, len(self.bodies) * 6)    
      W1 = np.random.normal(0.0, 1.0, len(self.bodies) * 6)
      Wcor = W1 + np.random.normal(0.0, 1.0, len(self.bodies) * 6)

 
      # Add noise contribution sqrt(2kT/dt)*N^{1/2}*W
      velocities += stochastic.stochastic_forcing_eig_symm(mobility_bodies, factor = np.sqrt(4*self.kT / dt), z = W1)
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
      mobility_bodies_rfd = np.linalg.pinv(np.dot(K.T, np.dot(resistance_blobs, K)), rcond=1e-14)
     
      # Calc RFD term for cor. step
      Nhalf_rfd = stochastic.stochastic_forcing_eig_symm(mobility_bodies_rfd, factor = 1.0, z = rfd_noise)
      Nhalf_Nhalf_rfd = stochastic.stochastic_forcing_eig_symm(mobility_bodies, factor = 1.0, z = Nhalf_rfd)
      mobility_bodies_rfd = (self.kT / self.rf_delta) * ( Nhalf_Nhalf_rfd - np.dot(mobility_bodies,rfd_noise) )
     
      # Update location orientation
      for k, b in enumerate(self.bodies):
        b.location = b.location + velocities[6*k:6*k+3] * dt * 0.5
        quaternion_dt = Quaternion.from_rotation((velocities[6*k+3:6*k+6]) * dt * 0.5)
        b.orientation = quaternion_dt * b.orientation
       
       
      # Corrector step
     
      # Solve mobility problem
      velocities_new, mobility_bodies_new = self.solve_mobility_problem_dense_algebra()
     
      # Add noise contribution sqrt(2kT/dt)*N^{1/2}*W
      velocities_new += stochastic.stochastic_forcing_eig_symm(mobility_bodies_new, factor = np.sqrt(self.kT / dt), z = Wcor) + mobility_bodies_rfd
     
     
      # Update location orientation
      for k, b in enumerate(self.bodies):
        b.location_new = b.location + velocities_new[6*k:6*k+3] * dt
        quaternion_dt = Quaternion.from_rotation((velocities_new[6*k+3:6*k+6]) * dt)
        b.orientation_new = quaternion_dt * b.orientation
     
      # Check positions, if valid return
      valid_configuration = True
      for b in self.bodies:
        valid_configuration = b.check_function(b.location_new, b.orientation_new)
        if valid_configuration is False:
          break
      if valid_configuration is True:
	if (self.mobility_test == 'True'):
          self.vel = velocities_new
          for b in self.bodies:
            b.location = b.location_old
            b.orientation = b.orientation_old
	  return
        else:
          for b in self.bodies:
            b.location = b.location_new
            b.orientation = b.orientation_new
          return

      print 'Invalid configuration'
    return
  
  def Fixman(self, dt):
    '''
    Take a time step of length dt using the Fixman schame.
    The function uses dense algebra methods to solve the equations.

    The linear and angular velocities are sorted lile
    velocities = (v_1, w_1, v_2, w_2, ...)
    where v_i and w_i are the linear and angular velocities of body i.
    '''
    while True:
     
      # Save initial configuration
      for k, b in enumerate(self.bodies):
        b.location_old = b.location
        b.orientation_old = b.orientation
     
      # Solve mobility problem
      velocities_mid, mobility_bodies = self.solve_mobility_problem_dense_algebra()

      # Generate random vector
      W1 = np.random.normal(0.0, 1.0, len(self.bodies) * 6)
      W_cor = W1 + np.random.normal(0.0, 1.0, len(self.bodies) * 6);

      # Compute noise contribution for pred. step sqrt(2kT/dt)*N^{1/2}*W1
      Nhalf_W1 = stochastic.stochastic_forcing_eig(mobility_bodies, factor = np.sqrt(4*self.kT / dt),z = W1)
     
      # Compute noise contribution for cor. step sqrt(2kT/dt)*N^{1/2}*W1
      Nhalf_Wcor = stochastic.stochastic_forcing_eig(mobility_bodies, factor = np.sqrt(self.kT / dt),z = W_cor)
      Ninvhalf_cor = np.dot(np.linalg.pinv(mobility_bodies, rcond=1e-14),Nhalf_Wcor)
     
      velocities_mid += Nhalf_W1
      # velocities += stochastic.stochastic_forcing_cholesky(mobility_bodies, factor = np.sqrt(2*self.kT / dt))

      # Update location orientation to mid point
      for k, b in enumerate(self.bodies):
	b.location = b.location + velocities_mid[6*k:6*k+3] * dt * 0.5
	quaternion_dt = Quaternion.from_rotation((velocities_mid[6*k+3:6*k+6]) * dt * 0.5)
	b.orientation = quaternion_dt * b.orientation
   
      # Solve mobility problem predictor step
      velocities_new, mobility_bodies = self.solve_mobility_problem_dense_algebra()
      velocities_new += np.dot(mobility_bodies,Ninvhalf_cor)
   
      # Update location orientation to end point
      for k, b in enumerate(self.bodies):
	b.location_new = b.location_old + velocities_new[6*k:6*k+3] * dt
	quaternion_dt = Quaternion.from_rotation((velocities_new[6*k+3:6*k+6]) * dt)
	b.orientation_new = quaternion_dt * b.orientation_old
       
      # Check positions, if valid return 
      valid_configuration = True
      for b in self.bodies:
        valid_configuration = b.check_function(b.location_new, b.orientation_new)
        if valid_configuration is False:
          break
      if valid_configuration is True:
	if (self.mobility_test == 'True'):
          self.vel = velocities_new
          #self.mean_disp = self.mean_disp + (r_vectors_blobs_new - r_vectors_blobs_old)
          for b in self.bodies:
            b.location = b.location_old
            b.orientation = b.orientation_old
	  return
        else:
          for b in self.bodies:
            b.location = b.location_new
            b.orientation = b.orientation_new
          return

      print 'Invalid configuration'
    return

  def solve_mobility_problem(self,rand_RHS = None): 
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
      if (self.calc_force_torque == 'True'):
        force_torque = self.force_torque_calculator(self.bodies, r_vectors_blobs)
      else:
	force_torque = np.zeros((len(self.bodies) * 2,3))

      # Set right hand side
      System_size = self.Nblobs * 3 + len(self.bodies) * 6
	
      # Set RHS
      RHS = np.reshape(np.concatenate([slip, -force_torque]), (System_size))
      
      # Add random RHS if present
      if rand_RHS is not None:
	RHS += rand_RHS

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
      if (self.calc_force_torque == 'True'):
        force_torque = self.force_torque_calculator(self.bodies, r_vectors_blobs)
      else:
	force_torque = np.zeros((len(self.bodies) * 2,3))

      # Calculate block-diagonal matrix K
      K = self.calc_K_matrix(self.bodies, self.Nblobs)
     
      # Calculate mobility (N) at the body level. Use np.linalg.inv or np.linalg.pinv
      mobility_bodies = np.linalg.pinv(np.dot(K.T, np.dot(resistance_blobs, K)), rcond=1e-14)

      # Compute velocities
      return (np.dot(mobility_bodies, np.reshape(force_torque, 6*len(self.bodies))), mobility_bodies)

  def solve_mobility_problem_DLA(self): 
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

      # Calculate block-diagonal matrix K
      K = self.calc_K_matrix(self.bodies, self.Nblobs)
     
      # Calculate constraint force due to slip l = M^{-1}*slip
      force_slip = np.dot(K.T,np.dot(resistance_blobs, np.reshape(slip, (3*self.Nblobs,1))))
      
      # Calculate force-torque on bodies
      if (self.calc_force_torque == 'True'):
        force_torque = self.force_torque_calculator(self.bodies, r_vectors_blobs)
      else:
	force_torque = np.zeros((len(self.bodies) * 2,3))
      
      # Calculate RHS
      FT = np.reshape(force_torque, 6*len(self.bodies))
      FTS = FT + np.reshape(force_slip, 6*len(self.bodies))

      # Calculate mobility (N) at the body level. Use np.linalg.inv or np.linalg.pinv
      mobility_bodies = np.linalg.pinv(np.dot(K.T, np.dot(resistance_blobs, K)), rcond=1e-14)

      # Compute velocities
      return (np.dot(mobility_bodies, FTS), mobility_bodies, mobility_blobs, resistance_blobs, K, r_vectors_blobs)
    
# Callback generator
def make_callback():
    closure_variables = dict(counter=0, residuals=[]) 
    def callback(residuals):
        closure_variables["counter"] += 1
        closure_variables["residuals"].append(residuals)
        print closure_variables["counter"], residuals
    return callback


