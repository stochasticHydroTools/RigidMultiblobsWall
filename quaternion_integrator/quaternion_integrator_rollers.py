'''
Integrator for several rigid bodies.
'''
import numpy as np
import math as m
import scipy.sparse.linalg as spla
from functools import partial
import time

from quaternion import Quaternion
from stochastic_forcing import stochastic_forcing as stochastic
from mobility import mobility as mob

import scipy

class QuaternionIntegratorRollers(object):
  '''
  Integrator that timesteps using deterministic forwars Euler scheme.
  '''  
  def __init__(self, bodies, Nblobs, scheme, tolerance = None, domain = 'single_wall'): 
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
    self.deterministic_torque_previous_step = None
    self.first_step = True
    self.kT = 0.0
    self.tolerance = 1e-08
    self.rf_delta = 1e-06
    self.invalid_configuration_count = 0
    self.wall_overlaps = 0
    self.omega_one_roller = None
    self.free_kinematics = 'True'
    self.det_iterations_count = 0
    self.stoch_iterations_count = 0
    self.domain = domain
    if domain == 'single_wall':
      self.mobility_trans_times_force = mob.single_wall_mobility_trans_times_force_pycuda
      self.mobility_trans_times_torque = mob.single_wall_mobility_trans_times_torque_pycuda
      self.mobility_rot_times_force = mob.single_wall_mobility_rot_times_force_pycuda
      self.mobility_rot_times_torque = mob.single_wall_mobility_rot_times_torque_pycuda
      self.mobilit_trans_times_force_torque = mob.single_wall_mobility_trans_times_force_torque_pycuda
    elif domain == 'no_wall':
      self.mobility_trans_times_force = mob.no_wall_mobility_trans_times_force_pycuda
      self.mobility_trans_times_torque = mob.no_wall_mobility_trans_times_torque_pycuda
      self.mobility_rot_times_force = mob.no_wall_mobility_rot_times_force_pycuda
      self.mobility_rot_times_torque = mob.no_wall_mobility_rot_times_torque_pycuda
      self.mobilit_trans_times_force_torque = mob.no_wall_mobility_trans_times_force_torque_pycuda

    # Optional variables
    self.build_stochastic_block_diagonal_preconditioner = None
    self.periodic_length = None
    self.calc_slip = None
    self.calc_force_torque = None
    self.mobility_inv_blobs = None
    self.first_guess = None
    self.preconditioner = None
    self.mobility_vector_prod = None    
    self.hydro_interactions = None
    if tolerance is not None:
      self.tolerance = tolerance
    return 

  def advance_time_step(self, dt, *args, **kwargs):
    '''
    Advance time step with integrator self.scheme
    '''
    return getattr(self, self.scheme.replace('_rollers', ''))(dt)
    

  def deterministic_forward_euler(self, dt): 
    ''' 
    Take a time step of length dt using the deterministic forward Euler scheme. 
    ''' 
    while True: 
      # Compute deterministic velocity
      if self.hydro_interactions==1: 
      	det_velocity, det_torque = self.compute_deterministic_velocity_and_torque()
      else:
      	det_velocity, det_torque = self.compute_deterministic_velocity_and_torque_uncorrelated()

      # Update position   
      for k, b in enumerate(self.bodies):
        b.location_new = b.location + dt * det_velocity[3*k : 3*(k+1)]

      # Check if configuration is valid and update postions if so      
      valid_configuration = True
      if self.domain == 'single_wall':
        for b in self.bodies:
          if b.location_new[2] < 0.0:      
            valid_configuration = False
            break
      if valid_configuration is True:
        for b in self.bodies:
          b.location = b.location_new            
          if self.domain == 'single_wall':
            if b.location[2] < self.a:
              self.wall_overlaps += 1            
        return

      self.invalid_configuration_count += 1
      print 'Invalid configuration'
    return

  
  def stochastic_first_order(self, dt):
    '''
    Take a time step of length dt using a first order
    stochastic integrator.
    '''
    while True:
      # Save initial configuration
      for k, b in enumerate(self.bodies):
        b.location_old = b.location

      if self.hydro_interactions==1: 
         # Compute deterministic velocity
      	det_velocity, det_torque = self.compute_deterministic_velocity_and_torque()
      	# Compute stochastic velocity
      	stoch_velocity = self.compute_stochastic_linear_velocity(dt)
      else:
      	det_velocity, det_torque = self.compute_deterministic_velocity_and_torque_uncorrelated()
      	stoch_velocity = self.compute_stochastic_linear_velocity_uncorrelated (dt)

      # Add velocities
      velocity = det_velocity + stoch_velocity

      # Update position   
      for k, b in enumerate(self.bodies):
        b.location_new = b.location_old + dt * velocity[3*k : 3*(k+1)]

      # Check if configuration is valid and update postions if so      
      valid_configuration = True
      if self.domain == 'single_wall':
        for b in self.bodies:
          if b.location_new[2] < 0.0:      
            valid_configuration = False
            break
      if valid_configuration is True:
        for b in self.bodies:
          b.location = b.location_new          
          if b.location[2] < self.a:
            self.wall_overlaps += 1              
        return

      self.invalid_configuration_count += 1
      print 'Invalid configuration'
    return


  def deterministic_adams_bashforth(self, dt):
    '''
    Take a time step of length dt using a first order
    stochastic integrator.
    '''
    while True:
      # Save initial configuration
      for k, b in enumerate(self.bodies):
        b.location_old = b.location

      # Compute deterministic velocity
      if self.hydro_interactions==1: 
      	det_velocity, det_torque = self.compute_deterministic_velocity_and_torque()
      else:
      	det_velocity, det_torque = self.compute_deterministic_velocity_and_torque_uncorrelated()


      # Add velocities
      if self.first_step is False:
        # Use Adams-Bashforth
        velocity = 1.5 * det_velocity - 0.5 * self.velocities_previous_step 
      else:
        # Use forward Euler
        velocity = det_velocity
        self.first_step = False

      # Update position   
      for k, b in enumerate(self.bodies):
        b.location_new = b.location_old + dt * velocity[3*k : 3*(k+1)]

      # Check if configuration is valid and update postions if so      
      valid_configuration = True
      if self.domain == 'single_wall':    
        for k, b in enumerate(self.bodies):
          if b.location_new[2] < 0.0:      
            valid_configuration = False
            break
      if valid_configuration is True:
        self.first_step = False
        self.velocities_previous_step = det_velocity
        for b in self.bodies:
          b.location = b.location_new          
          if self.domain == 'single_wall':    
            if b.location[2] < self.a:
              self.wall_overlaps += 1      
        return

      self.invalid_configuration_count += 1
      print 'Invalid configuration'
    return
    

  def stochastic_adams_bashforth(self, dt):
    '''
    Take a time step of length dt using a first order
    stochastic integrator.
    '''
    while True:
      # Save initial configuration
      for k, b in enumerate(self.bodies):
        b.location_old = b.location

      if self.hydro_interactions==1: 
         # Compute deterministic velocity
      	det_velocity, det_torque = self.compute_deterministic_velocity_and_torque()
      	# Compute stochastic velocity
      	stoch_velocity = self.compute_stochastic_linear_velocity(dt)
      else:
      	det_velocity, det_torque = self.compute_deterministic_velocity_and_torque_uncorrelated()
      	stoch_velocity = self.compute_stochastic_linear_velocity_uncorrelated (dt)

      # Add velocities
      if self.first_step is False:
        # Use Adams-Bashforth
        velocity = 1.5 * det_velocity - 0.5 * self.velocities_previous_step + stoch_velocity
      else:
        # Use forward Euler
        velocity = det_velocity + stoch_velocity
        self.first_step = False

      # Update position   
      for k, b in enumerate(self.bodies):
        b.location_new = b.location_old + dt * velocity[3*k : 3*(k+1)]

      # Check if configuration is valid and update postions if so      
      valid_configuration = True
      if self.domain == 'single_wall':
        for k, b in enumerate(self.bodies):
          if b.location_new[2] < 0.0:      
            valid_configuration = False
            break
      if valid_configuration is True:
        self.first_step = False
        self.velocities_previous_step = det_velocity
        for b in self.bodies:
          b.location = b.location_new          
          if self.domain == 'single_wall':
            if b.location[2] < self.a:
              self.wall_overlaps += 1      
        return

      self.invalid_configuration_count += 1
      print 'Invalid configuration'
    return
    

  def stochastic_mid_point(self, dt):
    '''
    Take a time step of length dt using a first order
    stochastic integrator. 

    q^{n+1/2} = q^n + 0.5 * dt * (M*F)^n + sqrt(kT*dt) * (M^n)^{1/2} * W_1
    q^{n+1} = q^n + dt * (M*F)^{n+1/2} 
            + sqrt(kT*dt) * ((M^n)^{1/2} * W_1 + (M^{n+1/2})^{1/2} * W_2)
            + (kT/delta) * dt * (M(q^n+0.5*delta) - M(q^n-0.5*delta))

    The force F also includes the deterministic torque.
    '''
    while True:
      # Save initial configuration
      for k, b in enumerate(self.bodies):
        b.location_old = b.location

      # Compute thermal drift (kT * div_t(M_tt))
      drift = self.compute_linear_thermal_drift()

      # Compute deterministic velocity and torque
      det_velocity, det_torque = self.compute_deterministic_velocity_and_torque()

      # Compute stochastic velocity without drift for first half step
      stoch_velocity_1 = self.compute_stochastic_linear_velocity_without_drift(0.5 * dt)

      # Add velocities
      velocity = det_velocity + stoch_velocity_1
      
      # Update blobs coordinates half time step
      for k, b in enumerate(self.bodies):
        b.location = b.location_old + (0.5 * dt) * velocity[3*k : 3*(k+1)]

      # Check if configuration is valid if not repeat step
      if self.domain == 'single_wall':    
        valid_configuration = True
        for k, b in enumerate(self.bodies):
          if b.location[2] < 0.0:      
            valid_configuration = False
            self.invalid_configuration_count += 1
            print 'Invalid configuration'
            break
        if valid_configuration is False:
          # Restore configuration
          for k, b in enumerate(self.bodies):
            b.location = b.location_old
          continue

      # Compute deterministic velocity and torque
      det_velocity, det_torque = self.compute_deterministic_velocity_and_torque()

      # Compute stochastic velocity without drift for second half step
      stoch_velocity_2 = self.compute_stochastic_linear_velocity_without_drift(0.5 * dt)

      # Add velocities 
      velocity = det_velocity + drift + (stoch_velocity_1 + stoch_velocity_2) * 0.5

      # Update blobs coordinates half time step
      for k, b in enumerate(self.bodies):
        b.location = b.location_old + dt * velocity[3*k : 3*(k+1)]
      
      # Check if configuration is valid if not repeat step
      if self.domain == 'single_wall':
        valid_configuration = True
        for k, b in enumerate(self.bodies):
          if b.location[2] < 0.0:      
            valid_configuration = False
            self.invalid_configuration_count += 1
            print 'Invalid configuration'
            break
        if valid_configuration is False:
          # Restore configuration
          for k, b in enumerate(self.bodies):
            b.location = b.location_old
          continue
        # Count overlaps and return
        for k, b in enumerate(self.bodies):
          if b.location[2] < self.a:
            self.wall_overlaps += 1            
      return


  def stochastic_mid_point_version_2(self, dt):
    '''
    Take a time step of length dt using a first order
    stochastic integrator. 

    q^{n+1/2} = q^n + 0.5 * dt * (M*F)^n + sqrt(kT*dt) * (M^n)^{1/2} * W_1
    q^{n+1} = q^n + dt * (M*F)^{n+1/2} 
            + sqrt(kT*dt) * (M^n)^{1/2} * (W_1 + W_2)
            + (kT/delta) * dt * (M(q^n+0.5*delta) - M(q^n-0.5*delta))

    The force F also includes the deterministic torque.
    '''
    while True:
      # Save initial configuration
      for k, b in enumerate(self.bodies):
        b.location_old = b.location

      # Compute thermal drift (kT * div_t(M_tt))
      drift = self.compute_linear_thermal_drift()

      # Compute deterministic velocity and torque
      det_velocity, det_torque = self.compute_deterministic_velocity_and_torque()

      # Compute stochastic velocity without drift for first half step
      stoch_velocity_1 = self.compute_stochastic_linear_velocity_without_drift(0.5 * dt)

      # Compute stochastic velocity without drift for second half step
      stoch_velocity_2 = self.compute_stochastic_linear_velocity_without_drift(0.5 * dt)

      # Add velocities
      velocity = det_velocity + stoch_velocity_1
      
      # Update blobs coordinates half time step
      for k, b in enumerate(self.bodies):
        b.location = b.location_old + (0.5 * dt) * velocity[3*k : 3*(k+1)]

      # Check if configuration is valid if not repeat step
      if self.domain == 'single_wall':
        valid_configuration = True
        for k, b in enumerate(self.bodies):
          if b.location[2] < 0.0:      
            valid_configuration = False
            self.invalid_configuration_count += 1
            print 'Invalid configuration'
            break
        if valid_configuration is False:
          # Restore configuration
          for k, b in enumerate(self.bodies):
            b.location = b.location_old
          continue

      # Compute deterministic velocity and torque
      det_velocity, det_torque = self.compute_deterministic_velocity_and_torque()

      # Add velocities 
      velocity = det_velocity + drift + (stoch_velocity_1 + stoch_velocity_2) * 0.5

      # Update blobs coordinates half time step
      for k, b in enumerate(self.bodies):
        b.location = b.location_old + dt * velocity[3*k : 3*(k+1)]
      
      # Check if configuration is valid if not repeat step
      if self.domain == 'single_wall':      
        valid_configuration = True
        for k, b in enumerate(self.bodies):
          if b.location[2] < 0.0:      
            valid_configuration = False
            self.invalid_configuration_count += 1
            print 'Invalid configuration'
            break
        if valid_configuration is False:
          # Restore configuration
          for k, b in enumerate(self.bodies):
            b.location = b.location_old
          continue
        # Count overlaps and return
        for k, b in enumerate(self.bodies):
          if b.location[2] < self.a:
            self.wall_overlaps += 1            
      return


  def stochastic_trapezoidal(self, dt):
    '''
    Take a time step of length dt using a first order
    stochastic integrator. 

    q^{*} = q^n + dt * (M*F)^n + sqrt(2*kT*dt) * (M^n)^{1/2} * W_1
    q^{n+1} = q^n + 0.5 * dt * ( (M*F)^n + (M*F)^{*}) 
            + sqrt(2*kT*dt) * (M^n)^{1/2} 
            + (kT/delta) * dt * (M(q^n+0.5*delta) - M(q^n-0.5*delta))

    The force F also includes the deterministic torque.
    '''
    while True:
      # Save initial configuration
      for k, b in enumerate(self.bodies):
        b.location_old = b.location

      # Compute thermal drift (kT * div_t(M_tt))
      drift = self.compute_linear_thermal_drift()

      # Compute deterministic velocity and torque
      det_velocity_1, det_torque = self.compute_deterministic_velocity_and_torque()

      # Compute stochastic velocity without drift 
      stoch_velocity = self.compute_stochastic_linear_velocity_without_drift(dt)

      # Add velocities
      velocity = det_velocity_1 + stoch_velocity
      
      # Update blobs coordinates half time step
      for k, b in enumerate(self.bodies):
        b.location = b.location_old + dt * velocity[3*k : 3*(k+1)]

      # Check if configuration is valid if not repeat step
      if self.domain == 'single_wall':
        valid_configuration = True
        for k, b in enumerate(self.bodies):
          if b.location[2] < 0.0:      
            valid_configuration = False
            self.invalid_configuration_count += 1
            print 'Invalid configuration'
            break
        if valid_configuration is False:
          # Restore configuration
          for k, b in enumerate(self.bodies):
            b.location = b.location_old
          continue

      # Compute deterministic velocity and torque
      det_velocity_2, det_torque = self.compute_deterministic_velocity_and_torque()

      # Add velocities 
      velocity = 0.5 * (det_velocity_1 + det_velocity_2) + drift + stoch_velocity

      # Update blobs coordinates half time step
      for k, b in enumerate(self.bodies):
        b.location = b.location_old + dt * velocity[3*k : 3*(k+1)]
      
      # Check if configuration is valid if not repeat step
      if self.domain == 'single_wall':
        valid_configuration = True
        for k, b in enumerate(self.bodies):
          if b.location[2] < 0.0:      
            valid_configuration = False
            self.invalid_configuration_count += 1
            print 'Invalid configuration'
            break
        if valid_configuration is False:
          # Restore configuration
          for k, b in enumerate(self.bodies):
            b.location = b.location_old
          continue
        # Count overlaps and return
        for k, b in enumerate(self.bodies):
          if b.location[2] < self.a:
            self.wall_overlaps += 1            
      return


  def compute_deterministic_velocity_and_torque(self):
    '''
    Compute the torque on bodies rotating with a prescribed
    angular velocity and subject to forces, i.e., solve the
    linear system
    
    M_rr * T = omega - M_rt * forces
    
    Then compute the translational velocity

    v = M_tr * T + M_tt * forces
    
    It returns the velocities and torques (v,T).
    '''
    # Create auxiliar variables
    Nblobs = len(self.bodies)
    blob_mass = 1.0

    # Get blobs coordinates
    r_vectors_blobs = np.empty((Nblobs, 3))
    for k, b in enumerate(self.bodies):
      r_vectors_blobs[k] = b.location

    # Compute one-blob forces (same function for all blobs)
    # force = np.zeros(r_vectors_blobs.size)
    force = self.calc_one_blob_forces(r_vectors_blobs, blob_radius = self.a, blob_mass = blob_mass)

    # Compute blob-blob forces (same function for all pair of blobs)
    force += self.calc_blob_blob_forces(r_vectors_blobs, blob_radius = self.a)  
    force = np.reshape(force, force.size)

    # Use constraint motion or free kinematics
    if self.free_kinematics == 'False':
      # Set rollers angular velocity
      omega = np.empty(3 * len(self.bodies))
      for i in range(len(self.bodies)):
        omega[3*i : 3*(i+1)] = self.get_omega_one_roller()

      # Set RHS = omega - M_rt * force 
      RHS = omega - self.mobility_rot_times_force(r_vectors_blobs, force, self.eta, self.a, periodic_length = self.periodic_length)

      # Set linear operator 
      system_size = 3 * len(self.bodies)
      def mobility_rot_torque(torque, r_vectors = None, eta = None, a = None, periodic_length = None):
        return self.mobility_rot_times_torque(r_vectors, torque, eta, a, periodic_length = periodic_length)
      linear_operator_partial = partial(mobility_rot_torque, r_vectors = r_vectors_blobs, eta = self.eta, a = self.a, periodic_length = self.periodic_length)
      A = spla.LinearOperator((system_size, system_size), matvec = linear_operator_partial, dtype='float64')

      # Scale RHS to norm 1
      RHS_norm = np.linalg.norm(RHS)
      if RHS_norm > 0:
        RHS = RHS / RHS_norm

      # Solve linear system 
      counter = gmres_counter(print_residual = self.print_residual)
      (sol_precond, info_precond) = spla.gmres(A, 
                                               RHS, 
                                               x0=self.deterministic_torque_previous_step, 
                                               tol=self.tolerance, 
                                               maxiter=1000,
                                               callback = counter) 
      self.det_iterations_count += counter.niter
      self.deterministic_torque_previous_step = sol_precond

      # Scale solution with RHS norm
      if RHS_norm > 0:
        sol_precond = sol_precond * RHS_norm
    else:
      # This is free kinematics, compute torque
      sol_precond = self.get_torque()

    # Compute linear velocity
    velocity  = self.mobility_trans_times_force(r_vectors_blobs, force, self.eta, self.a, periodic_length = self.periodic_length)
    if np.any(sol_precond):
      velocity += self.mobility_trans_times_torque(r_vectors_blobs, sol_precond, self.eta, self.a, periodic_length = self.periodic_length)
    
    # Return linear velocity and torque
    return velocity, sol_precond
      

  def compute_deterministic_velocity_and_torque_uncorrelated(self):
    '''
    Compute the torque on bodies rotating with a prescribed
    angular velocity and subject to forces, i.e., solve the
    linear system
    
    M_rr * T = omega - M_rt * forces
    
    Then compute the translational velocity

    v = M_tr * T + M_tt * forces
    
    It returns the velocities and torques (v,T).
    '''
    # Create auxiliar variables
    Nblobs = len(self.bodies)
    blob_mass = 1.0

    # Get blobs coordinates
    r_vectors_blobs = np.empty((Nblobs, 3))
    for k, b in enumerate(self.bodies):
      r_vectors_blobs[k] = b.location

    torque = np.empty(3 * Nblobs)
    
    # Compute one-blob forces (same function for all blobs)
    force = self.calc_one_blob_forces(r_vectors_blobs, blob_radius = self.a, blob_mass = blob_mass) 
    # Compute blob-blob forces (same function for all pair of blobs)
    force += self.calc_blob_blob_forces(r_vectors_blobs, blob_radius = self.a)  
    force = np.reshape(force, force.size)
    
    # Define prefactor for self rr mobility: 1/(6*pi*eta*a**3)
    factor_mob_tt = 1/(6*m.pi*self.eta*self.a)
    
    # Define prefactor for self rr mobility: 1/(6*pi*eta*a**3)
    factor_mob_rr = 1/(6*m.pi*self.eta*self.a**3)
    
    # Define prefactor for self rt mobility: 1/(6*pi*eta*a**2)
    factor_mob_rt = 1/(6*m.pi*self.eta*self.a**2)
    
    
    # Compute mobility coeffs
    h_adim_eff = np.empty(Nblobs)
    damping = np.empty(Nblobs)
    mu_rt_para  = np.empty(Nblobs)
    mu_tt_para  = np.empty(Nblobs)
    mu_tt_perp  = np.empty(Nblobs)
    for i in range(Nblobs):        
        # max(h/a,1) : artifact to ensure that mobility goes to zero
          h_over_a = r_vectors_blobs[i][2]/self.a
    	  h_adim_eff[i] = max(h_over_a, 1.0)    
    	  # Damping factor : damping = 1.0 if z_i >= blob_radius, damping = z_i / blob_radius if 0< z_i < blob_radius, damping = 0 if z_i < 0
    	  damping[i] = 1.0
    	  if h_over_a < 0.0:
	    damping[i] = 0.0 
	  elif h_over_a <= 1.0:
	    damping[i] = h_over_a
	    
    	  # mu_rt_para from Swan Brady
    	  mu_rt_para[i] = factor_mob_rt*( 3/(32*h_adim_eff[i]**4) ) * damping[i]
    	  # mu_perp from Swan Brady
    	  mu_tt_perp[i] = factor_mob_tt*( 1 - 9/(8*h_adim_eff[i]) + 1/(2*h_adim_eff[i]**3) - 1/(8*h_adim_eff[i]**5) ) * damping[i]
        # mu_para from Swan Brady
    	  mu_tt_para[i] = factor_mob_tt*( 1 - 9/(16*h_adim_eff[i]) + 2/(16*h_adim_eff[i]**3) - 1/(16*h_adim_eff[i]**5) ) * damping[i]

    # Use constraint motion or free kinematics
    if self.free_kinematics == 'False':
      # Set rollers angular velocity
      omega = np.empty(3 * Nblobs)      
      
      for i in range(Nblobs):
      	omega[3*i : 3*(i+1)] = self.get_omega_one_roller()
      	# mu_rr_perp from Swan Brady
      	mu_rr_perp_inv = 1/( factor_mob_rr*( 3/4 - 3/(32*h_adim_eff[i]**3) ) * damping[i]) 
      	# mu_rr_para from Swan Brady
      	mu_rr_para_inv = 1/( factor_mob_rr*( 3/4 - 15/(64*h_adim_eff[i]**3) ) * damping[i]) 
      	#  T = M_rr^{-1}*( omega - M_rt * forces )
        torque[3*i] = mu_rr_para_inv*(omega[3*i] + mu_rt_para[i]*force[3*i+1])
        torque[3*i+1] = mu_rr_para_inv*(omega[3*i+1] - mu_rt_para[i]*force[3*i])
        torque[3*i+2] = mu_rr_perp_inv*omega[3*i+2]

     
    else:
      # This is free kinematics, compute torque
      torque = self.get_torque()

    # Compute linear velocity
    velocity = np.empty(3 * Nblobs)
    for i in range(Nblobs):
    	velocity[3*i] = mu_tt_para[i]*force[3*i] + mu_rt_para[i]*torque[3*i+1]
    	velocity[3*i+1] = mu_tt_para[i]*force[3*i+1] - mu_rt_para[i]*torque[3*i]
    	velocity[3*i+2] = mu_tt_perp[i]*force[3*i+2]
    	

    # Return linear velocity and torque
    return velocity, torque


  def compute_stochastic_velocity(self, dt):
    '''
    Compute stochastic torque and velocity. First,
    solve for the torque
    
    M_rr * T = -kT*div_t(M_rt) - sqrt(2*kT) * (N^{1/2}*W)_r,

    then set linear velocity
    
    v_stoch = M_tr * T + sqrt(2*kT) * (N^{1/2}*W)_t + kT*div_t(M_tt).

    Here N = (M_tt M_tr; M_rt M_rr) is the grand mobility matrix.
    We use random finite difference to compute the divergence
    terms. Note that in principle we should include the term
    div_r(M_rr) in the torque equation and div_r(M_tr) in the
    velocity equation but they are zero for a roller.

    This function returns the stochastic velocity v_stoch.
    '''
    # Create auxiliar variables
    Nblobs = len(self.bodies)
    blob_mass = 1.0

    # Get blobs coordinates
    r_vectors_blobs = np.empty((Nblobs, 3))
    for k, b in enumerate(self.bodies):
      r_vectors_blobs[k] = b.location

    # Generate random vector
    z = np.random.randn(6 * Nblobs)
    
    # Define grand mobility matrix
    def grand_mobility_matrix(force_torque, r_vectors = None, eta = None, a = None, periodic_length = None):
      half_size = force_torque.size / 2
      # velocity = self.mobility_trans_times_force_torque(r_vectors, force_torque[0:half_size], force_torque[half_size:], eta, a, periodic_length = periodic_length)
      velocity  = self.mobility_trans_times_force(r_vectors, force_torque[0:half_size], eta, a, periodic_length = periodic_length)
      velocity += self.mobility_trans_times_torque(r_vectors, force_torque[half_size:], eta, a, periodic_length = periodic_length)
      angular_velocity  = self.mobility_rot_times_force(r_vectors, force_torque[0:half_size], eta, a, periodic_length = periodic_length)
      angular_velocity += self.mobility_rot_times_torque(r_vectors, force_torque[half_size:], eta, a, periodic_length = periodic_length)      
      return np.concatenate([velocity, angular_velocity])
    partial_grand_mobility_matrix = partial(grand_mobility_matrix, 
                                            r_vectors = r_vectors_blobs, 
                                            eta = self.eta, 
                                            a = self.a,
                                            periodic_length = self.periodic_length)

    # Generate noise term sqrt(2*kT) * N^{1/2} * z
    velocities_noise, it_lanczos = stochastic.stochastic_forcing_lanczos(factor = np.sqrt(2*self.kT / dt),
                                                                         tolerance = self.tolerance, 
                                                                         dim = self.Nblobs * 6, 
                                                                         mobility_mult = partial_grand_mobility_matrix,
                                                                         z = z,
                                                                         print_residual = self.print_residual)
    self.stoch_iterations_count += it_lanczos

    # Compute divergence terms div_t(M_rt) and div_t(M_tt)
    if self.kT > 0.0 and self.domain != 'no_wall':
      # 1. Generate random displacement
      dx_stoch = np.reshape(np.random.randn(Nblobs * 3), (Nblobs, 3))
      # 2. Displace blobs
      r_vectors_blobs += dx_stoch * (self.rf_delta * self.a * 0.5)
      # 3. Compute M_rt(r+0.5*dx) * dx_stoch
      div_M_rt = self.mobility_rot_times_force(r_vectors_blobs, np.reshape(dx_stoch, dx_stoch.size), self.eta, self.a, 
                                                                 periodic_length = self.periodic_length)
      div_M_tt = self.mobility_trans_times_force(r_vectors_blobs, np.reshape(dx_stoch, dx_stoch.size), self.eta, self.a, 
                                                                   periodic_length = self.periodic_length)
      # 4. Displace blobs in the other direction
      r_vectors_blobs -= dx_stoch * self.rf_delta * self.a 

      # 5. Compute -M_rt(r-0.5*dx) * dx_stoch
      div_M_rt -= self.mobility_rot_times_force(r_vectors_blobs, np.reshape(dx_stoch, dx_stoch.size), self.eta, self.a, 
                                                                  periodic_length = self.periodic_length)
      div_M_tt -= self.mobility_trans_times_force(r_vectors_blobs, np.reshape(dx_stoch, dx_stoch.size), self.eta, self.a, 
                                                                    periodic_length = self.periodic_length)
    
      # Reset blobs location
      r_vectors_blobs += dx_stoch * (self.rf_delta * self.a * 0.5)
    else:
      div_M_rt = np.zeros(Nblobs * 3)
      div_M_tt = np.zeros(Nblobs * 3)

    # Use constraint motion or free kinematics
    if self.free_kinematics == 'False':
       # Set RHS = -kT*div_t(M_rt) - sqrt(2*kT) * (N^{1/2}*W)_r,
      RHS = -velocities_noise[velocities_noise.size / 2:] - div_M_rt * (self.kT / (self.rf_delta * self.a))

      # Set linear operator
      system_size = 3 * len(self.bodies)
      def mobility_rot_torque(torque, r_vectors = None, eta = None, a = None, periodic_length = None):
        return self.mobility_rot_times_torque(r_vectors, torque, eta, a, periodic_length = periodic_length)
      linear_operator_partial = partial(mobility_rot_torque, r_vectors = r_vectors_blobs, eta = self.eta, a = self.a, periodic_length = self.periodic_length)
      A = spla.LinearOperator((system_size, system_size), matvec = linear_operator_partial, dtype='float64')

      # Scale RHS to norm 1
      RHS_norm = np.linalg.norm(RHS)
      if RHS_norm > 0:
        RHS = RHS / RHS_norm

      # Solve linear system 
      counter = gmres_counter(print_residual = self.print_residual)
      (sol_precond, info_precond) = spla.gmres(A, 
                                               RHS, 
                                               tol=self.tolerance, 
                                               maxiter=1000,
                                               callback=counter) 
      self.det_iterations_count += counter.niter

      # Scale solution with RHS norm
      if RHS_norm > 0:
        sol_precond = sol_precond * RHS_norm
    else:
      # This is free kinematics, stochastic torque set to zero
      # because we only care for the translational degrees of freedom
      sol_precond = np.zeros(3 * Nblobs)

    # Compute stochastic velocity v_stoch = M_tr * T + sqrt(2*kT) * (N^{1/2}*W)_t + kT*div_t(M_tt).
    v_stoch = self.mobility_trans_times_torque(r_vectors_blobs, sol_precond, self.eta, self.a, periodic_length = self.periodic_length)
    v_stoch += velocities_noise[0 : velocities_noise.size / 2] + (self.kT / (self.rf_delta * self.a)) * div_M_tt 
    return v_stoch


  def compute_stochastic_linear_velocity(self, dt):
    '''
    Compute stochastic linear velocity
    
    v_stoch = sqrt(2*kT) * M_tt^{1/2}*W + kT*div_t(M_tt).

    This function returns the stochastic velocity v_stoch.
    '''
    # Create auxiliar variables
    Nblobs = len(self.bodies)

    # Get blobs coordinates
    r_vectors_blobs = np.empty((Nblobs, 3))
    for k, b in enumerate(self.bodies):
      r_vectors_blobs[k] = b.location

    # Generate random vector
    z = np.random.randn(3 * Nblobs)
    
    # Define mobility matrix
    def mobility_matrix(force, r_vectors = None, eta = None, a = None, periodic_length = None):
      return self.mobility_trans_times_force(r_vectors, force, eta, a, periodic_length = periodic_length)
    partial_mobility_matrix = partial(mobility_matrix, 
                                            r_vectors = r_vectors_blobs, 
                                            eta = self.eta, 
                                            a = self.a,
                                            periodic_length = self.periodic_length)

    # Generate noise term sqrt(2*kT) * N^{1/2} * z
    velocities_noise, it_lanczos = stochastic.stochastic_forcing_lanczos(factor = np.sqrt(2*self.kT / dt),
                                                                         tolerance = self.tolerance, 
                                                                         dim = self.Nblobs * 3, 
                                                                         mobility_mult = partial_mobility_matrix,
                                                                         z = z,
                                                                         print_residual = self.print_residual)
    self.stoch_iterations_count += it_lanczos

    # Compute divergence term div_t(M_tt)
    # 1. Generate random displacement
    if self.kT > 0.0 and self.domain != 'no_wall':
      dx_stoch = np.reshape(np.random.randn(Nblobs * 3), (Nblobs, 3))
      # 2. Displace blobs
      r_vectors_blobs += dx_stoch * (self.rf_delta * self.a * 0.5)
      # 3. Compute M_tt(r+0.5*dx) * dx_stoch
      div_M_tt = self.mobility_trans_times_force(r_vectors_blobs, np.reshape(dx_stoch, dx_stoch.size), self.eta, self.a, 
                                                                   periodic_length = self.periodic_length)
      # 4. Displace blobs in the other direction
      r_vectors_blobs -= dx_stoch * self.rf_delta * self.a
      # 5. Compute -M_tt(r-0.5*dx) * dx_stoch
      div_M_tt -= self.mobility_trans_times_force(r_vectors_blobs, np.reshape(dx_stoch, dx_stoch.size), self.eta, self.a, 
                                                                    periodic_length = self.periodic_length)
      # Reset blobs location
      r_vectors_blobs += dx_stoch * (self.rf_delta * self.a * 0.5)
    else:
      div_M_tt = np.zeros(velocities_noise.size)

    # Compute stochastic velocity v_stoch = sqrt(2*kT/dt) * M_tt^{1/2}*W + kT*div_t(M_tt).
    return velocities_noise + (self.kT / (self.rf_delta * self.a)) * div_M_tt 
  
  
  def compute_stochastic_linear_velocity_uncorrelated(self, dt):
    '''
    Compute stochastic linear velocity for uncorrelated particles
    
    v_stoch = sqrt(2*kT) * M_tt^{1/2}*W + kT*div_t(M_tt).

    This function returns the stochastic velocity v_stoch.
    '''
    # Create auxiliar variables
    Nblobs = len(self.bodies)

    # Get blobs coordinates
    r_vectors_blobs = np.empty((Nblobs, 3))
    for k, b in enumerate(self.bodies):
      r_vectors_blobs[k] = b.location

    # Generate random vector
    z = np.random.randn(3 * Nblobs)
    
    # Define prefactor for self mobility: 1/(6*pi*eta*a)
    factor_mob_tt = 1/(6*m.pi*self.eta*self.a)
    
    # Define prefactor for Brownian displacement: sqrt(2kT/dt)
    factor_disp = np.sqrt(2*self.kT / dt)
    
    # Define velocity vector
    v_stoch = np.empty(3*Nblobs)
    
    for k in range(Nblobs):
    	# max(h/a,1) : artifact to ensure that mobility goes to zero
	h_over_a = r_vectors_blobs[k][2]/self.a
	h_adim_eff = max(h_over_a, 1.0)    
	# Damping factor : damping = 1.0 if z_i >= blob_radius, damping = z_i / blob_radius if 0< z_i < blob_radius, damping = 0 if z_i < 0
	damping = 1.0
	if h_over_a < 0.0:
	  damping = 0.0 
	elif h_over_a <= 1.0:
	  damping = h_over_a
    	# mu_perp from Swan Brady
    	mu_tt_perp = factor_mob_tt*( 1 - 9/(8*h_adim_eff) + 1/(2*h_adim_eff**3) - 1/(8*h_adim_eff**5) ) * damping
    	# mu_para from Swan Brady
    	mu_tt_para = factor_mob_tt*( 1 - 9/(16*h_adim_eff) + 2/(16*h_adim_eff**3) - 1/(16*h_adim_eff**5) ) * damping
    	# div(Mtt) has one nonzero term: d(mu_perp)/dh 
    	deriv_mu_tt_perp = factor_mob_tt*( 9/(8*h_adim_eff**2) - 3/(2*h_adim_eff**4) + 5/(8*h_adim_eff**6) ) * damping
    	
    	# Compute stochastic velocity v_stoch = sqrt(2*kT/dt) * M_tt^{1/2}*W + kT*div_t(M_tt).
    	v_stoch[3*k:3*k+2] = factor_disp*np.sqrt(mu_tt_para)*z[3*k:3*k+2]
    	v_stoch[3*k+2] = factor_disp*np.sqrt(mu_tt_perp)*z[3*k+2] + self.kT*deriv_mu_tt_perp  
    
    return v_stoch 


  def compute_stochastic_linear_velocity_without_drift(self, dt):
    '''
    Compute stochastic linear velocity
    
    v_stoch = sqrt(2*kT) * M_tt^{1/2}*W 

    This function returns the stochastic velocity v_stoch.
    '''
    # Create auxiliar variables
    Nblobs = len(self.bodies)

    # Get blobs coordinates
    r_vectors_blobs = np.empty((Nblobs, 3))
    for k, b in enumerate(self.bodies):
      r_vectors_blobs[k] = b.location

    # Generate random vector
    z = np.random.randn(3 * Nblobs)
    
    # Define mobility matrix
    def mobility_matrix(force, r_vectors = None, eta = None, a = None, periodic_length = None):
      return self.mobility_trans_times_force(r_vectors, force, eta, a, periodic_length = periodic_length)
    partial_mobility_matrix = partial(mobility_matrix, 
                                            r_vectors = r_vectors_blobs, 
                                            eta = self.eta, 
                                            a = self.a,
                                            periodic_length = self.periodic_length)

    # Generate noise term sqrt(2*kT) * N^{1/2} * z
    velocities_noise, it_lanczos = stochastic.stochastic_forcing_lanczos(factor = np.sqrt(2*self.kT / dt),
                                                                         tolerance = self.tolerance, 
                                                                         dim = self.Nblobs * 3, 
                                                                         mobility_mult = partial_mobility_matrix,
                                                                         z = z,
                                                                         print_residual = self.print_residual)
    self.stoch_iterations_count += it_lanczos

    # Return velocity
    return velocities_noise 


  def compute_linear_thermal_drift(self):
    '''
    Compute the thermal drift kT*div_t(M_tt) using random finite
    differences. 
    
    drift = (kT / delta) * (M_tt(q+0.5*delta) - M_tt(q^n-0.5*delta))
    '''
    # Create auxiliar variables
    Nblobs = len(self.bodies)

    # Compute divergence term div_t(M_tt)
    if self.kT > 0.0 and self.domain != 'no_wall':
      # 0. Get blobs coordinates
      r_vectors_blobs = np.empty((Nblobs, 3))
      for k, b in enumerate(self.bodies):
        r_vectors_blobs[k] = b.location
      # 1. Generate random displacement
      dx_stoch = np.reshape(np.random.randn(Nblobs * 3), (Nblobs, 3))
      # 2. Displace blobs
      r_vectors_blobs += dx_stoch * (self.rf_delta * self.a * 0.5)
      # 3. Compute M_tt(r+0.5*dx) * dx_stoch
      div_M_tt = self.mobility_trans_times_force(r_vectors_blobs, np.reshape(dx_stoch, dx_stoch.size), self.eta, self.a, 
                                                                   periodic_length = self.periodic_length)
      # 4. Displace blobs in the other direction
      r_vectors_blobs -= dx_stoch * self.rf_delta * self.a 
      # 5. Compute -M_tt(r-0.5*dx) * dx_stoch
      div_M_tt -= self.mobility_trans_times_force(r_vectors_blobs, np.reshape(dx_stoch, dx_stoch.size), self.eta, self.a, 
                                                                    periodic_length = self.periodic_length)
    else:
      div_M_tt = np.zeros(velocities_noise.size)
    return (self.kT / (self.rf_delta * self.a)) * div_M_tt 


  def get_omega_one_roller(self):
    '''
    The the angular velocity of one roller for
    prescribed kinematics. 
    '''
    return self.omega_one_roller


  def get_torque(self):
    '''
    Get torque acting on the blobs for free kinematics.
    In this version is set to zero.
    '''
    Nblobs = len(self.bodies)
    torques = np.empty(3 * Nblobs)
    for i in range(Nblobs):
        torques[3*i: 3*(i+1)] = self.get_omega_one_roller()*8.0*m.pi*self.eta*self.a**3
    return torques


class gmres_counter(object):
  '''
  Callback generator to count iterations. 
  '''
  def __init__(self, print_residual = False):
    self.print_residual = print_residual
    self.niter = 0
  def __call__(self, rk=None):
    self.niter += 1
    if self.print_residual is True:
      if self.niter == 1:
        print 'gmres =  0 1'
      print 'gmres = ', self.niter, rk

