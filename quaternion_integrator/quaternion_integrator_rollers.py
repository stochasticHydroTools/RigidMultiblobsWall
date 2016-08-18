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
    self.deterministic_torque_previous_step = None
    self.first_step = True
    self.kT = 0.0
    self.tolerance = 1e-08
    self.rf_delta = 1e-05

    # Optional variables
    self.build_stochastic_block_diagonal_preconditioner = None
    self.periodic_length = None
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
    return getattr(self, self.scheme.replace('_rollers', ''))(dt)
    

  def deterministic_forward_euler(self, dt): 
    ''' 
    Take a time step of length dt using the deterministic forward Euler scheme. 
    ''' 
    while True: 
      # Compute deterministic velocity
      det_velocity, det_torque = self.compute_deterministic_velocity_and_torque()

      # Update position   
      for k, b in enumerate(self.bodies):
        b.location_new = b.location + dt * det_velocity[3*k : 3*(k+1)]

      # Check if configuration is valid and update postions if so
      valid_configuration = True
      for b in self.bodies:
        if b.location_new[2] < self.a:      
          valid_configuration = False
          break
      if valid_configuration is True:
        for b in self.bodies:
          b.location = b.location_new            
        return
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

      # Compute deterministic velocity
      det_velocity, det_torque = self.compute_deterministic_velocity_and_torque()
      
      # Compute stochastic velocity
      stoch_velocity = self.compute_stochastic_velocity(dt)

      # Add velocities
      velocity = det_velocity + stoch_velocity

      # Update position   
      for k, b in enumerate(self.bodies):
        b.location_new = b.location_old + dt * velocity[3*k : 3*(k+1)]

      # Check if configuration is valid and update postions if so
      valid_configuration = True
      for b in self.bodies:
        if b.location_new[2] < self.a:      
          valid_configuration = False
          break
      if valid_configuration is True:
        for b in self.bodies:
          b.location = b.location_new            
        return
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

      # Compute deterministic velocity
      det_velocity, det_torque = self.compute_deterministic_velocity_and_torque()
      
      # Compute stochastic velocity
      stoch_velocity = self.compute_stochastic_velocity(dt)

      # Add velocities
      if self.First_step is False:
        # Use Adams-Bashforth
        velocity = 1.5 * det_velocity - 0.5 * self.velocities_previous_step + stoch_velocity
      else:
        # Use forward Euler
        velocity = det_velocity + stoch_velocity

      # Update position   
      for k, b in enumerate(self.bodies):
        b.location_new = b.location_old + dt * velocity[3*k : 3*(k+1)]

      # Check if configuration is valid and update postions if so
      valid_configuration = True
      for b in self.bodies:
        if b.location_new[2] < self.a:      
          valid_configuration = False
          break
      if valid_configuration is True:
        self.first_step = False
        self.velocities_previous_step = det_velocity
        for b in self.bodies:
          b.location = b.location_new            
        return
      print 'Invalid configuration'
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
    force = np.zeros(r_vectors_blobs.size)
    force = self.calc_one_blob_forces(r_vectors_blobs, blob_radius = self.a, blob_mass = blob_mass)
  
    # Compute blob-blob forces (same function for all pair of blobs)
    force += self.calc_blob_blob_forces(r_vectors_blobs, blob_radius = self.a)  
    force = np.reshape(force, force.size)

    # Set rollers angular velocity
    omega = np.empty(3 * len(self.bodies))
    for i in range(len(self.bodies)):
      omega[3*i : 3*(i+1)] = self.get_omega_one_roller()

    # Set RHS = omega - M_rt * force
    RHS = omega - mob.single_wall_mobility_rot_times_force_pycuda(r_vectors_blobs, force, self.eta, self.a)

    # Set linear operator
    system_size = 3 * len(self.bodies)
    def mobility_rot_torque(torque, r_vectors = None, eta = None, a = None):
      return mob.single_wall_mobility_rot_times_torque_pycuda(r_vectors, torque, eta, a)
    linear_operator_partial = partial(mobility_rot_torque, r_vectors = r_vectors_blobs, eta = self.eta, a = self.a)
    A = spla.LinearOperator((system_size, system_size), matvec = linear_operator_partial, dtype='float64')

    # Scale RHS to norm 1
    RHS_norm = np.linalg.norm(RHS)
    if RHS_norm > 0:
      RHS = RHS / RHS_norm

    # Solve linear system # callback=make_callback()
    (sol_precond, info_precond) = spla.gmres(A, 
                                             RHS, 
                                             x0=self.deterministic_torque_previous_step, 
                                             tol=self.tolerance, 
                                             maxiter=1000, 
                                             restart=60,
                                             callback=make_callback()) 
    self.deterministic_torque_previous_step = sol_precond

    # Scale solution with RHS norm
    if RHS_norm > 0:
      sol_precond = sol_precond * RHS_norm

    # Compute linear velocity
    velocity = mob.single_wall_mobility_trans_times_force_torque_pycuda(r_vectors_blobs, force, sol_precond, self.eta, self.a)
    # velocity  = mob.single_wall_mobility_trans_times_force_pycuda(r_vectors_blobs, force, self.eta, self.a, periodic_length = self.periodic_length)
    # velocity += mob.single_wall_mobility_trans_times_torque_pycuda(r_vectors_blobs, sol_precond, self.eta, self.a, periodic_length = self.periodic_length)


    # Set force and torque, this is for testing the mobility functions
    if False:
      force = np.zeros(3 * Nblobs)
      torque = np.zeros(3 * Nblobs)
      
      velocity = mob.single_wall_mobility_trans_times_force_torque_pycuda(r_vectors_blobs, force, torque, self.eta, self.a)
      angular_velocity  = mob.single_wall_mobility_rot_times_force_pycuda(r_vectors_blobs, force, self.eta, self.a)
      angular_velocity += mob.single_wall_mobility_rot_times_torque_pycuda(r_vectors_blobs, torque, self.eta, self.a)      

      print 'force ', force
      print 'torque ', torque

      print 'velocity ', velocity
      print 'omega ', angular_velocity, '\n\n\n'
    
    # Return linear velocity and torque
    return velocity, sol_precond
      

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
      # velocity = mob.single_wall_mobility_trans_times_force_torque_pycuda(r_vectors, force_torque[0:half_size], force_torque[half_size:], eta, a)
      velocity  = mob.single_wall_mobility_trans_times_force_pycuda(r_vectors, force_torque[0:half_size], eta, a, periodic_length = periodic_length)
      velocity += mob.single_wall_mobility_trans_times_torque_pycuda(r_vectors, force_torque[half_size:], eta, a, periodic_length = periodic_length)
      angular_velocity  = mob.single_wall_mobility_rot_times_force_pycuda(r_vectors, force_torque[0:half_size], eta, a, periodic_length = periodic_length)
      angular_velocity += mob.single_wall_mobility_rot_times_torque_pycuda(r_vectors, force_torque[half_size:], eta, a, periodic_length = periodic_length)      
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
                                                                         max_iter = 100,
                                                                         name = 'data/rollers/noise.dat')

    # Compute divergence terms div_t(M_rt) and div_t(M_tt)
    # 1. Generate random displacement
    dx_stoch = np.reshape(np.random.randn(Nblobs * 3), (Nblobs, 3))
    # 2. Displace blobs
    r_vectors_blobs += dx_stoch * (self.rf_delta * 0.5)
    # 3. Compute M_rt(r+0.5*dx) * dx_stoch
    div_M_rt = mob.single_wall_mobility_rot_times_force_pycuda(r_vectors_blobs, np.reshape(dx_stoch, dx_stoch.size), self.eta, self.a, 
                                                                 periodic_length = self.periodic_length)
    div_M_tt = mob.single_wall_mobility_trans_times_force_pycuda(r_vectors_blobs, np.reshape(dx_stoch, dx_stoch.size), self.eta, self.a, 
                                                                 periodic_length = self.periodic_length)
    # 4. Displace blobs in the other direction
    r_vectors_blobs -= dx_stoch * self.rf_delta 
    # 5. Compute -M_rt(r-0.5*dx) * dx_stoch
    div_M_rt -= mob.single_wall_mobility_rot_times_force_pycuda(r_vectors_blobs, np.reshape(dx_stoch, dx_stoch.size), self.eta, self.a, 
                                                                periodic_length = self.periodic_length)
    div_M_tt -= mob.single_wall_mobility_trans_times_force_pycuda(r_vectors_blobs, np.reshape(dx_stoch, dx_stoch.size), self.eta, self.a, 
                                                                  periodic_length = self.periodic_length)
    # Set RHS = -kT*div_t(M_rt) - sqrt(2*kT) * (N^{1/2}*W)_r,
    RHS = -velocities_noise[velocities_noise.size / 2:] - div_M_rt * (self.kT / self.rf_delta)

    # Reset blobs location
    r_vectors_blobs += dx_stoch * (self.rf_delta * 0.5)

    # Set linear operator
    system_size = 3 * len(self.bodies)
    def mobility_rot_torque(torque, r_vectors = None, eta = None, a = None):
      return mob.single_wall_mobility_rot_times_torque_pycuda(r_vectors, torque, eta, a)
    linear_operator_partial = partial(mobility_rot_torque, r_vectors = r_vectors_blobs, eta = self.eta, a = self.a)
    A = spla.LinearOperator((system_size, system_size), matvec = linear_operator_partial, dtype='float64')

    # Scale RHS to norm 1
    RHS_norm = np.linalg.norm(RHS)
    if RHS_norm > 0:
      RHS = RHS / RHS_norm

    # Solve linear system # callback=make_callback()
    (sol_precond, info_precond) = spla.gmres(A, 
                                             RHS, 
                                             tol=self.tolerance, 
                                             maxiter=1000, 
                                             restart=60,
                                             callback=make_callback()) 

    # Scale solution with RHS norm
    if RHS_norm > 0:
      sol_precond = sol_precond * RHS_norm

    # Compute stochastic velocity v_stoch = M_tr * T + sqrt(2*kT) * (N^{1/2}*W)_t + kT*div_t(M_tt).
    v_stoch = mob.single_wall_mobility_trans_times_torque_pycuda(r_vectors_blobs, sol_precond, self.eta, self.a, periodic_length = self.periodic_length)
    v_stoch += velocities_noise[0 : velocities_noise.size / 2] + self.kT * div_M_tt

    return v_stoch
  


  def get_omega_one_roller(self):
    return np.array([0.0, 1.0, 0.0])


# Callback generator
def make_callback():
    closure_variables = dict(counter=0, residuals=[]) 
    def callback(residuals):
        closure_variables["counter"] += 1
        closure_variables["residuals"].append(residuals)
        print closure_variables["counter"], residuals
    return callback


