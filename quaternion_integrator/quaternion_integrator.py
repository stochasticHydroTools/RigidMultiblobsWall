'''
Simple integrator for N quaternions.
'''
import numpy as np
from quaternion import Quaternion

class QuaternionIntegrator(object):
  '''
  Integrator that timesteps using Fixman quaternion updates.
  '''
  def __init__(self, mobility, initial_orientation, torque_calculator, 
               has_location = False, initial_location = None,
               force_calculator = None):
    '''
    Set up components of the integrator.  
    args: 
    
      mobility: function that takes a vector of positions (quaternions) 
                and returns a matrix of the mobility evaluated there.

      torque_calculator: function that takes a vector of positions
                         (quaternions) and returns the torque evaluated there as 
                         a numpy array where the first three components are the 
                         torque on the first quaternion, etc.

      initial_orientation: vector of quaternions representing the initial 
                           configuration of the system.
    
      has_location: boolean indicating whether we keep location
                    as well as orientation.
    '''
    self.mobility = mobility
    self.dim = len(initial_orientation)
    self.torque_calculator = torque_calculator
    self.orientation = initial_orientation
    self.has_location = has_location
    self.location = initial_location
    self.force_calculator = force_calculator
    
    #TODO, if we use location, check that we have a force calculator and
    # an iniital location.
    
    self.rf_delta = 1e-8  # delta for RFD term in RFD step

    #TODO: Make this dynamic
    self.kT = 1.0

  def fixman_time_step(self, dt):
    ''' Take a timestep of length dt using the Fixman method '''
    if self.has_location:
      # Handle integrator with location as well.
      mobility  = self.mobility(self.location, self.orientation)
      mobility_half = np.linalg.cholesky(mobility)
      noise = np.random.normal(0.0, 1.0, self.dim*6)
      force = self.force_calculator(self.location, self.orientation)
      torque = self.torque_calculator(self.location, self.orientation)
      velocity_and_omega = (np.dot(mobility, np.concatenate([force, torque])) +
                            np.sqrt(4.0*self.kT/dt)*
                            np.dot(mobility_half, noise))
      velocity = velocity_and_omega[0:(3*self.dim)]
      omega = velocity_and_omega[(3*self.dim):(6*self.dim)]
    else:
      mobility  = self.mobility(self.orientation)
      mobility_half = np.linalg.cholesky(mobility)
      noise = np.random.normal(0.0, 1.0, self.dim*3)
      torque = self.torque_calculator(self.orientation)
      omega = (np.dot(mobility, torque) + 
               np.sqrt(4.0*self.kT/dt)*np.dot(mobility_half, noise))

    # Update each quaternion at a time.
    orientation_midpoint = []
    for i in range(self.dim):
      quaternion_dt = Quaternion.from_rotation((omega[(i*3):(i*3+3)])*dt/2.)
      orientation_midpoint.append(quaternion_dt*self.orientation[i])
      
    if self.has_location:
      location_midpoint = self.location + 0.5*dt*velocity
    
    if self.has_location:
      mobility_tilde = self.mobility(location_midpoint, orientation_midpoint)
      noise = noise + np.random.normal(0.0, 1.0, self.dim*6)
      force_tilde = self.force_calculator(location_midpoint, orientation_midpoint)
      torque_tilde = self.torque_calculator(location_midpoint, orientation_midpoint)
      mobility_half_inv = np.linalg.inv(mobility_half)
      velocity_and_omega_tilde = (
        np.dot(mobility_tilde, 
               np.concatenate([force_tilde, torque_tilde])) + np.sqrt(self.kT/dt)*
        np.dot(mobility_tilde, np.dot(mobility_half_inv, noise)))
      velocity_tilde = velocity_and_omega_tilde[0:(3*self.dim)]
      omega_tilde = velocity_and_omega_tilde[(3*self.dim):(6*self.dim)]
    else:
      mobility_tilde = self.mobility(orientation_midpoint)
      noise = noise + np.random.normal(0.0, 1.0, self.dim*3)
      torque_tilde = self.torque_calculator(orientation_midpoint)
      mobility_half_inv = np.linalg.inv(mobility_half)
      omega_tilde = (
        np.dot(mobility_tilde, torque_tilde) + np.sqrt(self.kT/dt)*
        np.dot(mobility_tilde, np.dot(mobility_half_inv, noise)))
    
    new_orientation = []
    for i in range(self.dim):
      quaternion_dt = Quaternion.from_rotation((omega_tilde[(i*3):(i*3+3)])*dt)
      new_orientation.append(quaternion_dt*self.orientation[i])
      
    if self.has_location:
      new_location = self.location + dt*velocity_tilde
      self.location = new_location

    self.orientation = new_orientation


  def rfd_time_step(self, dt):
    ''' Take a timestep of length dt using the RFD method '''
    if self.has_location:
      # Handle integrator with location as well.
      mobility  = self.mobility(self.location, self.orientation)
      mobility_half = np.linalg.cholesky(mobility)
      rfd_noise = np.random.normal(0.0, 1.0, self.dim*6)
      # Calculate RFD location.
      rfd_location = self.location + self.rf_delta*rfd_noise[0:3*self.dim]
      # Update each quaternion at a time for RFD orientation.
      rfd_orientation = []
      for i in range(self.dim):
        quaternion_dt = Quaternion.from_rotation((self.rf_delta*
                                                  rfd_noise[(3*self.dim + i*3):
                                                            (3*self.dim + i*3+3)]))
        rfd_orientation.append(quaternion_dt*self.orientation[i])

      # divergence term d_x(N) : \Psi^T 
      divergence_term = self.kT*np.dot(
        (self.mobility(rfd_location, rfd_orientation) - mobility),
        rfd_noise/self.rf_delta)

      noise = np.random.normal(0.0, 1.0, self.dim*6)
      force = self.force_calculator(self.location, self.orientation)
      torque = self.torque_calculator(self.location, self.orientation)
      velocity_and_omega = (np.dot(mobility, np.concatenate([force, torque])) +
                            np.sqrt(2.0*self.kT/dt)*
                            np.dot(mobility_half, noise) +
                            divergence_term)
      velocity = velocity_and_omega[0:(3*self.dim)]
      omega = velocity_and_omega[(3*self.dim):(6*self.dim)]
      
      new_location = self.location + dt*velocity
      self.location = new_location

    else:
      rfd_noise = np.random.normal(0.0, 1.0, self.dim*3)
      mobility  = self.mobility(self.orientation)
      mobility_half = np.linalg.cholesky(mobility)
      torque = self.torque_calculator(self.orientation)
      noise = np.random.normal(0.0, 1.0, self.dim*3)
      # Update each quaternion at a time for rfd orientation.
      rfd_orientation = []
      for i in range(self.dim):
        quaternion_dt = Quaternion.from_rotation((self.rf_delta*
                                                  rfd_noise[(i*3):(i*3+3)]))
        rfd_orientation.append(quaternion_dt*self.orientation[i])

      # divergence term d_x(M) : \Psi^T 
      divergence_term = self.kT*np.dot(
        (self.mobility(rfd_orientation) - mobility),
        rfd_noise/self.rf_delta)
      omega = (np.dot(mobility, torque) + 
               np.sqrt(2.*self.kT/dt)*
               np.dot(mobility_half, noise) +
               divergence_term)

    # For with location and without location, we update orientation the same way.

    new_orientation = []
    for i in range(self.dim):
      quaternion_dt = Quaternion.from_rotation((omega[(i*3):(i*3+3)])*dt)
      new_orientation.append(quaternion_dt*self.orientation[i])

    self.orientation = new_orientation

    
  def additive_em_time_step(self, dt):
    ''' 
    Take a simple Euler Maruyama step assuming that the mobility is
    constant.  This for testing and debugging.  We also use it to make sure
    that we need the drift for the correct distribution, etc.
    '''
    mobility = self.mobility(self.orientation)
    mobility_half = np.linalg.cholesky(mobility)
    torque = self.torque_calculator(self.orientation)
    noise = np.random.normal(0.0, 1.0, self.dim*3)
    omega = (np.dot(mobility, torque) + 
             np.sqrt(2.0*self.kT/dt)*np.dot(mobility_half, noise))
    new_orientation = []
    for i in range(self.dim):
      quaternion_dt = Quaternion.from_rotation((omega[(i*3):(i*3+3)])*dt)
      new_orientation.append(quaternion_dt*self.orientation[i])
      
    self.orientation = new_orientation

  def estimate_divergence(self):
    ''' 
    Estimate the divergence term with a deterministic
    finite difference approach. This is for a single quaternion.
    '''
    delta = 1e-6
    div_term = np.zeros(3)
    for k in range(3):
      omega = np.zeros(3)
      omega[k] = 1.
      quaternion_dt = Quaternion.from_rotation(omega*delta/2.)
      quaternion_tilde_1 = quaternion_dt*self.orientation[0]
      quaternion_dt = Quaternion.from_rotation(-1.*omega*delta/2.)
      quaternion_tilde_2 = quaternion_dt*self.orientation[0]
      div_term += np.inner((self.mobility([quaternion_tilde_1]) -
                            self.mobility([quaternion_tilde_2])),
                           omega/delta)
      
    return div_term

    
  def estimate_drift(self, dt, n_steps, scheme):
    ''' Emperically estimate the drift term in the absence of torque. 
    For now this is just without location.  TODO: add location.'''
    old_torque = self.torque_calculator
    if self.has_location:
      def zero_torque(orientation, location):
        return np.zeros(3*len(orientation))
      def zero_force(orientation, location):
        return np.zeros(3*len(orientation))
      old_force = self.force_calculator
      self.force_calculator = zero_force
      initial_location = self.location
    else:
      def zero_torque(orientation):
        return np.zeros(3*len(orientation))
    self.torque_calculator = zero_torque
    initial_orientation = self.orientation

    drift_samples = []
    for k in range(n_steps):
      if scheme == 'FIXMAN':
        self.fixman_time_step(dt)
      elif scheme == 'RFD':
        self.rfd_time_step(dt)
      else:
        raise Exception('scheme must be FIXMAN or RFD for drift estimation.')
      # For now, hard code to 1 dimensional integrator.
      for k in range(self.orientation):
        orientation_increment = self.orientation[k]*initial_orientation[k].inverse()
        drift_angle = orientation_increment.rotation_angle()
      drift_samples.append(drift_angle)
      self.orientation = initial_orientation

    avg_drift = np.mean(drift_samples)/dt
    std_dev = np.std(drift_samples)/np.sqrt(n_steps)/dt
    
    return [avg_drift, std_drift]
      
      
      
    
