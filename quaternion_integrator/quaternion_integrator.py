'''
Simple integrator for N quaternions.
'''
import numpy as np
from quaternion import Quaternion

class QuaternionIntegrator(object):
  '''
  Integrator that timesteps using Fixman quaternion updates.
  '''
  def __init__(self, mobility, initial_position, torque_calculator):
    '''
    Set up components of the integrator.  
    args: 
    
      mobility: function that takes a vector of positions (quaternions) 
                and returns a matrix of the mobility evaluated there.

      torque_calculator: function that takes a vector of positions
                         (quaternions) and returns the torque evaluated there as 
                         a numpy array where the first three components are the 
                         torque on the first quaternion, etc.

      initial_position: vector of quaternions representing the initial configuration
                        of the system.
    '''
    self.mobility = mobility
    self.dim = len(initial_position)
    self.torque_calculator = torque_calculator
    self.position = initial_position
    self.path = [self.position]   # Save the trajectory.

    self.rf_delta = 1e-6  # delta for RFD term in RFD step

    #TODO: Make this dynamic
    self.kT = 1.0

    
  def fixman_time_step(self, dt):
    ''' Take a timestep of length dt using the Fixman method '''
    mobility  = self.mobility(self.position)
    mobility_half = np.linalg.cholesky(mobility)
    torque = self.torque_calculator(self.position)
    noise = np.random.normal(0.0, 1.0, self.dim*3)
    omega = (np.dot(mobility, torque) + 
             np.sqrt(2.0*self.kT/dt)*np.dot(mobility_half, noise))

    # Update each quaternion at a time.
    position_midpoint = []
    for i in range(self.dim):
      quaternion_dt = Quaternion.from_rotation((omega[(i*3):(i*3+3)])*dt/2.)
      position_midpoint.append(quaternion_dt*self.position[i])
      
    mobility_tilde = self.mobility(position_midpoint)
    torque_tilde = self.torque_calculator(position_midpoint)
    mobility_half_inv = np.linalg.inv(mobility_half)
    omega_tilde = (np.dot(mobility_tilde, torque_tilde) +
                   np.sqrt(2*self.kT/dt)*
                   np.dot(mobility_tilde, np.dot(mobility_half_inv, noise)))
    
    new_position = []
    for i in range(self.dim):
      quaternion_dt = Quaternion.from_rotation((omega_tilde[(i*3):(i*3+3)])*dt)
      new_position.append(quaternion_dt*self.position[i])
      
    self.position = new_position
    self.path.append(new_position)


  def rfd_time_step(self, dt):
    ''' Take a timestep of length dt using the RFD method '''
    rfd_noise = self.rf_delta*np.random.normal(0.0, 1.0, self.dim*3)
    mobility  = self.mobility(self.position)
    mobility_half = np.linalg.cholesky(mobility)
    torque = self.torque_calculator(self.position)
    noise = np.random.normal(0.0, 1.0, self.dim*3)

    # Update each quaternion at a time for rfd position.
    rfd_position = []
    for i in range(self.dim):
      quaternion_dt = Quaternion.from_rotation((rfd_noise[(i*3):(i*3+3)]))
      rfd_position.append(quaternion_dt*self.position[i])

    # divergence term d_x(M) : \Psi^T 
    divergence_term = self.kT*np.dot(
      self.mobility(rfd_position) - mobility,
      rfd_noise/self.rf_delta)
    print "divergence term is ", divergence_term
    omega = (np.dot(mobility, torque) + 
             np.sqrt(2.*self.kT/dt)*
             np.dot(mobility_half, noise) +
             divergence_term)
                        
    new_position = []
    for i in range(self.dim):
      quaternion_dt = Quaternion.from_rotation((omega[(i*3):(i*3+3)])*dt)
      new_position.append(quaternion_dt*self.position[i])
    
    self.position = new_position
    self.path.append(new_position)

    
  def additive_em_time_step(self, dt):
    ''' 
    Take a simple Euler Maruyama step assuming that the mobility is
    constant.  This for testing and debugging.  We also use it to make sure
    that we need the drift for the correct distribution, etc.
    '''
    mobility = self.mobility(self.position)
    mobility_half = np.linalg.cholesky(mobility)
    torque = self.torque_calculator(self.position)
    noise = np.random.normal(0.0, 1.0, self.dim*3)
    omega = (np.dot(mobility, torque) + 
             np.sqrt(2.0*self.kT/dt)*np.dot(mobility_half, noise))
    new_position = []
    for i in range(self.dim):
      quaternion_dt = Quaternion.from_rotation((omega[(i*3):(i*3+3)])*dt)
      new_position.append(quaternion_dt*self.position[i])
      
    self.position = new_position
    self.path.append(new_position)


    

    
    
