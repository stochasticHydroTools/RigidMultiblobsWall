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
                         (quaternions) and returns the torque evaluated there.

      initial_position: vector of quaternions representing the initial configuration
                        of the system.
    '''
    self.mobility = mobility
    self.dim = len(initial_position)
    self.torque_calculator = torque_calculator
    self.position = initial_position

    #TODO: Make this dynamic
    self.kT = 1.0

    
  def FixmanTimeStep(self, dt):
    ''' Take a timestep of length dt using the Fixman method '''
    mobility  = self.mobility(self._position)
    mobility_half = np.linalg.cholesky(mobility)
    torque = self.torque_calculator(self.position)
    noise = np.random.normal(0.0, 1.0, self.dim*4)
    
    omega = mobility*torque + np.sqrt(2.0*self.kT/dt)*mobility_half*noise
    # Update each quaternion
    position_midpoint = []
    for i in range(self.dim):
      quaternion_dt = Quaternion.FromRotation((omega[i:i+4])*dt/2.)
      positon_midpoint.append(quaternion_dt*self.position[i])
      
    mobility_tilde = self.mobility(position_midpoint)
    torque_tilde = self.torque_calculator(position_midpoint)
    
    

    

    
    
