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

    # Optional variables
    self.calc_slip = None
    self.calc_force_torque = None
    self.mobility_blobs = None
    self.mobility_body = None
    
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
    ''' 
    print 'Integrator starting (gmres)' 
    while True: 
      # print 'bodies\n', self.bodies[0].reference_configuration
      # Compute velocities 

      # Update positions 

      # Check positions if valid return 
      return
      

  def deterministic_forward_euler_dense_algebra(self, dt): 
    ''' 
    Take a time step of length dt using the deterministic forward Euler scheme. 
    The function uses dense algebra methods to solve the equations.
    ''' 
    print 'Integrator starting (dense algebra)' 
    while True: 
      # Calculate slip on blobs
      if self.calc_slip is not None:
        slip = self.calc_slip(self.bodies, self.Nblobs)
      else:
        slip = np.zeros((self.Nblobs, 3))

      # Calculate mobility at the blob level

      # Calculate constraint for due to slip l = M^{-1}*slip

      # Calculate force-torque on bodies

      # Calculate mobility at the body level N

      # Compute velocities

      # Update location orientation

      # Check positions, if valid return

      

      return
      
      
  
    




